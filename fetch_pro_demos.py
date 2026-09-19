r"""从 HLTV 批量抓职业赛 demo（每张地图凑够 N 场）。

背景：HLTV 上每场比赛有一个"Demo download"压缩包（.rar），里面是这场**所有
地图**的 .dem。所以按"比赛"抓、按"地图"存，是最高效的做法：
一场 BO3 里有 Mirage 或 Dust2 就值得下。

流程：
  1. 翻 `hltv.org/results?content=demo` 的列表页，收集候选比赛；
  2. 逐场读比赛页，取地图列表 + demo 下载 id（`data-demo-link`）；
  3. 只下"地图列表里含 de_dust2 / de_mirage"且该图还没凑够的比赛；
  4. 下载 .rar（支持断点续传）→ 用本机 7z 解压 → 读每个 .dem 的 header；
  5. 需要的地图归档到 `demos\<地图>\`，其余删掉；写 `demos\manifest.json`。

用法：
    .venv\Scripts\python.exe fetch_pro_demos.py --per-map 10
    .venv\Scripts\python.exe fetch_pro_demos.py --per-map 10 --max-matches 40
    .venv\Scripts\python.exe fetch_pro_demos.py --per-map 2 --dry-run   # 只看候选

注意：HLTV 页面本身不要求登录；下载会 302 跳到 r2-demos.hltv.org 的对象存储。
脚本对每次请求之间留了间隔，避免给对方压力。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from curl_cffi import requests
from demoparser2 import DemoParser

ROOT = Path(__file__).resolve().parent
HLTV = "https://www.hltv.org"
MIRRORED_MAPS = {"de_dust2", "de_mirage"}
MAP_LABEL = {"de_dust2": "Dust2", "de_mirage": "Mirage"}
# 顶级赛事关键词（--tier top 时用来筛赛事层级）
TOP_EVENT_KEYWORDS = ("starladder", "esl", "blast", "iem", "pgl", "major",
                      "dreamhack", "cct", "esports world cup")

# 本机随其他软件安装的 7z，用来解 .rar（7-Zip 自带 RAR 解码器）
SEVEN_ZIP_CANDIDATES = [
    r"C:\Program Files\5EClient\resources\7z.exe",
    r"C:\Program Files\NVIDIA Corporation\NVIDIA App\7z.exe",
    r"C:\Program Files\D5\D5 Launcher\7z.exe",
    r"C:\Program Files\Chaos Group\V-Ray\Swarm 1.4\node_modules\p7zip-bin-win32\7z.exe",
    r"C:\Program Files\Autodesk\AdODIS\V1\Setup\7za.exe",
]


def find_7z() -> str:
    for cand in SEVEN_ZIP_CANDIDATES:
        if Path(cand).exists():
            return cand
    for name in ("7z", "7za", "unrar"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("找不到能解 RAR 的工具（7z / unrar），先装一个 7-Zip。")


@dataclass
class MatchInfo:
    url: str
    match_id: str
    maps: list[str]
    demo_id: str | None
    event: str = ""
    date: str = ""
    teams: str = ""
    demos: list[dict] = field(default_factory=list)


class Hltv:
    def __init__(self, delay: float = 1.5, cookies: dict | None = None,
                 user_agent: str | None = None) -> None:
        # cf_clearance 与 User-Agent 绑定，所以必须沿用浏览器那次会话的 UA
        self.impersonate = "edge101"
        self.headers = {"Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}
        if user_agent:
            self.headers["User-Agent"] = user_agent
        self.s = requests.Session(impersonate=self.impersonate, headers=self.headers)
        if cookies:
            self.s.cookies.update(cookies)
        self.delay = delay

    def get(self, url: str, **kw):
        time.sleep(self.delay)
        r = self.s.get(url, impersonate=self.impersonate, timeout=60, **kw)
        if r.status_code == 403 and "hltv.org" in url:
            raise SystemExit(
                "被 Cloudflare 拦住了（403）。cookie 可能已过期，重新跑一次：\n"
                "  .venv\\Scripts\\python.exe hltv_session.py --no-headless")
        return r

    def results(self, offset: int = 0) -> list[str]:
        r = self.get(f"{HLTV}/results?content=demo&offset={offset}")
        if r.status_code != 200:
            return []
        return list(dict.fromkeys(re.findall(r'href="(/matches/\d+/[^"]+)"', r.text)))

    def match(self, link: str) -> MatchInfo | None:
        r = self.get(HLTV + link)
        if r.status_code != 200:
            return None
        text = r.text
        maps = re.findall(r'class="mapname">\s*([^<]+?)\s*<', text)
        demo = re.search(r'data-demo-link="(/download/demo/(\d+))"', text)
        event = re.search(r'class="event text-ellipsis">\s*([^<]+?)\s*<', text)
        date = re.search(r'class="date"[^>]*>\s*([^<]+?)\s*<', text)
        teams = re.findall(r'class="teamName[^"]*">\s*([^<]+?)\s*<', text)
        match_id = re.search(r"/matches/(\d+)/", link)
        return MatchInfo(
            url=HLTV + link,
            match_id=match_id.group(1) if match_id else link,
            maps=[m.strip() for m in maps],
            demo_id=demo.group(2) if demo else None,
            event=event.group(1) if event else "",
            date=date.group(1) if date else "",
            teams=" vs ".join(dict.fromkeys(teams))[:60],
        )


def download(url: str, dest: Path, cookies: dict, user_agent: str | None,
             chunk: int = 1 << 20) -> Path:
    """带断点续传的流式下载。

    两个坑（都实测过）：
      1. curl_cffi 的 Response 不支持 `with`，必须手动 close；
      2. 下载不能复用"浏览用"的 session —— 它会带上被刷新过的 __cf_bm 等 cookie，
         反而被拦。这里每次都用固定的 cookie + UA 直连，并且先自己解析 302，
         再直接向 r2-demos.hltv.org 取文件（R2 **必须**带 cookie，否则 403）。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": user_agent} if user_agent else {}

    if "r2-demos.hltv.org" not in url:
        r = requests.get(url, impersonate="edge101", timeout=120, cookies=cookies,
                         headers={**headers, "Referer": HLTV + "/"},
                         allow_redirects=False)
        r.close()
        if r.status_code in (301, 302, 303, 307, 308):
            url = r.headers["location"]
        elif r.status_code == 403:
            raise SystemExit("HLTV 下载入口 403：重跑 hltv_session.py 刷新 cookie。")
        elif r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}（下载入口）")

    have = dest.stat().st_size if dest.exists() else 0
    range_headers = {"Range": f"bytes={have}-"} if have else {}
    r = requests.get(url, impersonate="edge101", timeout=3600, stream=True,
                     cookies=cookies, headers={**headers, **range_headers})
    try:
        if have and r.status_code != 206:
            have = 0  # 服务器不支持续传，重来
            if dest.exists():
                dest.unlink()
        if r.status_code == 403:
            raise SystemExit(
                "下载被 Cloudflare 拦住（403）：重跑 hltv_session.py 刷新 cookie 后继续，"
                "已下载的部分会自动续传。")
        if r.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {r.status_code}")
        mode = "ab" if have else "wb"
        written = have
        mark = written
        with open(dest, mode) as fh:
            for block in r.iter_content(chunk_size=chunk):
                fh.write(block)
                written += len(block)
                if written - mark > 50 * 1024 * 1024:
                    mark = written
                    print(f"    ... {written/1024/1024:.0f} MB", flush=True)
    finally:
        r.close()
    return dest


def extract(seven_zip: str, archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([seven_zip, "x", "-y", f"-o{out_dir}", str(archive)],
                          capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"7z 解压失败：{proc.stdout[-300:]}{proc.stderr[-300:]}")


def probe_demo(path: Path) -> dict:
    parser = DemoParser(str(path))
    header = parser.parse_header()
    events = parser.parse_events(["round_start", "player_death"])
    counts = {name: frame.shape[0] for name, frame in events}
    return {
        "map_name": header.get("map_name"),
        "rounds": counts.get("round_start", 0),
        "kills": counts.get("player_death", 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-map", type=int, default=10, help="每张地图要几场（默认 10）")
    ap.add_argument("--max-matches", type=int, default=60, help="最多检查多少场比赛")
    ap.add_argument("--min-rounds", type=int, default=13, help="低于这个回合数视为不完整")
    ap.add_argument("--pages", type=int, default=8, help="翻多少页 results 列表（每页 100 场）")
    ap.add_argument("--demos-dir", type=Path, default=Path("demos"))
    ap.add_argument("--work-dir", type=Path, default=Path(".incoming"))
    ap.add_argument("--cookies", type=Path, default=Path("_incoming/hltv_cookies.json"),
                    help="hltv_session.py 导出的 cookie/UA 文件")
    ap.add_argument("--dry-run", action="store_true", help="只列候选，不下载")
    ap.add_argument("--tier", choices=("all", "top"), default="all",
                    help="赛事层级：all=全部正式比赛（默认），top=只取顶级赛事")
    ap.add_argument("--no-auto-refresh", action="store_true",
                    help="被 Cloudflare 拦时不要自动重开浏览器续期 cookie")
    args = ap.parse_args()

    seven_zip = find_7z()
    print(f"解压工具：{seven_zip}")

    def refresh_cookie() -> tuple[dict, str | None]:
        """cookie 失效时自动用本机 Edge 重新过一次 Cloudflare。"""
        if args.no_auto_refresh:
            return {}, None
        print("  cookie 失效，自动重开浏览器过一次 Cloudflare ...", flush=True)
        import hltv_session
        jar, ua = hltv_session.refresh_session(quiet=True)
        print(f"  已刷新 {len(jar)} 个 cookie（含 cf_clearance="
              f"{'cf_clearance' in jar}）", flush=True)
        return jar, ua

    cookies, user_agent = {}, None
    cookie_path = args.cookies if args.cookies.is_absolute() else ROOT / args.cookies
    if cookie_path.exists():
        payload = json.loads(cookie_path.read_text(encoding="utf-8"))
        cookies = payload.get("cookies", payload)
        user_agent = payload.get("user_agent")
        print(f"使用 cookie：{cookie_path.name}（{len(cookies)} 个，"
              f"含 cf_clearance={'cf_clearance' in cookies}）")
    else:
        print(f"提示：没有找到 {cookie_path}，先跑 hltv_session.py 过一次 Cloudflare。")
    hltv = Hltv(cookies=cookies, user_agent=user_agent)

    have = {m: 0 for m in MIRRORED_MAPS}
    kept = {m: [] for m in MIRRORED_MAPS}
    manifest_path = args.demos_dir / "manifest.json"
    known_files = set()
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in old.get("demos", []):
            if item.get("map_name") in have:
                have[item["map_name"]] += 1
                kept[item["map_name"]].append(item)
                known_files.add(item.get("file"))
        print(f"已有职业 demo：{ {k: v for k, v in have.items()} }（从 manifest.json 读）")

    # 也认一下磁盘上已经在 demos\<地图>\ 里的文件（脚本中途中断过也能续上）
    for map_name in MIRRORED_MAPS:
        for dem in sorted((args.demos_dir / map_name).glob("*.dem")):
            rel = str(dem.resolve().relative_to(ROOT))
            if rel in known_files:
                continue
            try:
                probe = probe_demo(dem)
            except Exception as exc:  # noqa: BLE001
                print(f"  ！已有文件解析失败，忽略：{dem.name}（{exc}）")
                continue
            have[map_name] += 1
            kept[map_name].append({
                "map_name": map_name, "file": rel,
                "match_id": dem.name.split("_")[0], "match_url": "",
                "teams": "", "event": "", "date": "",
                "rounds": probe["rounds"], "kills": probe["kills"],
                "size_bytes": dem.stat().st_size, "source": "HLTV 职业赛",
            })
            known_files.add(rel)
    if any(kept.values()):
        print(f"  磁盘上已有：{ {k: len(v) for k, v in kept.items()} }")

    candidates: list[MatchInfo] = []
    seen: set[str] = set()
    for page in range(args.pages):
        links = hltv.results(offset=page * 100)
        if not links:
            break
        print(f"列表页 offset={page * 100}：{len(links)} 场")
        for link in links:
            if link in seen:
                continue
            seen.add(link)
            if len(candidates) >= args.max_matches:
                break
            info = hltv.match(link)
            if info is None or not info.demo_id:
                continue
            if args.tier == "top" and not any(
                    k in info.event.lower() for k in TOP_EVENT_KEYWORDS):
                continue
            wanted = [m for m in MIRRORED_MAPS
                      if MAP_LABEL[m] in info.maps and have[m] < args.per_map]
            if not wanted:
                continue
            info.demos = wanted
            candidates.append(info)
            print(f"  候选 {info.match_id} {info.teams} [{', '.join(info.maps)}] "
                  f"缺 {wanted}")
        if all(have[m] + sum(1 for c in candidates if m in c.demos) >= args.per_map
               for m in MIRRORED_MAPS):
            break
        if len(candidates) >= args.max_matches:
            break

    print(f"\n共找到 {len(candidates)} 场候选比赛")
    if args.dry_run:
        for c in candidates:
            print(f"  {c.match_id}  {c.url}\n     地图 {c.maps}  需要 {c.demos}")
        return 0
    if not candidates:
        print("没有候选。可以加大 --pages / --max-matches，或稍后再试。")
        return 1

    args.work_dir.mkdir(parents=True, exist_ok=True)
    for idx, info in enumerate(candidates, 1):
        missing = [m for m in info.demos if have[m] < args.per_map]
        if not missing:
            continue
        slug = re.sub(r"[^\w]+", "-", info.url.rsplit("/", 1)[-1])[:60]
        archive = args.work_dir / f"{info.match_id}-{slug}.rar"
        print(f"\n[{idx}/{len(candidates)}] {info.match_id} {info.teams} "
              f"→ 需要 {missing}")
        out_dir = args.work_dir / info.match_id
        dem_files: list[Path] = []
        for attempt in (1, 2):
            try:
                if not archive.exists() or archive.stat().st_size < 10_000_000:
                    download(f"{HLTV}/download/demo/{info.demo_id}", archive,
                             cookies, user_agent)
                size_mb = archive.stat().st_size / 1024 / 1024
                if attempt == 1:
                    print(f"    下载完成 {size_mb:.1f} MB")
                if not out_dir.exists() or not list(out_dir.glob("*.dem")):
                    extract(seven_zip, archive, out_dir)
                dem_files = sorted(out_dir.rglob("*.dem"))
                print(f"    解出 {len(dem_files)} 个 .dem")
                break
            except SystemExit as exc:
                if attempt == 1 and not args.no_auto_refresh and "403" in str(exc):
                    cookies, user_agent = refresh_cookie()
                    hltv = Hltv(cookies=cookies, user_agent=user_agent)
                    continue
                print(f"    !! 失败：{exc}")
                break
            except Exception as exc:  # noqa: BLE001
                print(f"    !! 失败：{type(exc).__name__} {exc}")
                break
        if not dem_files:
            continue

        for dem in dem_files:
            try:
                probe = probe_demo(dem)
            except Exception as exc:  # noqa: BLE001
                print(f"    !! {dem.name} 解析失败：{exc}")
                continue
            map_name = probe["map_name"]
            if map_name not in missing:
                continue
            if probe["rounds"] < args.min_rounds:
                print(f"    ~ {dem.name} 只有 {probe['rounds']} 回合，判为不完整，跳过")
                continue
            target_dir = args.demos_dir / map_name
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{info.match_id}_{dem.name}"
            shutil.move(str(dem), str(target))
            have[map_name] += 1
            record = {
                "map_name": map_name,
                "file": str(target.resolve().relative_to(ROOT)),
                "match_id": info.match_id,
                "match_url": info.url,
                "teams": info.teams,
                "event": info.event,
                "date": info.date,
                "rounds": probe["rounds"],
                "kills": probe["kills"],
                "size_bytes": target.stat().st_size,
                "source": "HLTV 职业赛",
            }
            kept[map_name].append(record)
            print(f"    ✓ {map_name}：{target.name}（{probe['rounds']} 回合 / "
                  f"{probe['kills']} 击杀）→ 已有 {have[map_name]}/{args.per_map}")

        # 归档压缩包与剩余文件，省磁盘
        try:
            archive.unlink()
            shutil.rmtree(out_dir, ignore_errors=True)
        except OSError:
            pass
        manifest_path.write_text(json.dumps(
            {"demos": kept["de_dust2"] + kept["de_mirage"], "per_map_target": args.per_map},
            ensure_ascii=False, indent=2), encoding="utf-8")
        if all(have[m] >= args.per_map for m in MIRRORED_MAPS):
            print("\n两张图都凑够了。")
            break

    print(f"\n最终：de_dust2 {have['de_dust2']}/{args.per_map}，"
          f"de_mirage {have['de_mirage']}/{args.per_map}")
    print(f"清单写入 {manifest_path}")
    print("下一步：.venv\\Scripts\\python.exe ingest_demos.py --demos demos --source pro")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
