r"""用本机 Edge（无头、独立配置目录）过一次 Cloudflare，把通行 cookie 导出给脚本用。

为什么需要：HLTV 的页面会被 Cloudflare 的 JS 挑战拦住，纯 HTTP 客户端（curl_cffi
换指纹也一样）会拿到 403。而真实浏览器能自动通过挑战并拿到 `cf_clearance` cookie。
本脚本：

  1. 用项目内的独立用户目录启动 Edge（无头），打开 HLTV 列表页；
  2. 通过 CDP 等页面标题不再是 "Just a moment..." 为止（即挑战已过）；
  3. 导出该浏览器上下文里的 cookies 到 `_incoming/hltv_cookies.json`；
  4. 立刻用 curl_cffi 带这组 cookie 验证一次，确认能拿到 200。

之后 `fetch_pro_demos.py --cookies _incoming/hltv_cookies.json` 就能正常抓取。

用法：
    .venv\Scripts\python.exe hltv_session.py
    .venv\Scripts\python.exe hltv_session.py --no-headless   # 万一无头被识别，弹出窗口跑

说明：
  * 浏览器的用户数据目录写在项目内（`.browser-profile`），不动你平时用的浏览器；
  * 浏览器进程用完即关，cookie 文件也在项目内。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import websocket
from curl_cffi import requests

ROOT = Path(__file__).resolve().parent
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
TARGET = "https://www.hltv.org/results?content=demo"


def launch(profile: Path, port: int, headless: bool) -> subprocess.Popen:
    args = [
        str(EDGE),
        f"--user-data-dir={profile}",
        f"--remote-debugging-port={port}",
        "--no-first-run", "--no-default-browser-check",
        "--disable-features=Translate,OptimizationHints",
        "--window-size=1200,900",
    ]
    if headless:
        args.append("--headless=new")
    else:
        # 非无头模式：把窗口挪到屏幕外，避免打扰用户，但仍然是"真实浏览器"
        args += ["--window-position=-32000,-32000", "--window-size=1280,900"]
    args.append("about:blank")
    return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ws_url(port: int, timeout: float = 30) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/json/version", timeout=3)
            if r.status_code == 200:
                return r.json()["webSocketDebuggerUrl"]
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    raise SystemExit("连不上 Edge 的调试端口")


class Cdp:
    def __init__(self, url: str) -> None:
        self.ws = websocket.create_connection(url, timeout=40,
                                              suppress_origin=True)
        self.n = 0

    def call(self, method: str, params: dict | None = None, session: str | None = None):
        self.n += 1
        msg = {"id": self.n, "method": method, "params": params or {}}
        if session:
            msg["sessionId"] = session
        self.ws.send(json.dumps(msg))
        while True:
            data = json.loads(self.ws.recv())
            if data.get("id") == self.n:
                if "error" in data:
                    raise RuntimeError(f"{method}: {data['error']}")
                return data.get("result", {})


def refresh_session(port: int = 9333, profile: Path | None = None,
                    out: Path | None = None, wait: float = 45,
                    headless: bool = False, quiet: bool = False) -> tuple[dict, str]:
    """跑一次浏览器过挑战，导出 (cookies, user_agent) 并落盘。可被其他脚本调用。"""
    profile = profile or (ROOT / ".browser-profile")
    out = out or (ROOT / "_incoming" / "hltv_cookies.json")

    def say(msg: str) -> None:
        if not quiet:
            print(msg)

    out.parent.mkdir(parents=True, exist_ok=True)
    proc = launch(profile, port, headless)
    say(f"Edge 已启动（profile={profile}）")
    try:
        cdp = Cdp(ws_url(port))
        target = cdp.call("Target.createTarget", {"url": TARGET})
        session = cdp.call("Target.attachToTarget",
                           {"targetId": target["targetId"], "flatten": True})["sessionId"]
        cdp.call("Page.enable", session=session)

        challenge_words = ("just a moment", "请稍候", "attention required",
                           "checking your browser", "正在检查")

        def page_state() -> tuple[str, int, str]:
            # 用 JSON 返回：标题里可能自带 "|"，用分隔符拼字符串会解析错
            expr = ("JSON.stringify({t: document.title, c: document.querySelectorAll("
                    "'a[href^=\"/matches/\"]').length, ua: navigator.userAgent})")
            res = cdp.call("Runtime.evaluate",
                           {"expression": expr, "returnByValue": True},
                           session=session)
            raw = res.get("result", {}).get("value") or "{}"
            try:
                data = json.loads(raw)
            except Exception:  # noqa: BLE001
                return "", 0, ""
            return data.get("t", ""), int(data.get("c", 0)), data.get("ua", "")

        passed = False
        deadline = time.time() + wait
        waited = 0.0
        while time.time() < deadline:
            time.sleep(2)
            waited += 2
            try:
                title, count, user_agent = page_state()
            except Exception:  # noqa: BLE001
                continue
            say(f"  [{waited:>4.0f}s] 标题={title!r}  比赛链接={count}")
            if count > 0:
                passed = True
                break
        if not passed:
            say("  警告：还没看到比赛链接，挑战可能未通过")

        cookies = cdp.call("Network.getCookies", {"urls": [TARGET]}, session=session)["cookies"]
        jar = {c["name"]: c["value"] for c in cookies}
        payload = {"cookies": jar, "user_agent": user_agent, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        say(f"导出 {len(jar)} 个 cookie -> {out}")
        say(f"  含 cf_clearance: {'cf_clearance' in jar}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()
        say("Edge 已关闭")
    return jar, user_agent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=9333)
    ap.add_argument("--profile", type=Path, default=ROOT / ".browser-profile")
    ap.add_argument("--out", type=Path, default=ROOT / "_incoming" / "hltv_cookies.json")
    ap.add_argument("--wait", type=float, default=45, help="最多等多少秒过挑战")
    ap.add_argument("--no-headless", action="store_true")
    args = ap.parse_args()
    if not EDGE.exists():
        raise SystemExit(f"找不到 Edge：{EDGE}")

    jar, user_agent = refresh_session(args.port, args.profile, args.out, args.wait,
                                      headless=not args.no_headless)

    # 立刻验证：cf_clearance 与 UA 绑定，所以必须带同一个 UA
    for imp in ("edge101", "chrome"):
        verify = requests.get(TARGET, impersonate=imp, timeout=45, cookies=jar,
                              headers={"User-Agent": user_agent,
                                       "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"})
        links = verify.text.count("/matches/")
        print(f"验证（impersonate={imp}）：HTTP {verify.status_code}，"
              f"页面里 /matches/ 出现 {links} 次")
        if verify.status_code == 200 and links:
            print("成功：现在可以用 --cookies 抓取职业 demo 了。")
            return 0
    print("失败：cookie + UA 组合仍被拦，稍后再试或换网络环境。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
