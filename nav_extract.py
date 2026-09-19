"""从本地 CS2 安装里抽出某张地图的导航网格（.nav）。

为什么需要它：空间句法要分析的"可通行空间"，雷达图上看不出来（俯视渲染里
屋顶和地面是叠在一起的）。而 `.nav` 是 Valve 自己烘培的导航网格，给出每块
可通行地面多边形的世界坐标，是最权威的"哪些地方能走"的定义。

用法：
    .venv\\Scripts\\python.exe nav_extract.py de_dust2
    .venv\\Scripts\\python.exe nav_extract.py de_mirage
    .venv\\Scripts\\python.exe nav_extract.py de_dust2 --steam "D:\\Steam"

输出：data\\game_nav\\<map>.nav，并打印尺寸与 SHA256（写报告时要引用）。
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from vpk_extract import Vpk

DEFAULT_LIBS = [
    Path(r"C:\Program Files (x86)\Steam"),
    Path(r"C:\Program Files\Steam"),
    Path(r"D:\Steam"),
    Path(r"E:\Steam"),
]


def find_map_vpk(map_name: str, steam_lib: Path | None) -> Path:
    """在常见的 Steam 库目录里找 maps\\<map>.vpk。"""
    libs = [steam_lib] if steam_lib else []
    libs += [p for p in DEFAULT_LIBS if p not in libs]
    # 顺带读一下 libraryfolders.vdf，支持装在别的盘
    for lib in list(libs):
        vdf = lib / "steamapps" / "libraryfolders.vdf"
        if vdf.is_file():
            for line in vdf.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line.startswith('"path"'):
                    raw = line.split('"')[3].replace("\\\\", "\\")
                    extra = Path(raw)
                    if extra not in libs:
                        libs.append(extra)
    rel = Path("steamapps/common/Counter-Strike Global Offensive/game/csgo/maps") / f"{map_name}.vpk"
    for lib in libs:
        candidate = lib / rel
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "没找到地图包。用 --steam 指定 Steam 根目录，或确认 CS2 装在哪个盘。\n"
        f"找过的路径示例：{ (libs[0] / rel) if libs else rel }"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("map", help="地图名，例如 de_dust2")
    ap.add_argument("--steam", type=Path, default=None, help="Steam 根目录")
    ap.add_argument("--out", type=Path, default=Path("data/game_nav"), help="输出目录")
    args = ap.parse_args()

    vpk_path = find_map_vpk(args.map, args.steam)
    print(f"地图包: {vpk_path}")
    vpk = Vpk(vpk_path)

    want = f"maps/{args.map}.nav".lower()
    matches = [e for e in vpk.entries() if e["path"].lower() == want]
    if not matches:
        raise SystemExit(f"{vpk_path.name} 里没有 {want}")
    entry = matches[0]
    blob = vpk.read(entry)

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{args.map}.nav"
    target.write_bytes(blob)
    print(f"抽出 {entry['path']}  {len(blob):,} bytes  ->  {target}")
    print(f"SHA256 {hashlib.sha256(blob).hexdigest()}")


if __name__ == "__main__":
    main()
