r"""一条命令跑完整条管道（每一步都会自动存快照，可回溯）。

用法：
    # 只跑几何层（不需要 demo，几张图都行）
    .venv\Scripts\python.exe run_all.py de_dust2 de_mirage

    # 几何层 + 行为层：demo 放在 demos\ 下（可按地图分子目录）
    .venv\Scripts\python.exe run_all.py de_dust2 de_mirage --demos demos

    # 只想先跑某个文件
    .venv\Scripts\python.exe run_all.py de_mirage --demo demos/test_demo.dem

    # 跑完再出两图对比
    .venv\Scripts\python.exe run_all.py de_dust2 de_mirage --demos demos --compare

行为层与几何层是解耦的：某张图没有 demo，脚本会自动跳过它的行为分析，
几何层照跑不误（de_dust2 当前就是这种情况）。

参数：
    --demos DIR     行为数据来源目录（递归找 .dem），默认 demos
    --demo PATH     单个 demo 文件
    --step N        轨迹采样间隔（tick），默认 16（=0.25 秒）
    --cell N        句法栅格边长（像素），默认 16
    --erode N       可通行区向内腐蚀格数，默认 0
    --skip-extract  已有 data/game_nav/<map>.nav 时跳过抽取
    --compare       最后跑 compare_maps.py
    --no-snapshot   不经过快照工具（默认每一步都用 version.py run 包一层）
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(label: str, cmd: list[str], use_snapshot: bool) -> int:
    print(f"\n{'=' * 78}\n>>> {label}\n{'=' * 78}")
    full = ([PY, str(ROOT / "_history" / "version.py"), "run", label, "--", *cmd]
            if use_snapshot else cmd)
    proc = subprocess.run(full, cwd=ROOT)
    if proc.returncode != 0:
        print(f"!! 步骤失败（退出码 {proc.returncode}）：{label}")
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("maps", nargs="+")
    ap.add_argument("--demos", type=Path, default=None, help="demo 目录（递归找 .dem）")
    ap.add_argument("--demo", type=Path, default=None, help="单个 demo 文件")
    ap.add_argument("--step", type=int, default=16)
    ap.add_argument("--cell", type=int, default=16)
    ap.add_argument("--erode", type=int, default=0)
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--no-snapshot", action="store_true")
    args = ap.parse_args()
    snapshot = not args.no_snapshot

    # 1) 行为数据：一次把 demos 目录下所有 demo 抽完，按地图分好
    source = args.demo or args.demos or Path("demos")
    have_behaviour: set[str] = set()
    if source is not None and Path(source).exists():
        code = run(f"批量抽取 demo（{source}）",
                   [PY, "ingest_demos.py", "--demos", str(source), "--step", str(args.step)],
                   snapshot)
        if code != 0:
            print("（行为数据抽取失败，继续跑几何层）")
        else:
            for meta in (ROOT / "out" / "behaviour").glob("*_meta.json"):
                have_behaviour.add(json.loads(meta.read_text(encoding="utf-8"))["map_name"])
            print(f"已备好行为数据的地图：{sorted(have_behaviour) or '（无）'}")
    else:
        print(f"没有找到 demo 来源（{source}），只跑几何层")

    # 2) 逐图跑几何层，有行为数据的顺带跑相关分析
    for map_name in args.maps:
        nav_path = ROOT / "data" / "game_nav" / f"{map_name}.nav"
        if not args.skip_extract or not nav_path.exists():
            code = run(f"[{map_name}] 抽取导航网格",
                       [PY, "nav_extract.py", map_name], snapshot)
            if code != 0:
                return code
        code = run(f"[{map_name}] 解析导航网格 + 与雷达图对位",
                   [PY, "navmesh.py", map_name, "--npz"], snapshot)
        if code != 0:
            return code
        code = run(f"[{map_name}] 空间句法 VGA",
                   [PY, "space_syntax.py", map_name,
                    "--cell", str(args.cell), "--erode", str(args.erode)], snapshot)
        if code != 0:
            return code
        if map_name in have_behaviour:
            code = run(f"[{map_name}] 行为 × 句法相关分析",
                       [PY, "analyse_behaviour.py", map_name], snapshot)
            if code != 0:
                return code
        else:
            print(f"\n[{map_name}] 跳过行为层：{ROOT / 'out' / 'behaviour'} 里没有它的 demo")

    # 3) 可选：两图对比
    if args.compare and len(args.maps) > 1:
        code = run("多图构形对比", [PY, "compare_maps.py", *args.maps], snapshot)
        if code != 0:
            return code

    print("\n全部完成。产物在 out\\，快照在 _history\\（list 可查看）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
