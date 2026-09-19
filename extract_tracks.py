"""把一个 demo 里每个玩家的移动轨迹抽成表（而不只是击杀点）。

为什么需要：热力图只用到了击杀时刻，但空间句法要对照的是"人在空间里的分布"，
而且轨迹还有第二个用途——**给雷达图的可通行区域做标注**：玩家走过的地方一定
能走，这比肉眼判断画面里哪块是地面、哪块是屋顶可靠得多。

用法：
    .venv\\Scripts\\python.exe extract_tracks.py demos/test_demo.dem
    .venv\\Scripts\\python.exe extract_tracks.py demos/test_demo.dem --step 16

输出：out\\tracks.parquet
    列：tick, steamid, name, team_num, side, X, Y, Z, health, round_index, seconds_into_round
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
from demoparser2 import DemoParser

TEAM_NUM_TO_SIDE = {2: "T", 3: "CT"}


def round_windows() -> list[tuple[int, int]]:
    """从已解析的回合事件里取每回合的 [开始, 结束) tick 区间。"""
    starts = pl.read_parquet("out/round_start.parquet")
    try:
        ends = pl.read_parquet("out/round_end.parquet")
    except FileNotFoundError:
        ends = None

    start_list = sorted(int(t) for t in starts["tick"].to_list() if t is not None)
    end_list = sorted(int(t) for t in ends["tick"].to_list() if t is not None) if ends is not None else []

    windows: list[tuple[int, int]] = []
    for i, start in enumerate(start_list):
        # 下一个回合开始（或最后一个回合结束）就是本次回合的右边界
        candidates = [t for t in end_list if t > start] + start_list[i + 1:i + 2]
        end = min(candidates) if candidates else start + 64 * 200
        windows.append((start, end))
    return windows


def freeze_end_ticks() -> list[int]:
    """回合冻结结束（玩家可以动的）tick。第一个回合可能没有这条事件。"""
    try:
        frame = pl.read_parquet("out/round_freeze_end.parquet")
    except FileNotFoundError:
        return []
    return sorted(int(t) for t in frame["tick"].to_list() if t is not None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("demo", nargs="?", default="demos/test_demo.dem")
    ap.add_argument("--step", type=int, default=16, help="每多少个 tick 采一个样本（64 tick = 1 秒）")
    ap.add_argument("--out", type=Path, default=Path("out/tracks.parquet"))
    args = ap.parse_args()

    demo_path = Path(args.demo)
    if not demo_path.exists():
        raise SystemExit(f"demo not found: {demo_path}")

    parser = DemoParser(str(demo_path))
    header = parser.parse_header()
    tick_rate = int(header.get("tick_rate", 64))
    print(f"demo: {demo_path.name}  map={header.get('map_name')}  tick_rate={tick_rate}")

    windows = round_windows()
    print(f"rounds: {len(windows)} 个")
    freezes = freeze_end_ticks()
    # 每个回合配一个 freeze_end：取该回合开始之后的第一个
    round_freeze: list[int | None] = []
    for start, _end in windows:
        candidate = next((f for f in freezes if f > start), None)
        round_freeze.append(candidate)
    print("回合 -> freeze_end tick:", list(zip([w[0] for w in windows], round_freeze)))

    wanted_ticks: list[int] = []
    for start, end in windows:
        wanted_ticks.extend(range(start, end, args.step))
    print(f"解析 {len(wanted_ticks)} 个 tick（每 {args.step} tick 一个采样，"
          f"约每 {args.step / tick_rate:.2f} 秒）...")

    frame = pl.from_pandas(
        parser.parse_ticks(["X", "Y", "Z", "team_num", "health"], ticks=wanted_ticks)
    )
    # 列名在不同版本里可能带前缀，统一成不带前缀的写法
    rename = {}
    for col in frame.columns:
        if col in {"X", "Y", "Z", "team_num", "health"}:
            continue
        tail = col.split("_")[-1]
        if tail in {"X", "Y", "Z", "health"} and tail not in frame.columns:
            rename[col] = tail
    if rename:
        frame = frame.rename(rename)

    print(f"原始行数：{frame.shape[0]:,}  列：{frame.columns}")

    if "health" in frame.columns:
        frame = frame.filter(pl.col("health") > 0)
    frame = frame.filter(pl.col("team_num").is_in([2, 3]))
    frame = frame.drop_nulls(["X", "Y"])

    frame = frame.with_columns(
        pl.col("team_num").replace_strict(TEAM_NUM_TO_SIDE, default=None).alias("side")
    )

    # 标注每个样本属于第几个回合、进入回合多少秒
    round_index = pl.lit(-1, dtype=pl.Int32)
    seconds = pl.lit(None, dtype=pl.Float64)
    expr_index = pl.when(pl.col("tick") < 0).then(-1)
    for i, (start, end) in enumerate(windows):
        in_round = (pl.col("tick") >= start) & (pl.col("tick") < end)
        expr_index = pl.when(in_round).then(i).otherwise(expr_index)
        seconds = pl.when(in_round).then((pl.col("tick") - start) / tick_rate).otherwise(seconds)
    frame = frame.with_columns(
        expr_index.cast(pl.Int32).alias("round_index"),
        seconds.alias("seconds_into_round"),
    )

    # 相对"冻结结束"的时间：分析开局自然移动要用这个，而不是相对回合开始
    freeze_expr = pl.lit(None, dtype=pl.Int64)
    after_expr = pl.lit(None, dtype=pl.Float64)
    for i, (start, end) in enumerate(windows):
        f = round_freeze[i]
        if f is None:
            continue
        in_round = (pl.col("tick") >= start) & (pl.col("tick") < end)
        freeze_expr = pl.when(in_round).then(f).otherwise(freeze_expr)
        after_expr = pl.when(in_round).then((pl.col("tick") - f) / tick_rate).otherwise(after_expr)
    frame = frame.with_columns(
        freeze_expr.alias("freeze_end_tick"),
        after_expr.alias("seconds_after_freeze"),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(args.out)
    meta = {
        "demo": str(demo_path),
        "map_name": header.get("map_name"),
        "tick_rate": tick_rate,
        "sample_step_ticks": args.step,
        "rounds": len(windows),
        "round_freeze_end": [None if f is None else int(f) for f in round_freeze],
        "rows": frame.shape[0],
    }
    meta_path = args.out.parent / "tracks_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {meta_path}（记录 demo 属于哪张地图，防止张冠李戴）")
    print(f"wrote {args.out}  {frame.shape[0]:,} 行")
    for side in ("CT", "T"):
        sub = frame.filter(pl.col("side") == side)
        print(f"  {side}: {sub.shape[0]:,} 个样本，"
              f"X {sub['X'].min():.0f}..{sub['X'].max():.0f}，"
              f"Y {sub['Y'].min():.0f}..{sub['Y'].max():.0f}")


if __name__ == "__main__":
    main()
