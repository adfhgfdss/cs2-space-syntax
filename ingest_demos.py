r"""批量吃 demo：一张或多张地图、一场或多场 demo，一条命令全部抽完。

为什么需要它：单场 demo 的脚本（extract_events.py / extract_tracks.py）一次只处理
一个文件、还只写一份 `out\tracks.parquet`。要做两图对比、要把样本量从 1 场堆到
10-20 场，就需要一个批处理入口。

它做的事：
  1. 扫描 `demos\` 下所有 `.dem`（可以按 `demos\<地图>\` 分目录，也可以平铺）；
  2. 读每个 demo 的 header 判断它属于哪张地图，自动分组；
  3. 抽击杀/回合事件 + 每 0.25 秒一个玩家位置采样；
  4. 给每行打上 `demo_id`，并把 `round_index` 编成**跨 demo 全局唯一**的编号
     （这样后面的流量统计不会被不同 demo 的同号回合串起来）；
  5. 按地图落盘到 `out\behaviour\<地图>_*.parquet` 和 `<地图>_meta.json`。

用法：
    .venv\Scripts\python.exe ingest_demos.py                      # 吃 demos\ 下全部
    .venv\Scripts\python.exe ingest_demos.py --demos demos        # 指定目录
    .venv\Scripts\python.exe ingest_demos.py --only de_dust2      # 只处理某张图
    .venv\Scripts\python.exe ingest_demos.py --step 32            # 采样更稀（更快）

之后的行为分析直接读按地图分好的表：
    .venv\Scripts\python.exe analyse_behaviour.py de_dust2
    .venv\Scripts\python.exe analyse_behaviour.py de_mirage
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl
from demoparser2 import DemoParser

from extract_events import PLAYER_PROPS, add_side_columns, parse_events

EVENT_NAMES = ["player_death", "round_start", "round_end", "round_freeze_end"]
TEAM_NUM_TO_SIDE = {2: "T", 3: "CT"}


def round_windows(starts: list[int], ends: list[int]) -> list[tuple[int, int]]:
    """把 round_start / round_end 配成 [开始, 结束) 区间。"""
    windows: list[tuple[int, int]] = []
    for i, start in enumerate(starts):
        later = [t for t in ends if t > start] + starts[i + 1:i + 2]
        end = min(later) if later else start + 64 * 200
        windows.append((start, end))
    return windows


def process_demo(path: Path, step: int, round_offset: int, source: str) -> dict:
    """抽一场 demo，返回 {map_name, events, tracks, rounds, freeze_ends}。"""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    sha256 = digest.hexdigest()
    size_bytes = path.stat().st_size

    parser = DemoParser(str(path))
    header = parser.parse_header()
    map_name = header.get("map_name")
    tick_rate = int(header.get("tick_rate", 64))
    if not map_name:
        raise SystemExit(f"{path.name}: header 里没有 map_name，无法判断地图")

    events = {
        name: add_side_columns(frame)
        for name, frame in parse_events(parser, EVENT_NAMES, player=PLAYER_PROPS).items()
    }

    starts = sorted(int(t) for t in events["round_start"]["tick"].to_list() if t is not None)
    ends = sorted(int(t) for t in events["round_end"]["tick"].to_list() if t is not None)
    windows = round_windows(starts, ends)
    freezes = sorted(int(t) for t in events["round_freeze_end"]["tick"].to_list()
                     if t is not None)
    round_freeze = [next((f for f in freezes if f > s), None) for s, _ in windows]

    wanted: list[int] = []
    for start, stop in windows:
        wanted.extend(range(start, stop, step))

    tracks = pl.from_pandas(
        parser.parse_ticks(["X", "Y", "Z", "team_num", "health"], ticks=wanted)
    )
    rename = {}
    for col in tracks.columns:
        tail = col.split("_")[-1]
        if tail in {"X", "Y", "Z", "health"} and tail not in tracks.columns:
            rename[col] = tail
    if rename:
        tracks = tracks.rename(rename)
    if "health" in tracks.columns:
        tracks = tracks.filter(pl.col("health") > 0)
    tracks = tracks.filter(pl.col("team_num").is_in([2, 3])).drop_nulls(["X", "Y"])

    demo_id = str(path.relative_to(Path("demos"))).replace("\\", "/") if path.is_relative_to(Path("demos")) else path.name
    demo_id = demo_id[:-4] if demo_id.endswith(".dem") else demo_id

    # 回合编号跨 demo 全局唯一：round_index；另外保留 demo 内的序号
    index_expr = pl.lit(-1, dtype=pl.Int32)
    local_expr = pl.lit(-1, dtype=pl.Int32)
    seconds = pl.lit(None, dtype=pl.Float64)
    after = pl.lit(None, dtype=pl.Float64)
    freeze_tick = pl.lit(None, dtype=pl.Int64)
    for i, (start, stop) in enumerate(windows):
        in_round = (pl.col("tick") >= start) & (pl.col("tick") < stop)
        index_expr = pl.when(in_round).then(round_offset + i).otherwise(index_expr)
        local_expr = pl.when(in_round).then(i).otherwise(local_expr)
        seconds = pl.when(in_round).then((pl.col("tick") - start) / tick_rate).otherwise(seconds)
        f = round_freeze[i]
        if f is not None:
            freeze_tick = pl.when(in_round).then(f).otherwise(freeze_tick)
            after = pl.when(in_round).then((pl.col("tick") - f) / tick_rate).otherwise(after)

    tracks = tracks.with_columns(
        pl.lit(demo_id).alias("demo_id"),
        pl.lit(source).alias("source"),
        index_expr.cast(pl.Int32).alias("round_index"),
        local_expr.cast(pl.Int32).alias("round_in_demo"),
        seconds.alias("seconds_into_round"),
        freeze_tick.alias("freeze_end_tick"),
        after.alias("seconds_after_freeze"),
        pl.col("team_num").replace_strict(TEAM_NUM_TO_SIDE, default=None).alias("side"),
    )
    for name, frame in events.items():
        if "tick" in frame.columns and frame.height:
            events[name] = frame.with_columns(pl.lit(demo_id).alias("demo_id"))

    print(f"  {demo_id}: 地图 {map_name}，回合 {len(windows)}，"
          f"击杀 {events['player_death'].height}，轨迹 {tracks.height:,} 行，"
          f"{size_bytes/1024/1024:.1f} MB")
    return {
        "demo_id": demo_id,
        "path": str(path),
        "map_name": map_name,
        "source": source,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "tick_rate": tick_rate,
        "rounds": len(windows),
        "kills": events["player_death"].height,
        "rows": tracks.height,
        "events": events,
        "tracks": tracks,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demos", type=Path, default=Path("demos"))
    ap.add_argument("--step", type=int, default=16, help="每多少个 tick 采一个点（64 tick = 1 秒）")
    ap.add_argument("--only", default=None, help="只处理这张地图，例如 de_dust2")
    ap.add_argument("--out", type=Path, default=Path("out/behaviour"))
    ap.add_argument("--source", default="auto",
                    help='样本来源标签：pro / pug / auto（按目录名猜：demos→pro，demos_pug→pug）')
    args = ap.parse_args()

    def source_of(path: Path) -> str:
        if args.source != "auto":
            return args.source
        top = path.parts[0].lower() if path.parts else ""
        if "pug" in top:
            return "pug"
        if top == "demos":
            return "pro"
        return "unknown"

    if args.demos.is_file():
        demo_paths = [args.demos]
    else:
        demo_paths = sorted(p for p in args.demos.rglob("*.dem") if p.is_file())
    if not demo_paths:
        raise SystemExit(
            f"{args.demos} 下没有 .dem 文件。\n"
            f"把 demo 放进去（可以按地图分子目录，例如 demos/de_dust2/xxx.dem），再重跑本脚本。"
        )
    print(f"发现 {len(demo_paths)} 个 demo：")
    for p in demo_paths:
        print(f"  {p}")

    by_map: dict[str, list[dict]] = {}
    for path in demo_paths:
        try:
            info = process_demo(path, args.step, 0, source_of(path))  # 回合号稍后重编
        except Exception as exc:  # noqa: BLE001
            print(f"  !! 跳过 {path.name}：{type(exc).__name__} {exc}")
            continue
        if args.only and info["map_name"] != args.only:
            print(f"  （跳过 {info['demo_id']}：地图是 {info['map_name']}，只要 {args.only}）")
            continue
        by_map.setdefault(info["map_name"], []).append(info)

    if not by_map:
        if args.only:
            raise SystemExit(
                f"demos 目录里没有属于 {args.only} 的 demo。\n"
                f"把 {args.only} 的录像（.dem）放进 {args.demos}\\，"
                f"建议按地图建子目录：{args.demos}\\{args.only}\\xxx.dem"
            )
        raise SystemExit("没有可用的 demo。")

    args.out.mkdir(parents=True, exist_ok=True)
    for map_name, infos in sorted(by_map.items()):
        total_rounds = sum(i["rounds"] for i in infos)
        # 重新编号：跨 demo 全局唯一
        offset = 0
        tracks_parts, event_parts = [], {}
        for info in infos:
            tr = info["tracks"].with_columns(
                (pl.col("round_index") + offset).cast(pl.Int32).alias("round_index"))
            tracks_parts.append(tr)
            offset += info["rounds"]
            for name, frame in info["events"].items():
                event_parts.setdefault(name, []).append(frame)
        tracks = pl.concat(tracks_parts, how="diagonal_relaxed")
        tracks.write_parquet(args.out / f"{map_name}_tracks.parquet")
        for name, parts in event_parts.items():
            pl.concat(parts, how="diagonal_relaxed").write_parquet(
                args.out / f"{map_name}_{name}.parquet")

        meta = {
            "map_name": map_name,
            "source": infos[0]["source"] if infos else "unknown",
            "demo_count": len(infos),
            "rounds": total_rounds,
            "kills": sum(i["kills"] for i in infos),
            "rows": tracks.height,
            "bytes_total": sum(i["size_bytes"] for i in infos),
            "sample_step_ticks": args.step,
            "demos": [{k: v for k, v in i.items() if k not in ("events", "tracks")}
                      for i in infos],
        }
        (args.out / f"{map_name}_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[{map_name}] {len(infos)} 场 demo / {total_rounds} 回合 / "
              f"{meta['kills']} 击杀 / {tracks.height:,} 个轨迹采样")
        print(f"  -> {args.out}\\{map_name}_tracks.parquet 等")

    print("\n下一步：")
    for map_name in sorted(by_map):
        print(f"  .venv\\Scripts\\python.exe analyse_behaviour.py {map_name}")


if __name__ == "__main__":
    main()
