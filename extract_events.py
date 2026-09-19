"""Extract positional events from a CS2 demo using demoparser2 directly.

awpy 2.0.2's ``Demo.parse()`` breaks on current CS2 demos because the
``round_end.winner`` field is now an integer while awpy still calls ``.str``
methods on it. We only need the primitives here, so we drive demoparser2
directly and keep full control over the schema.

demoparser2 returns pandas; we convert to polars on the way in so the rest of
the pipeline has one dataframe library.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
from demoparser2 import DemoParser

# Player props we want merged onto each event. demoparser2 prefixes them by
# role: "attacker" is whoever fired, "user" is the subject of the event
# (the victim, for player_death).
PLAYER_PROPS = ["X", "Y", "Z", "team_num", "team_name"]

# CS2 team numbers, as they appear on player entities.
TEAM_NUM_TO_SIDE = {2: "T", 3: "CT"}
TEAM_NAME_TO_SIDE = {"TERRORIST": "T", "CT": "CT"}


def add_side_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Derive readable T/CT labels from the raw team numbers.

    demoparser2 exposes ``team_num`` per role but no ``side`` string, so the
    T/CT split your experiment needs has to be derived here.
    """
    for role in ("attacker", "user", "assister"):
        num_col = f"{role}_team_num"
        if num_col not in frame.columns:
            continue
        frame = frame.with_columns(
            pl.col(num_col)
            .replace_strict(TEAM_NUM_TO_SIDE, default=None, return_dtype=pl.String)
            .alias(f"{role}_side")
        )
    return frame


def parse_events(parser: DemoParser, names: list[str], **kwargs) -> dict[str, pl.DataFrame]:
    frames = parser.parse_events(names, **kwargs)
    # demoparser2 yields a list of (event_name, pandas.DataFrame) tuples.
    return {name: pl.from_pandas(frame) for name, frame in frames}


def main() -> None:
    demo_path = Path(sys.argv[1] if len(sys.argv) > 1 else "demos/test_demo.dem")
    if not demo_path.exists():
        raise SystemExit(f"demo not found: {demo_path}")

    parser = DemoParser(str(demo_path))

    header = parser.parse_header()
    print("=== header ===")
    for key in ("map_name", "tick_rate", "demo_version_name", "server_name"):
        if key in header:
            print(f"  {key}: {header[key]}")

    events = parse_events(
        parser,
        ["player_death", "round_start", "round_end", "round_freeze_end"],
        player=PLAYER_PROPS,
    )

    print("=== events ===")
    for name, frame in events.items():
        print(f"  {name}: {frame.shape[0]} rows, {frame.shape[1]} cols")

    events = {name: add_side_columns(frame) for name, frame in events.items()}

    kills = events["player_death"]
    pos_cols = [c for c in kills.columns if c.endswith(("_X", "_Y", "_Z"))]

    print("=== sample kills ===")
    keep = [c for c in ("tick", "attacker_name", "user_name", "attacker_side",
                        "user_side", "weapon", *pos_cols) if c in kills.columns]
    print("  " + " | ".join(keep))
    for row in kills.select(keep).head(6).iter_rows(named=True):
        cells = []
        for col in keep:
            value = row[col]
            cells.append(f"{value:.1f}" if isinstance(value, float) else str(value))
        print("  " + " | ".join(cells))

    print("=== coordinate ranges (attacker) ===")
    for axis in ("X", "Y", "Z"):
        col = f"attacker_{axis}"
        if col in kills.columns:
            series = kills[col].cast(pl.Float64, strict=False).drop_nulls()
            if not series.is_empty():
                print(f"  {col}: min={series.min():.1f} max={series.max():.1f}")

    print("=== side split ===")
    if "attacker_side" in kills.columns:
        counts = kills.group_by("attacker_side").len().sort("attacker_side")
        for row in counts.iter_rows(named=True):
            print(f"  killer side {row['attacker_side']}: {row['len']} kills")

    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    for name, frame in events.items():
        target = out_dir / f"{name}.parquet"
        frame.write_parquet(target)
    print(f"=== wrote {len(events)} parquet files to {out_dir}/ ===")

    meta = {
        "demo": str(demo_path),
        "map_name": header.get("map_name"),
        "tick_rate": header.get("tick_rate"),
        "events": {name: frame.shape[0] for name, frame in events.items()},
    }
    meta_path = out_dir / "events_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"=== wrote {meta_path}（记录 demo 属于哪张地图）===")


if __name__ == "__main__":
    main()
