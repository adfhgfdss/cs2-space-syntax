"""Objectively verify the world -> radar pixel transform.

The overview file declares where each team spawns, as normalised radar
coordinates (CTSpawn_x/y, TSpawn_x/y). At the moment freeze time ends, every
player is standing on their spawn. So we parse one tick just after freeze end
per round, convert those world positions to radar pixels, and compare the
per-team centroid against the declared spawn marker.

If the transform and the radar image match, the two should be close. If we
paired the wrong map image or got the axis flip wrong, they will be far apart.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from demoparser2 import DemoParser
from PIL import Image

from heatmap import read_overview, read_overview_all, world_to_radar

RADAR_SIZE = 1024
CT_TEAM_NUM = 3
T_TEAM_NUM = 2


def main() -> None:
    demo_path = Path("demos/test_demo.dem")
    overview = read_overview_all(Path("data/de_mirage.txt"))

    check_radar_coverage(overview)

    freeze_ends = pl.read_parquet("out/round_freeze_end.parquet")["tick"].to_list()
    # One second (64 ticks) after freeze time ends: everyone has run a little,
    # but nobody has crossed the map yet.
    sample_ticks = [int(t) + 64 for t in freeze_ends if t is not None]
    print(f"sampling {len(sample_ticks)} ticks, one per round")

    parser = DemoParser(str(demo_path))
    ticks = pl.from_pandas(
        parser.parse_ticks(["X", "Y", "team_num"], ticks=sample_ticks)
    )
    print(f"parsed {ticks.shape[0]} player-tick rows")

    print(f"{'team':>4} {'n':>5} {'world_x':>10} {'world_y':>10} "
          f"{'radar_x':>8} {'radar_y':>8} {'declared':>16} {'dist_px':>8}")

    for team_num, label, key_x, key_y in (
        (CT_TEAM_NUM, "CT", "CTSpawn_x", "CTSpawn_y"),
        (T_TEAM_NUM, "T", "TSpawn_x", "TSpawn_y"),
    ):
        subset = ticks.filter(pl.col("team_num") == team_num)
        if subset.is_empty():
            print(f"{label:>4}  no rows")
            continue
        mean_x = float(subset["X"].mean())
        mean_y = float(subset["Y"].mean())
        px, py = world_to_radar(
            np.array([mean_x]), np.array([mean_y]), overview
        )
        declared = (overview[key_x] * RADAR_SIZE, overview[key_y] * RADAR_SIZE)
        dist = float(np.hypot(px[0] - declared[0], py[0] - declared[1]))
        print(
            f"{label:>4} {subset.shape[0]:>5} {mean_x:>10.1f} {mean_y:>10.1f} "
            f"{px[0]:>8.1f} {py[0]:>8.1f} "
            f"({declared[0]:>5.0f},{declared[1]:>5.0f}) {dist:>8.1f}"
        )


def check_radar_coverage(overview: dict[str, float]) -> None:
    """Check that event positions land on drawn (opaque) parts of the radar.

    The radar image draws the map on a transparent background. If the image
    and the transform agree, kills must land on opaque pixels. A large share
    of points on transparent pixels means the image is a different map
    version (or the wrong map entirely).
    """
    radar = Image.open("data/de_mirage.png").convert("RGBA")
    alpha = np.array(radar.getchannel("A"))

    kills = pl.read_parquet("out/player_death.parquet")
    px, py = world_to_radar(
        kills["attacker_X"].cast(pl.Float64).to_numpy(),
        kills["attacker_Y"].cast(pl.Float64).to_numpy(),
        overview,
    )
    ix = np.clip(px.astype(int), 0, alpha.shape[1] - 1)
    iy = np.clip(py.astype(int), 0, alpha.shape[0] - 1)
    opaque = alpha[iy, ix] > 0
    print("=== radar image coverage test ===")
    print(f"  opaque pixels in image: {100 * (alpha > 0).mean():.1f}% of canvas")
    print(f"  kills landing on opaque pixels: {opaque.sum()}/{len(opaque)}"
          f" ({100 * opaque.mean():.1f}%)")


if __name__ == "__main__":
    main()
