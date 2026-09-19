"""Single-map example analysis: kills and early-round occupancy, split by side.

Two layers, matching the experiment design:

1. Kill density per side  - where fights actually resolve.
2. Occupancy in the first 15 seconds after freeze time ends - the closest
   in-game analogue of "natural movement", before tactics and utility have
   reshaped the space.

Splitting by side matters: CT and T have completely different objectives on
the same geometry, so pooling them averages two different movement logics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
from PIL import Image
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from heatmap import read_overview_all, world_to_radar  # noqa: E402

# Which map to analyse. Pass it on the command line, e.g.
#   python analyse_map.py de_dust2
# The demo in demos/ must be a recording of the same map.
MAP_NAME = sys.argv[1] if len(sys.argv) > 1 else "de_mirage"
DEMO_PATH = Path("demos/test_demo.dem")
TICK_RATE = 64
OCCUPANCY_WINDOW_SECONDS = 15
OCCUPANCY_SAMPLE_STEP = 32  # half a second
SIGMA = 18

SIDES = ("CT", "T")
SIDE_CMAP = {"all": "turbo", "CT": "Blues", "T": "Oranges"}

# Grid used for the machine-readable export. 16x16 over a 1024 px radar puts
# a cell at 64 px, i.e. 320 world units at scale 5 - roughly a doorway.
GRID_CELLS = 16

SPAWN_MARKERS = (
    ("CTSpawn_x", "CTSpawn_y", "CT spawn"),
    ("TSpawn_x", "TSpawn_y", "T spawn"),
    ("bombA_x", "bombA_y", "A site"),
    ("bombB_x", "bombB_y", "B site"),
)


def load_radar(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    radar = Image.open(path).convert("RGBA")
    return np.array(radar), radar.size


def density(px: np.ndarray, py: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    grid, _, _ = np.histogram2d(
        py, px, bins=[height, width], range=[[0, height], [0, width]]
    )
    smoothed = gaussian_filter(grid, sigma=SIGMA)
    peak = smoothed.max()
    return smoothed / peak if peak > 0 else smoothed


def draw_panel(
    ax: plt.Axes,
    radar_img: np.ndarray,
    size: tuple[int, int],
    px: np.ndarray,
    py: np.ndarray,
    title: str,
    cmap: str,
    overview: dict[str, float] | None = None,
) -> None:
    width, height = size
    ax.imshow(radar_img, extent=[0, width, height, 0])
    if len(px):
        heat = density(px, py, size)
        ax.imshow(
            np.ma.masked_where(heat < 0.05, heat),
            extent=[0, width, height, 0],
            cmap=cmap,
            alpha=0.6,
        )
        ax.scatter(px, py, s=7, c="white", edgecolors="black",
                   linewidths=0.3, alpha=0.8)
    if overview is not None:
        for key_x, key_y, label in SPAWN_MARKERS:
            if key_x in overview and key_y in overview:
                ax.plot(
                    overview[key_x] * width, overview[key_y] * height,
                    marker="+", markersize=11, markeredgewidth=1.6,
                    color="lime", linestyle="none",
                )
    ax.set_title(f"{title}\n{len(px)} events", fontsize=11)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()


def positions(frame: pl.DataFrame, side: str | None, columns: tuple[str, str]):
    if side is not None and "attacker_side" in frame.columns:
        frame = frame.filter(pl.col("attacker_side") == side)
    return (
        frame[columns[0]].cast(pl.Float64).to_numpy(),
        frame[columns[1]].cast(pl.Float64).to_numpy(),
    )


def build_kill_figure(
    radar_img: np.ndarray, size: tuple[int, int], overview: dict[str, float]
) -> None:
    kills = pl.read_parquet("out/player_death.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(19, 7), dpi=100)

    specs = [
        (None, "All kills", "all"),
        ("CT", "Killed by CT", "CT"),
        ("T", "Killed by T", "T"),
    ]
    for ax, (side, title, key) in zip(axes, specs):
        wx, wy = positions(kills, side, ("attacker_X", "attacker_Y"))
        px, py = world_to_radar(wx, wy, overview)
        draw_panel(ax, radar_img, size, px, py, title, SIDE_CMAP[key], overview)

    fig.suptitle(f"{MAP_NAME} - kill positions by side", fontsize=14)
    fig.tight_layout()
    out = Path("out") / f"{MAP_NAME}_kills_by_side.png"
    fig.savefig(out, facecolor="black")
    plt.close(fig)
    print(f"wrote {out}")


def sample_early_round_ticks() -> list[int]:
    """Ticks covering the first N seconds after freeze time, every round."""
    freeze_ends = pl.read_parquet("out/round_freeze_end.parquet")["tick"].to_list()
    window = OCCUPANCY_WINDOW_SECONDS * TICK_RATE
    ticks: list[int] = []
    for freeze_end in freeze_ends:
        if freeze_end is None:
            continue
        start = int(freeze_end) + OCCUPANCY_SAMPLE_STEP
        ticks.extend(range(start, start + window, OCCUPANCY_SAMPLE_STEP))
    return ticks


def build_occupancy_figure(
    radar_img: np.ndarray, size: tuple[int, int], overview: dict[str, float]
) -> pl.DataFrame:
    from demoparser2 import DemoParser

    ticks = sample_early_round_ticks()
    print(f"occupancy: sampling {len(ticks)} ticks "
          f"({OCCUPANCY_WINDOW_SECONDS}s window x rounds, "
          f"every {OCCUPANCY_SAMPLE_STEP / TICK_RATE:.1f}s)")

    parser = DemoParser(str(DEMO_PATH))
    frame = pl.from_pandas(parser.parse_ticks(["X", "Y", "team_num"], ticks=ticks))
    print(f"  parsed {frame.shape[0]} player-tick rows")

    fig, axes = plt.subplots(1, 3, figsize=(19, 7), dpi=100)
    for ax, side in zip(axes[:2], SIDES):
        team_num = 3 if side == "CT" else 2
        subset = frame.filter(pl.col("team_num") == team_num)
        px, py = world_to_radar(
            subset["X"].cast(pl.Float64).to_numpy(),
            subset["Y"].cast(pl.Float64).to_numpy(),
            overview,
        )
        draw_panel(
            ax, radar_img, size, px, py,
            f"{side} players, first {OCCUPANCY_WINDOW_SECONDS}s",
            SIDE_CMAP[side], overview,
        )

    # Difference panel: where the two sides' early-round presence diverges.
    ax = axes[2]
    width, height = size
    grids = {}
    for side in SIDES:
        team_num = 3 if side == "CT" else 2
        subset = frame.filter(pl.col("team_num") == team_num)
        px, py = world_to_radar(
            subset["X"].cast(pl.Float64).to_numpy(),
            subset["Y"].cast(pl.Float64).to_numpy(),
            overview,
        )
        grids[side] = density(px, py, size)
    diff = grids["CT"] - grids["T"]
    ax.imshow(radar_img, extent=[0, width, height, 0])
    limit = np.abs(diff).max()
    ax.imshow(
        np.ma.masked_where(np.abs(diff) < 0.1 * limit, diff),
        extent=[0, width, height, 0],
        cmap="coolwarm", vmin=-limit, vmax=limit, alpha=0.65,
    )
    ax.set_title("Occupancy difference\nred = CT only, blue = T only", fontsize=11)
    for key_x, key_y, _label in SPAWN_MARKERS:
        if key_x in overview and key_y in overview:
            ax.plot(
                overview[key_x] * width, overview[key_y] * height,
                marker="+", markersize=11, markeredgewidth=1.6,
                color="lime", linestyle="none",
            )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()

    fig.suptitle(
        f"{MAP_NAME} - early-round occupancy "
        f"(first {OCCUPANCY_WINDOW_SECONDS}s after freeze time)", fontsize=14
    )
    fig.tight_layout()
    out = Path("out") / f"{MAP_NAME}_occupancy_by_side.png"
    fig.savefig(out, facecolor="black")
    plt.close(fig)
    print(f"wrote {out}")
    return frame


def export_grid(
    overview: dict[str, float],
    radar_size: tuple[int, int],
    kills: pl.DataFrame,
    occupancy: pl.DataFrame,
) -> None:
    """Export per-cell counts in both radar pixels and world units.

    This is the join table for the spatial-syntax step: once depthmapX gives
    you integration/choice per area, match it to these cells by world
    coordinates (or by tracing the same cells onto the axial map) and the
    correlation analysis becomes a plain table join.
    """
    width, height = radar_size
    cell_w = width / GRID_CELLS
    cell_h = height / GRID_CELLS
    scale = overview["scale"]

    kill_px, kill_py = world_to_radar(
        kills["attacker_X"].cast(pl.Float64).to_numpy(),
        kills["attacker_Y"].cast(pl.Float64).to_numpy(),
        overview,
    )
    kill_side = kills["attacker_side"].to_list()

    occ_px, occ_py = world_to_radar(
        occupancy["X"].cast(pl.Float64).to_numpy(),
        occupancy["Y"].cast(pl.Float64).to_numpy(),
        overview,
    )
    occ_team = occupancy["team_num"].to_list()

    rows = []
    for gy in range(GRID_CELLS):
        for gx in range(GRID_CELLS):
            px0, px1 = gx * cell_w, (gx + 1) * cell_w
            py0, py1 = gy * cell_h, (gy + 1) * cell_h
            in_kill = (kill_px >= px0) & (kill_px < px1) & (kill_py >= py0) & (kill_py < py1)
            in_occ = (occ_px >= px0) & (occ_px < px1) & (occ_py >= py0) & (occ_py < py1)

            # World bounds. Y is flipped: the smaller pixel Y is the larger
            # world Y.
            world_x_min = overview["pos_x"] + px0 * scale
            world_x_max = overview["pos_x"] + px1 * scale
            world_y_min = overview["pos_y"] - py1 * scale
            world_y_max = overview["pos_y"] - py0 * scale

            rows.append({
                "col": gx,
                "row": gy,
                "px_left": round(px0, 1),
                "px_top": round(py0, 1),
                "world_x_min": round(world_x_min, 1),
                "world_x_max": round(world_x_max, 1),
                "world_y_min": round(world_y_min, 1),
                "world_y_max": round(world_y_max, 1),
                "kills_total": int(in_kill.sum()),
                "kills_ct": sum(
                    1 for hit, side in zip(in_kill, kill_side) if hit and side == "CT"
                ),
                "kills_t": sum(
                    1 for hit, side in zip(in_kill, kill_side) if hit and side == "T"
                ),
                "occupancy_ct": sum(
                    1 for hit, team in zip(in_occ, occ_team) if hit and team == 3
                ),
                "occupancy_t": sum(
                    1 for hit, team in zip(in_occ, occ_team) if hit and team == 2
                ),
            })

    out = Path("out") / f"{MAP_NAME}_grid.csv"
    pl.DataFrame(rows).write_csv(out)
    print(f"wrote {out} ({GRID_CELLS}x{GRID_CELLS} cells, "
          f"{cell_w:.0f}x{cell_h:.0f}px each)")


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    overview = read_overview_all(Path(f"data/{MAP_NAME}.txt"))
    radar_img, size = load_radar(Path(f"data/{MAP_NAME}.png"))
    print(f"{MAP_NAME}: radar {size[0]}x{size[1]}, "
          f"pos_x={overview['pos_x']} pos_y={overview['pos_y']} "
          f"scale={overview['scale']}")

    build_kill_figure(radar_img, size, overview)
    occupancy = build_occupancy_figure(radar_img, size, overview)
    export_grid(overview, size, pl.read_parquet("out/player_death.parquet"), occupancy)


if __name__ == "__main__":
    main()
