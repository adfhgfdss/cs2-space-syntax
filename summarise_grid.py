"""Print a plain-text summary of the exported per-cell grid table."""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

MAP_NAME = sys.argv[1] if len(sys.argv) > 1 else "de_mirage"


def show(title: str, frame: pl.DataFrame, sort_col: str, columns: list[str]) -> None:
    print(f"--- {title} ---")
    print("  " + "  ".join(f"{c:>12}" for c in columns))
    for row in frame.sort(sort_col, descending=True).head(8).iter_rows(named=True):
        print("  " + "  ".join(f"{row[c]!s:>12}" for c in columns))
    print()


def main() -> None:
    path = Path("out") / f"{MAP_NAME}_grid.csv"
    grid = pl.read_csv(path).with_columns(
        (pl.col("occupancy_ct") + pl.col("occupancy_t")).alias("occupancy_total")
    )

    n = grid.shape[0]
    print(f"{path}")
    print(f"  {n} cells ({int(n ** 0.5)}x{int(n ** 0.5)}), "
          f"{grid['kills_total'].gt(0).sum()} contain kills, "
          f"{grid['occupancy_total'].gt(0).sum()} contain occupancy samples")
    print()

    show(
        "cells with the most kills",
        grid,
        "kills_total",
        ["col", "row", "kills_total", "kills_ct", "kills_t",
         "occupancy_ct", "occupancy_t"],
    )
    show(
        "cells with the most occupancy",
        grid,
        "occupancy_total",
        ["col", "row", "occupancy_ct", "occupancy_t", "kills_total"],
    )

    # How one-sided is each side's territory?
    ct_only = grid.filter(
        (pl.col("occupancy_ct") > 0) & (pl.col("occupancy_t") == 0)
    ).shape[0]
    t_only = grid.filter(
        (pl.col("occupancy_t") > 0) & (pl.col("occupancy_ct") == 0)
    ).shape[0]
    shared = grid.filter(
        (pl.col("occupancy_ct") > 0) & (pl.col("occupancy_t") > 0)
    ).shape[0]
    print("--- territory overlap in the first 15 seconds ---")
    print(f"  cells visited by CT only : {ct_only}")
    print(f"  cells visited by T only  : {t_only}")
    print(f"  cells visited by both    : {shared}")


if __name__ == "__main__":
    main()
