"""Validate an overview .txt against its radar .png, with no demo required.

The overview file declares where both spawns and both bomb sites sit, as
normalised radar coordinates. Those four points are always on walkable
ground, so they must land on drawn (opaque) pixels of the radar image.

This catches the two failure modes that matter when you pair files from
different sources:

  * the radar image is from a different map version than the overview file
  * the wrong radar image was paired with the overview file entirely

Usage:
    python check_map_files.py de_dust2
    python check_map_files.py            # checks every map in data/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from heatmap import read_overview_all

LANDMARKS = (
    ("CTSpawn_x", "CTSpawn_y", "CT spawn"),
    ("TSpawn_x", "TSpawn_y", "T spawn"),
    ("bombA_x", "bombA_y", "bomb site A"),
    ("bombB_x", "bombB_y", "bomb site B"),
)


def check(stem: Path) -> bool:
    png_path = stem.with_suffix(".png")
    txt_path = stem.with_suffix(".txt")
    if not png_path.exists() or not txt_path.exists():
        print(f"{stem.name}: SKIP (need both .png and .txt)")
        return False

    overview = read_overview_all(txt_path)
    radar = Image.open(png_path).convert("RGBA")
    alpha = np.array(radar.getchannel("A"))
    height, width = alpha.shape
    drawn = 100 * (alpha > 0).mean()

    print(f"{stem.name}")
    print(f"  radar {width}x{height}, drawn area {drawn:.1f}% of canvas, "
          f"scale={overview.get('scale')}")

    ok = True
    for key_x, key_y, label in LANDMARKS:
        if key_x not in overview or key_y not in overview:
            print(f"  {label:<12} MISSING from overview file")
            ok = False
            continue
        x = min(int(overview[key_x] * width), width - 1)
        y = min(int(overview[key_y] * height), height - 1)
        opaque = alpha[y, x] > 0
        status = "OK" if opaque else "OUTSIDE DRAWN AREA"
        print(f"  {label:<12} pixel ({x:>4},{y:>4})  alpha={alpha[y, x]:>3}  {status}")
        ok &= bool(opaque)

    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    data_dir = Path("data")
    if len(sys.argv) > 1:
        targets = [data_dir / name for name in sys.argv[1:]]
    else:
        targets = sorted(p for p in data_dir.glob("*.txt"))

    if not targets:
        raise SystemExit("no map files found in data/")

    results = [check(t) for t in targets]
    print()
    print(f"{sum(results)}/{len(results)} map file pairs passed")


if __name__ == "__main__":
    main()
