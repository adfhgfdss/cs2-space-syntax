r"""数据盘点：比较两张图各自有多少可用数据，为"收敛到一张图"提供依据。

分两块看，因为它们的性质完全不同：

  * 几何层（构形数据）—— 只要有地图包就能拿到，两张图都有；
  * 行为层（demo 数据）—— 需要录像，**只有 de_mirage 有**。

用法：
    .venv\\Scripts\\python.exe data_inventory.py
    .venv\\Scripts\\python.exe data_inventory.py de_dust2 de_mirage

输出：out\\data_inventory.csv，并在终端打印可直接贴进文档的 Markdown 表。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from navmesh import NavMesh

ROOT = Path(__file__).resolve().parent

# 1 Source 单位 ≈ 1.905 cm，换算面积用
UNIT2_TO_M2 = 0.01905 ** 2


def polygon_area_xy(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def find_out_file(map_name: str, suffix: str) -> Path | None:
    """主线图的产物在 out\\ 下，附录图在 out\\appendix_*\\ 下。"""
    candidates = [ROOT / "out" / f"{map_name}{suffix}"]
    candidates += sorted(ROOT.glob(f"out/appendix_*/{map_name}{suffix}"))
    for path in candidates:
        if path.exists():
            return path
    return None


def geometry_stats(map_name: str) -> dict:
    npz = ROOT / "data" / "game_nav" / f"{map_name}_navmesh.npz"
    nav = NavMesh.from_npz(npz)
    area = sum(polygon_area_xy(nav.corners[idx][:, :2]) for idx in nav.polygons)
    lo, hi = nav.bounds()

    csv = find_out_file(map_name, "_syntax_grid.csv")
    meta = find_out_file(map_name, "_syntax_meta.json")
    stats = {
        "地图": map_name,
        "导航多边形数": len(nav.polygons),
        "导航角点数": len(nav.corners),
        "可通行面积(m²)": round(area * UNIT2_TO_M2, 1),
        "世界跨距(单位)": f"{hi[0]-lo[0]:.0f} × {hi[1]-lo[1]:.0f}",
        "句法栅格数": None,
        "可通行格子数": None,
        "格子边长(游戏单位)": None,
        "平均连通度": None,
        "整合度中位(拓扑)": None,
        "选择度前5%占比": None,
    }
    if csv is not None:
        df = pl.read_csv(csv)
        stats["可通行格子数"] = df.height
        stats["整合度中位(拓扑)"] = round(
            float(np.median(df["integration_hh_t"].to_numpy().astype(float))), 3)
        choice = np.sort(df["choice_t"].to_numpy().astype(float))[::-1]
        top = max(int(len(choice) * 0.05), 1)
        stats["选择度前5%占比"] = round(float(choice[:top].sum() / choice.sum()), 4)
    if meta is not None:
        m = json.loads(meta.read_text(encoding="utf-8"))
        stats["句法栅格数"] = f"{m['grid_cols']}×{m['grid_rows']}"
        stats["格子边长(游戏单位)"] = m["cell_world_units"]
        stats["平均连通度"] = m["visibility_mean_degree"]
    return stats


def behaviour_stats(map_name: str) -> dict:
    stats = {
        "地图": map_name,
        "demo 场次": 0,
        "回合数": 0,
        "击杀数": 0,
        "轨迹采样点": 0,
        "demo 体积(MB)": 0,
        "数据来源": "—",
        "格间移动次数": 0,
        "有行为的格子数": 0,
        "可通行格子覆盖率": None,
    }
    # 首选 ingest_demos.py 生成的按地图分表；没有才退回旧的单 demo 布局
    per_map = ROOT / "out" / "behaviour" / f"{map_name}_meta.json"
    legacy = ROOT / "out" / "tracks_meta.json"
    if per_map.exists():
        meta = json.loads(per_map.read_text(encoding="utf-8"))
    elif legacy.exists() and json.loads(legacy.read_text(encoding="utf-8")).get("map_name") == map_name:
        meta = json.loads(legacy.read_text(encoding="utf-8"))
        meta.setdefault("kills", 0)
    else:
        meta = None
    if meta:
        stats["demo 场次"] = meta.get("demo_count", 1 if meta.get("demos") else 0)
        stats["回合数"] = meta.get("rounds", 0)
        stats["击杀数"] = meta.get("kills", 0)
        stats["轨迹采样点"] = meta.get("rows", 0)
        stats["demo 体积(MB)"] = round(meta.get("bytes_total", 0) / 1e6, 1)
        demos = meta.get("demos") or []
        roots = {Path(d["path"]).parts[0] if Path(d["path"]).parts else "?" for d in demos}
        stats["数据来源"] = "本机 demos\\" if roots == {"demos"} else ", ".join(sorted(roots))
    grid = ROOT / "out" / f"{map_name}_behaviour_grid.csv"
    if grid.exists():
        df = pl.read_csv(grid)
        # flow = 每格"进 + 出"次数，所以一次移动被计了两次，除以 2 才是步数
        stats["格间移动次数"] = int(df["flow"].sum() / 2)
        stats["有行为的格子数"] = int((df["occ_all"] > 0).sum())
        stats["可通行格子覆盖率"] = round(
            float((df["occ_all"] > 0).sum() / df.height), 4)
    return stats


def markdown_table(rows: list[dict]) -> str:
    keys = [k for k in rows[0]]
    head = "| 口径 | " + " | ".join(r["地图"] for r in rows) + " |"
    sep = "|---|" + "---|" * len(rows)
    body = []
    for key in keys:
        if key == "地图":
            continue
        cells = []
        for r in rows:
            v = r[key]
            cells.append("—" if v in (None, "") else f"{v:,}" if isinstance(v, int) else str(v))
        body.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


def main() -> None:
    maps = sys.argv[1:] or ["de_dust2", "de_mirage"]
    geo_rows, beh_rows = [], []
    for name in maps:
        try:
            geo_rows.append(geometry_stats(name))
        except FileNotFoundError as exc:
            print(f"跳过 {name} 的几何层：{exc}")
        beh_rows.append(behaviour_stats(name))

    print("## 几何层（构形数据）\n")
    print(markdown_table(geo_rows))
    print("\n## 行为层（demo 数据）\n")
    print(markdown_table(beh_rows))

    merged = []
    for g, b in zip(geo_rows, beh_rows):
        merged.append({**g, **{k: v for k, v in b.items() if k != "地图"}})
    pl.DataFrame(merged).write_csv(ROOT / "out" / "data_inventory.csv")
    print("\nwrote out/data_inventory.csv")

    print("\n## 体量比（dust2 / mirage）")
    if len(geo_rows) == 2:
        g0, g1 = geo_rows
        for key in ("导航多边形数", "导航角点数", "可通行面积(m²)", "可通行格子数"):
            a, b = g0.get(key), g1.get(key)
            if a and b:
                print(f"  {key}: {a:,} / {b:,} = {a / b:.2f}×")
        print(f"  行为数据: {beh_rows[0]['轨迹采样点']:,} / {beh_rows[1]['轨迹采样点']:,}")


if __name__ == "__main__":
    main()
