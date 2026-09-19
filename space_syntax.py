r"""在 CS2 地图上做空间句法的 VGA（可见性图分析），不依赖 depthmapX。

为什么自己实现
--------------
depthmapX 的 VGA 需要手工把墙描到图上。CS2 的地图是三维多层结构，手工描图
既费时又不可复现。这里改用两条更靠得住的输入：

  1. 可通行空间 —— 来自游戏自带的导航网格（navmesh.py 解析），是权威定义；
  2. 遮挡关系   —— 用"可通行 / 不可通行"二值栅格近似：不可通行的格子当作墙。
     这是二维平面近似的标准做法，代价是叠层空间（上下两层在俯视图里重叠）
     会被压平，这一条写进报告的限制里。

计算内容（对齐 depthmapX 的 VGA 输出）
-------------------------------------
  connectivity      每个格子的可见格子数（可直接理解为连通度）
  isovist_area_px   等视域面积（可见格子数 × 格面积）
  integration_hh_t  拓扑整合度 Integration[HH]（按可见性图的步数距离）
  integration_hh_m  米制整合度（按可见性图的欧氏距离）
  choice_t          拓扑选择度（可见性图上的介数中心性）

用法：
    .venv\\Scripts\\python.exe space_syntax.py de_dust2
    .venv\\Scripts\\python.exe space_syntax.py de_mirage --cell 16

输出：
    out\\<map>_syntax_grid.csv    逐格指标（含像素/世界坐标边界，供与行为数据 join）
    out\\<map>_syntax_maps.png    四联图：可通行区 / 整合度 / 选择度 / 等视域面积
    out\\<map>_syntax_meta.json   参数与统计，写报告时引用
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw
from scipy import sparse
from scipy.sparse import csgraph
from scipy.ndimage import binary_closing, binary_erosion

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from heatmap import MapTransform, detect_transform, read_overview_all  # noqa: E402
from navmesh import NavMesh  # noqa: E402


# ------------------------------------------------------------------ 可通行掩码

def rasterise_walkable(nav: NavMesh, tf: MapTransform, cell: int,
                       width: int, height: int, erode: int = 1) -> np.ndarray:
    """把导航网格多边形栅格化成 (rows, cols) 的布尔掩码。

    cell  : 一个格子多少像素（格子越小越精细，但计算量按平方增长）
    erode : 向内腐蚀多少个格子。玩家有体积，贴着墙的格子可视作不可站，
            腐蚀一格能让"视点"落在真正能站人的位置。
    """
    cols = width // cell
    rows = height // cell
    canvas = Image.new("L", (cols, rows), 0)
    dr = ImageDraw.Draw(canvas)
    for indices in nav.polygons:
        pts = nav.corners[indices]
        px, py = tf.world_to_radar(pts[:, 0], pts[:, 1])
        # 雷达像素 -> 格子坐标
        dr.polygon(list(zip((px / cell).tolist(), (py / cell).tolist())), fill=255)
    mask = np.array(canvas) > 0
    # 先把多边形之间 1 格宽的缝补上（导航网格多边形是共边的，栅格化偶有细缝）
    mask = binary_closing(mask, np.ones((3, 3)))
    if erode > 0:
        mask = binary_erosion(mask, np.ones((3, 3)), iterations=erode)
    return mask


# ------------------------------------------------------------------ 可见性图

def visibility_graph(mask: np.ndarray, verbose: bool = True) -> np.ndarray:
    """逐格射线求可见性，返回 n×n 的布尔邻接矩阵（n = 可通行格子数）。

    射线用"沿直线等步长取样 + 四舍五入落到格心"实现，等价于栅格上的
    Bresenham。为了速度，所有目标格子一起向量化推进，而不是逐对判。
    """
    ys, xs = np.nonzero(mask)
    n = len(ys)
    adj = np.zeros((n, n), dtype=bool)
    t0 = time.time()
    for i in range(n):
        dy = ys - ys[i]
        dx = xs - xs[i]
        steps = np.maximum(np.abs(dy), np.abs(dx))
        blocked = np.zeros(n, dtype=bool)
        max_step = int(steps.max()) if n else 0
        safe = np.maximum(steps, 1)
        for k in range(1, max_step + 1):
            active = steps >= k
            if not active.any():
                break
            t = k / safe
            cy = np.clip(np.rint(ys[i] + dy * t), 0, mask.shape[0] - 1).astype(np.int32)
            cx = np.clip(np.rint(xs[i] + dx * t), 0, mask.shape[1] - 1).astype(np.int32)
            blocked |= active & ~mask[cy, cx]
        adj[i] = (~blocked) & (steps > 0)
        if verbose and n >= 500 and (i + 1) % 500 == 0:
            print(f"    可见性图 {i + 1}/{n} 源，已用 {time.time() - t0:.0f}s")
    return adj


def isovist_perimeter(mask: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """等视域的"周长"近似：可见格子中，有多少个的邻居格不可见。

    这是 depthmapX 里 isovist perimeter 的栅格离散版本，用来刻画视域的碎形程度。
    """
    ys, xs = np.nonzero(mask)
    n = len(ys)
    # 把可见矩阵还原成 (rows, cols) 的逐源可见图
    rows, cols = mask.shape
    perim = np.zeros(n)
    diffs = ((0, 1), (1, 0), (1, 1), (1, -1))
    for i in range(n):
        vis = np.zeros((rows, cols), dtype=bool)
        vis[ys[adj[i]], xs[adj[i]]] = True
        vis[ys[i], xs[i]] = True
        total = 0
        for dy, dx in diffs:
            shifted = np.zeros_like(vis)
            ys0, ys1 = max(0, dy), rows + min(0, dy)
            xs0, xs1 = max(0, dx), cols + min(0, dx)
            shifted[ys0:ys1, xs0:xs1] = vis[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
            total += int((vis & ~shifted).sum())
        perim[i] = total
    return perim


# ------------------------------------------------------------------ 句法指标

def integration_hh(dist: np.ndarray) -> np.ndarray:
    """Integration[HH]：把平均最短距离换算成整合度（Hillier & Hanson 标准式）。

    dist : (n, n) 距离矩阵，对角线应为 0。距离可以是步数（拓扑）或米（米制）。
    """
    n = dist.shape[0]
    if n < 4:
        return np.full(n, np.nan)
    total = dist.sum(axis=1)
    md = total / (n - 1)
    ra = 2.0 * (md - 1.0) / (n - 2.0)
    # 理论上的"菱形图"基准值 Dn，用于把不同规模图的 RA 标准化
    dn = 2.0 * (n * (np.log2((n + 2) / 3.0) - 1.0) + 1.0) / ((n - 1.0) * (n - 2.0))
    rra = ra / dn
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / rra


def brandes_unweighted(adj_csr: sparse.csr_matrix, sources: np.ndarray,
                       verbose: bool = True) -> np.ndarray:
    """无权图上的介数中心性（Brandes 算法），用稀疏矩阵 + bincount 向量化。

    选择度就是介数中心性：一个格子落在多少条最短路径上。depthmapX 的
    "Choice" 在拓扑模式下即此定义。
    """
    n = adj_csr.shape[0]
    between = np.zeros(n, dtype=np.float64)
    indptr, indices = adj_csr.indptr, adj_csr.indices
    t0 = time.time()
    for si, s in enumerate(sources):
        dist = np.full(n, -1, dtype=np.int32)
        sigma = np.zeros(n, dtype=np.float64)
        dist[s] = 0
        sigma[s] = 1.0
        frontier = np.array([s], dtype=np.int64)
        levels = []
        while frontier.size:
            levels.append(frontier)
            starts = indptr[frontier]
            counts = indptr[frontier + 1] - starts
            total = int(counts.sum())
            if total == 0:
                break
            # 把 frontier 里每个点的邻居拼成一个数组（等价的 repeat/cumsum 展开）
            offsets = np.repeat(starts, counts)
            inner = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
            src = np.repeat(frontier, counts)
            nbr = indices[offsets + inner]
            uniq, inv = np.unique(nbr, return_inverse=True)
            contrib = np.bincount(inv, weights=sigma[src], minlength=len(uniq))
            fresh = uniq[dist[uniq] < 0]
            if fresh.size == 0:
                break
            dist[fresh] = len(levels)
            sel = np.isin(uniq, fresh)
            sigma[fresh] = contrib[sel]
            frontier = fresh
        delta = np.zeros(n, dtype=np.float64)
        for level in reversed(levels[1:]):
            starts = indptr[level]
            counts = indptr[level + 1] - starts
            total = int(counts.sum())
            if total:
                offsets = np.repeat(starts, counts)
                inner = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
                src_local = np.repeat(np.arange(level.size), counts)
                dst = indices[offsets + inner]
                # Brandes 只累加"后继"（距离恰好 +1 的邻居）。同一层之间的边
                # 不是最短路径的一部分，算进去会重复计数——这一点在完全图
                # （例如一条直走廊）里会暴露成非零的选择度。
                successor = dist[dst] == dist[level][src_local] + 1
                val = np.zeros(total, dtype=np.float64)
                if successor.any():
                    s_idx = src_local[successor]
                    d_idx = dst[successor]
                    val[successor] = (sigma[level][s_idx] / sigma[d_idx]
                                      * (1.0 + delta[d_idx]))
                delta[level] = np.bincount(src_local, weights=val, minlength=level.size)
            between[level] += delta[level]
        if verbose and len(sources) >= 500 and (si + 1) % 250 == 0:
            print(f"    选择度 {si + 1}/{len(sources)} 源，已用 {time.time() - t0:.0f}s")
    # 无向图：每条最短路被正反各数一次，除 2
    return between / 2.0


# ------------------------------------------------------------------ 主流程

def compute(map_name: str, cell: int, choice_sources: int | None,
            erode: int, skip_perimeter: bool) -> dict:
    root = Path(__file__).resolve().parent
    npz = root / "data" / "game_nav" / f"{map_name}_navmesh.npz"
    if npz.exists():
        nav = NavMesh.from_npz(npz)
    else:
        from navmesh import read_nav
        nav = read_nav(root / "data" / "game_nav" / f"{map_name}.nav")

    radar_path = root / "data" / f"{map_name}.png"
    overview = read_overview_all(root / "data" / f"{map_name}.txt")
    radar = Image.open(radar_path).convert("RGBA")
    width, height = radar.size
    tf = detect_transform(overview, np.array(radar), nav)

    t0 = time.time()
    mask = rasterise_walkable(nav, tf, cell, width, height, erode=erode)
    rows, cols = mask.shape
    n_walk = int(mask.sum())
    cell_area_units = (cell * tf.scale) ** 2
    print(f"{map_name}: 栅格 {cols}×{rows}（每格 {cell}px ≈ {cell * tf.scale:.0f} 游戏单位），"
          f"可通行格子 {n_walk:,} 个（占栅格 {n_walk / mask.size * 100:.1f}%）")
    if n_walk == 0:
        raise SystemExit("可通行格子为 0，检查 navmesh 与雷达图是否对得上")

    print("计算可见性图 ...")
    adj = visibility_graph(mask)
    np.fill_diagonal(adj, False)
    adj_sym = adj | adj.T          # 可见性是对称关系，取并集消除射线取样误差
    degree = adj_sym.sum(axis=1).astype(np.int64)
    print(f"  可见性图完成，{time.time() - t0:.0f}s；平均连通度 {degree.mean():.1f}，"
          f"最大 {degree.max()}")

    ys, xs = np.nonzero(mask)
    coords = np.stack([ys, xs], axis=1).astype(np.float64) * cell + cell / 2.0
    csr = sparse.csr_matrix(adj_sym)
    # 注意：必须转成浮点再写权重，布尔矩阵的 data 是 bool，写进去会被截成 1/0
    weights = csr.astype(np.float64).copy()
    # 边权 = 格心之间的欧氏距离（像素）
    for i in range(csr.shape[0]):
        start, end = csr.indptr[i], csr.indptr[i + 1]
        if start == end:
            continue
        cols_idx = csr.indices[start:end]
        d = np.hypot(coords[cols_idx, 0] - coords[i, 0], coords[cols_idx, 1] - coords[i, 1])
        weights.data[start:end] = d

    print("计算拓扑距离矩阵（BFS/Dijkstra，全部源点）...")
    t1 = time.time()
    dist_t = csgraph.dijkstra(csr, unweighted=True, indices=np.arange(csr.shape[0]))
    print(f"  {time.time() - t1:.0f}s")
    print("计算米制距离矩阵 ...")
    t1 = time.time()
    dist_m = csgraph.dijkstra(weights, indices=np.arange(csr.shape[0]))
    print(f"  {time.time() - t1:.0f}s")

    # 不可达的点对用该矩阵的最大有限值兜底，避免 NaN 污染整合度
    for d in (dist_t, dist_m):
        finite = np.isfinite(d)
        fill = d[finite].max() if finite.any() else 1.0
        d[~finite] = fill
    np.fill_diagonal(dist_t, 0.0)
    np.fill_diagonal(dist_m, 0.0)

    integ_t = integration_hh(dist_t)
    integ_m = integration_hh(dist_m)

    if choice_sources is None or choice_sources >= n_walk:
        sources = np.arange(n_walk)
    else:
        rng = np.random.default_rng(20260919)
        sources = np.sort(rng.choice(n_walk, size=choice_sources, replace=False))
    print(f"计算选择度（介数中心性，{len(sources)} 个源点）...")
    t1 = time.time()
    choice_t = brandes_unweighted(csr, sources)
    if len(sources) < n_walk:
        choice_t *= n_walk / len(sources)
    print(f"  {time.time() - t1:.0f}s")

    perimeter = np.zeros(n_walk)
    if not skip_perimeter:
        print("计算等视域周长 ...")
        perimeter = isovist_perimeter(mask, adj_sym)

    # 指标搬到整张栅格上，便于出图和导出
    grid = {}
    for key, values, fill in (
        ("connectivity", degree.astype(np.float64), np.nan),
        ("isovist_area_px", degree.astype(np.float64) * cell * cell, np.nan),
        ("isovist_perimeter", perimeter, np.nan),
        ("integration_hh_t", integ_t, np.nan),
        ("integration_hh_m", integ_m, np.nan),
        ("choice_t", choice_t, np.nan),
    ):
        layer = np.full((rows, cols), fill, dtype=np.float64)
        layer[ys, xs] = values
        grid[key] = layer

    meta = {
        "map": map_name,
        "cell_px": cell,
        "cell_world_units": round(cell * tf.scale, 2),
        "grid_cols": cols,
        "grid_rows": rows,
        "walkable_cells": n_walk,
        "walkable_share": round(n_walk / mask.size, 4),
        "transform_form": tf.form,
        "transform": {"pos_x": overview["pos_x"], "pos_y": overview["pos_y"],
                      "scale": overview["scale"], "rotate": overview.get("rotate", 0)},
        "visibility_mean_degree": round(float(degree.mean()), 2),
        "visibility_max_degree": int(degree.max()),
        "choice_sources": int(len(sources)),
        "elapsed_seconds": round(time.time() - t0, 1),
        "metrics": {},
    }
    for key, layer in grid.items():
        values = layer[np.isfinite(layer)]
        meta["metrics"][key] = {
            "min": round(float(values.min()), 4),
            "median": round(float(np.median(values)), 4),
            "mean": round(float(values.mean()), 4),
            "max": round(float(values.max()), 4),
        }
    # 交叉检查：整合度与等视域面积、拓扑与米制整合度应当高度相关
    flat = {k: v[np.isfinite(v)] for k, v in grid.items()}
    if len(flat["integration_hh_t"]) > 10:
        meta["sanity_correlation"] = {
            "integration_t_vs_m": round(float(np.corrcoef(
                flat["integration_hh_t"], flat["integration_hh_m"])[0, 1]), 4),
            "integration_t_vs_isovist_area": round(float(np.corrcoef(
                flat["integration_hh_t"], flat["isovist_area_px"])[0, 1]), 4),
            "connectivity_vs_isovist_area": round(float(np.corrcoef(
                flat["connectivity"], flat["isovist_area_px"])[0, 1]), 4),
            "integration_t_vs_choice_t": round(float(np.corrcoef(
                flat["integration_hh_t"], flat["choice_t"])[0, 1]), 4),
        }

    out_dir = root / "out"
    out_dir.mkdir(exist_ok=True)
    export_grid_csv(map_name, grid, mask, tf, cell, rows, cols, out_dir)
    draw_maps(map_name, radar, grid, mask, tf, cell, rows, cols, out_dir)
    with open(out_dir / f"{map_name}_syntax_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta["sanity_correlation"], ensure_ascii=False, indent=2))
    print(f"总耗时 {meta['elapsed_seconds']}s；写出 "
          f"{map_name}_syntax_grid.csv / _syntax_maps.png / _syntax_meta.json")
    return meta


def export_grid_csv(map_name: str, grid: dict, mask: np.ndarray, tf: MapTransform,
                    cell: int, rows: int, cols: int, out_dir: Path) -> None:
    """逐格导出，后面和行为数据 join 就靠这个表。"""
    lines = ["col,row,px_left,px_top,px_right,px_bottom,"
             "world_x_min,world_x_max,world_y_min,world_y_max,walkable,"
             "connectivity,isovist_area_px,isovist_perimeter,"
             "integration_hh_t,integration_hh_m,choice_t"]
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c]:
                continue
            px0, py0 = c * cell, r * cell
            px1, py1 = px0 + cell, py0 + cell
            wx0, wy0 = tf.radar_to_world(np.array([px0]), np.array([py0]))
            wx1, wy1 = tf.radar_to_world(np.array([px1]), np.array([py1]))
            lines.append(
                f"{c},{r},{px0},{py0},{px1},{py1},"
                f"{float(wx0[0]):.1f},{float(wx1[0]):.1f},{float(wy1[0]):.1f},{float(wy0[0]):.1f},1,"
                f"{int(grid['connectivity'][r, c])},"
                f"{grid['isovist_area_px'][r, c]:.1f},"
                f"{grid['isovist_perimeter'][r, c]:.1f},"
                f"{grid['integration_hh_t'][r, c]:.4f},"
                f"{grid['integration_hh_m'][r, c]:.4f},"
                f"{grid['choice_t'][r, c]:.4f}"
            )
    path = out_dir / f"{map_name}_syntax_grid.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(lines) - 1} 个可通行格子)")


def draw_maps(map_name: str, radar: Image.Image, grid: dict, mask: np.ndarray,
              tf: MapTransform, cell: int, rows: int, cols: int, out_dir: Path) -> None:
    """四联图：可通行区、整合度、(选择度)、等视域面积，叠在雷达图上。"""
    panels = [
        ("walkable", "可通行空间（导航网格）", "Greens"),
        ("integration_hh_t", "整合度 Integration[HH]（拓扑）", "turbo"),
        ("choice_t", "选择度 Choice（拓扑）", "magma"),
        ("isovist_area_px", "等视域面积 Isovist area", "viridis"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(26, 7.6), dpi=100)
    radar_arr = np.array(radar)
    for ax, (key, title, cmap) in zip(axes, panels):
        ax.imshow(radar_arr, extent=[0, cols * cell, rows * cell, 0])
        if key == "walkable":
            layer = np.where(mask, 1.0, np.nan)
            ax.imshow(np.ma.masked_invalid(layer), extent=[0, cols * cell, rows * cell, 0],
                      cmap=cmap, alpha=0.75, interpolation="nearest")
        else:
            layer = grid[key].copy()
            # 用百分位裁剪，避免极值把色阶压平
            lo, hi = np.nanpercentile(layer, [2, 98])
            ax.imshow(np.ma.masked_invalid(layer), extent=[0, cols * cell, rows * cell, 0],
                      cmap=cmap, alpha=0.75, vmin=lo, vmax=hi)
        ax.set_title(f"{title}\n{map_name}", fontsize=12)
        ax.set_xlim(0, cols * cell)
        ax.set_ylim(rows * cell, 0)
        ax.set_axis_off()
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig.tight_layout()
    path = out_dir / f"{map_name}_syntax_maps.png"
    fig.savefig(path, facecolor="black")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("map")
    ap.add_argument("--cell", type=int, default=16, help="格子边长（像素），默认 16")
    ap.add_argument("--choice-sources", type=int, default=None,
                    help="选择度的源点采样数，默认全部")
    ap.add_argument("--erode", type=int, default=1, help="可通行区向内腐蚀的格数，默认 1")
    ap.add_argument("--skip-perimeter", action="store_true")
    args = ap.parse_args()
    compute(args.map, args.cell, args.choice_sources, args.erode, args.skip_perimeter)


if __name__ == "__main__":
    main()
