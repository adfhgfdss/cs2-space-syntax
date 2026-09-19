r"""自动生成轴线图（axial map），供 depthmapX 做轴向分析。

背景：传统轴线图是人工在 CAD 里描的——用最少、最长的直线覆盖全部凸空间。
这台机器上没有人工描图的条件，所以这里用可复现的算法近似它：

  1. 在可通行栅格上，沿 16 个方向生成"极大直线段"（两端都顶到障碍为止）；
  2. 去掉重复线，按"覆盖新格子数"贪心挑线，直到覆盖 99% 的可通行格子
     （这一步对应传统做法的"最少线数"要求）；
  3. 把共线且相接的线合并成更长的线；
  4. 导出成 DXF，交给 depthmapX 的 `MAPCONVERT -co axial` + `AXIAL` 计算
     整合度 / 选择度——**分析引擎是 depthmapX 本体**，只有几何生成是自动的。

用法（一般由 depthmap_analysis.py 调用）：
    python axial_map.py de_dust2 --cell 4 --out out/de_dust2_axial.dxf
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from heatmap import detect_transform, read_overview_all
from navmesh import NavMesh
from space_syntax import rasterise_walkable

DIRECTIONS = 16          # 生成方向数（每 22.5° 一条）
MAX_STEPS = 400          # 单方向最长推进步数


def walkable_mask(map_name: str, cell: int, erode: int = 0) -> tuple[np.ndarray, object, Image.Image]:
    nav = NavMesh.from_npz(Path("data/game_nav") / f"{map_name}_navmesh.npz")
    overview = read_overview_all(Path("data") / f"{map_name}.txt")
    radar = Image.open(Path("data") / f"{map_name}.png").convert("RGBA")
    tf = detect_transform(overview, np.array(radar), nav, verbose=False)
    mask = rasterise_walkable(nav, tf, cell, *radar.size, erode=erode)
    return mask, tf, radar


def maximal_runs(mask: np.ndarray, cell_px: int) -> list[tuple[float, float, float, float]]:
    """沿 16 个方向求极大直线段（两端顶到障碍），返回雷达像素坐标的线段。"""
    rows, cols = mask.shape
    ys, xs = np.nonzero(mask)
    origin_y = ys.astype(np.float64)
    origin_x = xs.astype(np.float64)
    height, width = mask.shape
    segments: dict[tuple, tuple] = {}

    for k in range(DIRECTIONS):
        angle = math.pi * k / DIRECTIONS
        dy, dx = math.sin(angle), math.cos(angle)
        # 正向 / 反向各自推进，记录能走多远
        reach_f = np.zeros(len(ys), dtype=np.int32)
        reach_b = np.zeros(len(ys), dtype=np.int32)
        cur_y, cur_x = origin_y.copy(), origin_x.copy()
        alive = np.ones(len(ys), dtype=bool)
        for step in range(1, MAX_STEPS + 1):
            if not alive.any():
                break
            cy = np.clip(np.rint(cur_y + dy), 0, height - 1).astype(np.int32)
            cx = np.clip(np.rint(cur_x + dx), 0, width - 1).astype(np.int32)
            ok = alive & mask[cy, cx] & (np.abs(cy - cur_y) + np.abs(cx - cur_x) > 0)
            reach_f[ok] = step
            alive &= ok
            cur_y, cur_x = np.where(alive, cy, cur_y), np.where(alive, cx, cur_x)
        cur_y, cur_x = origin_y.copy(), origin_x.copy()
        alive = np.ones(len(ys), dtype=bool)
        for step in range(1, MAX_STEPS + 1):
            if not alive.any():
                break
            cy = np.clip(np.rint(cur_y - dy), 0, height - 1).astype(np.int32)
            cx = np.clip(np.rint(cur_x - dx), 0, width - 1).astype(np.int32)
            ok = alive & mask[cy, cx] & (np.abs(cy - cur_y) + np.abs(cx - cur_x) > 0)
            reach_b[ok] = step
            alive &= ok
            cur_y, cur_x = np.where(alive, cy, cur_y), np.where(alive, cx, cur_x)

        # 只保留"自己是这组格子上最长"的那条（按起点去重，留下 reach 最大的）
        key_base = (k % (DIRECTIONS // 2),)  # 反向等价，用半圈去重
        order = np.argsort(-(reach_f + reach_b))
        seen: dict[tuple[int, int, int], int] = {}
        for idx in order:
            if reach_f[idx] + reach_b[idx] < 8:      # 太短的线不要
                continue
            y0 = int(round(origin_y[idx] - dy * reach_b[idx]))
            x0 = int(round(origin_x[idx] - dx * reach_b[idx]))
            y1 = int(round(origin_y[idx] + dy * reach_f[idx]))
            x1 = int(round(origin_x[idx] + dx * reach_f[idx]))
            # 端点排序 + 4px 量化去重
            if (y0, x0) > (y1, x1):
                (y0, x0), (y1, x1) = (y1, x1), (y0, x0)
            key = (*key_base, y0 // 2, x0 // 2, y1 // 2, x1 // 2)
            if key in seen:
                continue
            seen[key] = idx
            seg = (x0 * cell_px, y0 * cell_px, x1 * cell_px, y1 * cell_px)
            segments[seg] = seg
    return list(segments.values())


def cover_and_merge(mask: np.ndarray, segments: list[tuple], cell_px: int,
                    target: float = 0.99, max_lines: int = 400) -> list[tuple]:
    """贪心选线（覆盖尽量多的可通行格子）+ 共线合并。"""
    height, width = mask.shape
    flat_index = np.full((height, width), -1, dtype=np.int64)
    ys, xs = np.nonzero(mask)
    flat_index[ys, xs] = np.arange(len(ys))
    total = len(ys)

    # 预先算出每条线覆盖的格子
    line_cells = []
    for (x1, y1, x2, y2) in segments:
        length = math.hypot(x2 - x1, y2 - y1) / cell_px
        n = max(int(length) + 1, 2)
        t = np.linspace(0, 1, n)
        gy = np.clip(np.rint((y1 + (y2 - y1) * t) / cell_px).astype(int), 0, height - 1)
        gx = np.clip(np.rint((x1 + (x2 - x1) * t) / cell_px).astype(int), 0, width - 1)
        idx = flat_index[gy, gx]
        idx = np.unique(idx[idx >= 0])
        if idx.size:
            line_cells.append((idx, (x1, y1, x2, y2)))
    # 长线优先，减少候选
    line_cells.sort(key=lambda item: -item[0].size)
    line_cells = line_cells[:4000]

    covered = np.zeros(total, dtype=bool)
    chosen: list[tuple] = []
    remaining = line_cells
    while remaining and len(chosen) < max_lines:
        best = None
        best_gain = 0
        for idx, seg in remaining:
            gain = int((~covered[idx]).sum())
            if gain > best_gain:
                best_gain, best = gain, (idx, seg)
        if best is None or best_gain < 4:
            break
        idx, seg = best
        covered[idx] = True
        chosen.append(seg)
        remaining = [item for item in remaining if item[1] is not seg]
        if covered.mean() >= target:
            break
    return chosen


def merge_collinear(segments: list[tuple], angle_tol: float = 2.0,
                    dist_tol: float = 3.0) -> list[tuple]:
    """把方向相近、首尾相接的线段合并成更长的线。"""
    groups: list[list[tuple]] = []
    for seg in segments:
        x1, y1, x2, y2 = seg
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        placed = False
        for g in groups:
            gx1, gy1, gx2, gy2 = g[0]
            gang = math.degrees(math.atan2(gy2 - gy1, gx2 - gx1)) % 180.0
            if min(abs(ang - gang), 180 - abs(ang - gang)) < angle_tol:
                # 端点到该直线的距离
                vx, vy = gx2 - gx1, gy2 - gy1
                norm = math.hypot(vx, vy)
                if norm == 0:
                    continue
                d1 = abs((x1 - gx1) * vy - (y1 - gy1) * vx) / norm
                d2 = abs((x2 - gx1) * vy - (y2 - gy1) * vx) / norm
                if max(d1, d2) < dist_tol:
                    g.append(seg)
                    placed = True
                    break
        if not placed:
            groups.append([seg])

    merged = []
    for g in groups:
        pts = []
        for (x1, y1, x2, y2) in g:
            pts.append((x1, y1))
            pts.append((x2, y2))
        # 取投影到主方向上的两个极值点
        x1, y1, x2, y2 = g[0]
        vx, vy = x2 - x1, y2 - y1
        norm = math.hypot(vx, vy) or 1.0
        ux, uy = vx / norm, vy / norm
        proj = [(px * ux + py * uy, px, py) for px, py in pts]
        proj.sort()
        _, ax, ay = proj[0]
        _, bx, by = proj[-1]
        merged.append((ax, ay, bx, by))
    return merged


def write_dxf(segments: list[tuple], path: Path) -> None:
    """写最小 DXF（R12，LINE 实体），depthmapX 可以直接 IMPORT。"""
    out = ["0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1009",
           "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]
    for (x1, y1, x2, y2) in segments:
        out += ["0", "LINE", "8", "0",
                "10", f"{x1:.3f}", "20", f"{y1:.3f}",
                "11", f"{x2:.3f}", "21", f"{y2:.3f}"]
    out += ["0", "ENDSEC", "0", "EOF"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def build(map_name: str, cell: int = 4, max_lines: int = 400) -> dict:
    mask, tf, radar = walkable_mask(map_name, cell)
    seg = skeleton_lines(mask, cell)
    print(f"  {map_name}: 骨架折线简化后 {len(seg)} 条轴线")
    return {"mask": mask, "transform": tf, "radar": radar, "segments": seg}


def _thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen 细化：把可通行区域压成 1 像素宽的骨架。"""
    img = mask.astype(np.uint8).copy()
    changed = True
    while changed:
        changed = False
        for phase in (0, 1):
            p = np.pad(img, 1)
            p2, p3, p4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
            p5, p6, p7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
            p8, p9 = p[1:-1, :-2], p[:-2, :-2]
            nb = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            trans = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if phase == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (img == 1) & (nb >= 2) & (nb <= 6) & (trans == 1) & cond
            if remove.any():
                img[remove] = 0
                changed = True
    return img.astype(bool)


def _simplify(points: list[tuple[int, int]], tol: float = 2.5) -> list[tuple[int, int]]:
    """Douglas-Peucker 折线简化。"""
    if len(points) < 3:
        return points
    (x1, y1), (x2, y2) = points[0], points[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = math.hypot(dx, dy) or 1.0
    best, dist_max = 0, 0.0
    for i in range(1, len(points) - 1):
        px, py = points[i]
        d = abs((px - x1) * dy - (py - y1) * dx) / norm
        if d > dist_max:
            best, dist_max = i, d
    if dist_max > tol:
        left = _simplify(points[:best + 1], tol)
        right = _simplify(points[best:], tol)
        return left[:-1] + right
    return [points[0], points[-1]]


def skeleton_lines(mask: np.ndarray, cell_px: int) -> list[tuple[float, float, float, float]]:
    """骨架化 → 路径追踪 → 折线简化 → 共线合并，得到紧凑的轴线段集。"""
    # 只保留足够大的连通区域，去掉零星小岛（避免生成孤立轴线）
    from scipy import ndimage

    labels, n_labels = ndimage.label(mask)
    if n_labels > 1:
        sizes = ndimage.sum(mask, labels, range(1, n_labels + 1))
        keep = {i + 1 for i, size in enumerate(sizes) if size >= 0.01 * mask.sum()}
        mask = np.isin(labels, list(keep))

    skel = _thin(mask)
    ys, xs = np.nonzero(skel)
    if len(ys) == 0:
        return []
    cell_set = set(zip(xs.tolist(), ys.tolist()))
    neighbours = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

    def nbrs(pt):
        x, y = pt
        return [(x + dx, y + dy) for dx, dy in neighbours if (x + dx, y + dy) in cell_set]

    degree = {pt: len(nbrs(pt)) for pt in cell_set}
    nodes = {pt for pt, d in degree.items() if d != 2}      # 端点与交叉点
    visited_edges: set[frozenset] = set()
    paths: list[list[tuple[int, int]]] = []

    def walk(start, nxt):
        path = [start, nxt]
        prev, cur = start, nxt
        while cur not in nodes:
            step = [p for p in nbrs(cur) if p != prev]
            if not step:
                break
            prev, cur = cur, step[0]
            path.append(cur)
        return path

    for node in nodes:
        for nxt in nbrs(node):
            edge = frozenset((node, nxt))
            if edge in visited_edges:
                continue
            path = walk(node, nxt)
            for a, b in zip(path, path[1:]):
                visited_edges.add(frozenset((a, b)))
            paths.append(path)
    # 没被访问到的环
    for pt in cell_set:
        if degree[pt] == 2:
            for nxt in nbrs(pt):
                if frozenset((pt, nxt)) in visited_edges:
                    continue
                path = walk(pt, nxt)
                for a, b in zip(path, path[1:]):
                    visited_edges.add(frozenset((a, b)))
                paths.append(path)

    segments: list[tuple[float, float, float, float]] = []
    for path in paths:
        if len(path) < 2:
            continue
        simple = _simplify(path, tol=2.0)
        for (x1, y1), (x2, y2) in zip(simple, simple[1:]):
            if (x1, y1) == (x2, y2):
                continue
            segments.append((x1 * cell_px, y1 * cell_px, x2 * cell_px, y2 * cell_px))
    return snap_and_merge(segments, tol=cell_px * 1.5)


def snap_and_merge(segments: list[tuple[float, float, float, float]],
                   tol: float = 6.0) -> list[tuple[float, float, float, float]]:
    """端点吸附 + 相邻共线合并。

    这一步是**为了 depthmapX 能把线连起来**：它的轴向图靠"线是否相交/接触"
    建连接，端点差几个像素就会变成孤立线（实测会让 2/3 的线断开、指标变 -1）。
    所以先把容差内的端点吸附到同一坐标，再把共线的相邻段并成一条。
    """
    # 1) 端点聚类吸附
    points: list[tuple[float, float]] = []
    for (x1, y1, x2, y2) in segments:
        points += [(x1, y1), (x2, y2)]
    clusters: list[list[tuple[float, float]]] = []
    for p in points:
        for c in clusters:
            if math.hypot(p[0] - c[0][0], p[1] - c[0][1]) <= tol:
                c.append(p)
                break
        else:
            clusters.append([p])
    def snap(p):
        for c in clusters:
            if math.hypot(p[0] - c[0][0], p[1] - c[0][1]) <= tol:
                cx = sum(q[0] for q in c) / len(c)
                cy = sum(q[1] for q in c) / len(c)
                return (round(cx, 2), round(cy, 2))
        return p

    snapped = []
    for (x1, y1, x2, y2) in segments:
        a, b = snap((x1, y1)), snap((x2, y2))
        if a != b:
            snapped.append((a[0], a[1], b[0], b[1]))

    # 2) 共线且共享端点的相邻段合并（保留外层端点，保证仍与邻居相接）
    changed = True
    while changed:
        changed = False
        for i in range(len(snapped)):
            if changed:
                break
            for j in range(i + 1, len(snapped)):
                x1, y1, x2, y2 = snapped[i]
                u1, v1, u2, v2 = snapped[j]
                ends_i = [(x1, y1), (x2, y2)]
                ends_j = [(u1, v1), (u2, v2)]
                shared = next((p for p in ends_i if p in ends_j), None)
                if shared is None:
                    continue
                other_i = ends_i[1] if ends_i[0] == shared else ends_i[0]
                other_j = ends_j[1] if ends_j[0] == shared else ends_j[0]
                v1x, v1y = other_i[0] - shared[0], other_i[1] - shared[1]
                v2x, v2y = other_j[0] - shared[0], other_j[1] - shared[1]
                n1 = math.hypot(v1x, v1y) or 1e-9
                n2 = math.hypot(v2x, v2y) or 1e-9
                cos = (v1x * v2x + v1y * v2y) / (n1 * n2)
                if cos < -0.985:      # 夹角 > ~170°：近似共线
                    snapped[i] = (other_i[0], other_i[1], other_j[0], other_j[1])
                    snapped.pop(j)
                    changed = True
                    break
    return snapped


def draw_preview(result: dict, path: Path, map_name: str) -> None:
    radar = result["radar"].copy()
    canvas = Image.new("RGBA", radar.size, (255, 255, 255, 255))
    canvas.alpha_composite(radar)
    dr = ImageDraw.Draw(canvas)
    for (x1, y1, x2, y2) in result["segments"]:
        dr.line([x1, y1, x2, y2], fill=(255, 40, 40, 255), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(path)
    print(f"  wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("map")
    ap.add_argument("--cell", type=int, default=4)
    ap.add_argument("--max-lines", type=int, default=400)
    ap.add_argument("--dxf", type=Path, default=None)
    args = ap.parse_args()
    dxf = args.dxf or Path("out") / f"{args.map}_axial.dxf"
    result = build(args.map, args.cell, args.max_lines)
    write_dxf(result["segments"], dxf)
    print(f"  wrote {dxf}（{len(result['segments'])} 条轴线）")
    draw_preview(result, Path("out") / f"{args.map}_axial_preview.png", args.map)


if __name__ == "__main__":
    main()
