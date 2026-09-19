r"""CS2 导航网格（.nav）解析器：拿到"哪里能走"的权威定义。

为什么需要它
------------
雷达图是俯视渲染图，屋顶和地面叠在一起，肉眼看不出哪块能走。而 `.nav` 是
Valve 自己为每张图烘培的导航网格：一堆凸多边形，每个顶点的世界坐标，
正好就是"可通行地面"。有了它：

  * 可通行区域不再需要人工描图；
  * 没有 demo 的地图（比如现在的 de_dust2）也能做空间句法的几何层；
  * 多边形是世界坐标，可以直接拿来交叉验证坐标变换对不对。

格式（逆向得到，de_mirage/de_dust2 的 CS2 版本 version=36 实测通过）
---------------------------------------------------------------
    +0   u32   magic        0xFEEDFACE
    +4   u32   version      36
    +8   u32   sub_version
    +12  u32   flags        bit0 = is_analyzed
    ...  其余头部字段（占位到 +145）
    +145 u32   corner_count
    +149 f32×3×corner_count   角点坐标 (x, y, z)
         u32   polygon_count
         polygon_count × {
             u8    corner_count_of_this_polygon
             u32 × n          角点索引
             u32              预留（v35+ 有）
         }
         ...  其余是 areas / ladders / hiding spots，本模块暂不解析

awpy 2.0.2 自带的解析器只认 version 30..35，读不了当前 CS2 的 36，所以这里
自己实现。本模块只取几何（角点 + 多边形），因为可通行掩码只需要几何。

用法：
    .venv\\Scripts\\python.exe navmesh.py de_dust2          # 统计 + 出对照图
    .venv\\Scripts\\python.exe navmesh.py de_mirage
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MAGIC = 0xFEEDFACE


@dataclass
class NavMesh:
    """一张地图的导航网格几何。"""

    version: int
    sub_version: int
    corners: np.ndarray          # (N, 3) float32，世界坐标
    polygons: list[np.ndarray]   # 每个元素是该多边形的角点索引

    @property
    def polygon_sizes(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for indices in self.polygons:
            counts[len(indices)] = counts.get(len(indices), 0) + 1
        return dict(sorted(counts.items()))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.corners.min(axis=0), self.corners.max(axis=0)

    def to_npz(self, path: Path) -> None:
        """存成 npz，后面各步骤直接读，不必重复解析。"""
        flat = np.concatenate(self.polygons) if self.polygons else np.zeros(0, np.int32)
        sizes = np.array([len(p) for p in self.polygons], dtype=np.int32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, version=self.version, sub_version=self.sub_version,
                            corners=self.corners, flat=flat, sizes=sizes)

    @classmethod
    def from_npz(cls, path: Path) -> "NavMesh":
        data = np.load(path)
        flat, sizes = data["flat"], data["sizes"]
        polygons, pos = [], 0
        for size in sizes:
            polygons.append(flat[pos:pos + size])
            pos += int(size)
        return cls(int(data["version"]), int(data["sub_version"]),
                   data["corners"], polygons)


def find_corner_block(blob: bytes, search_limit: int = 200_000) -> int:
    """定位角点数组的起点（返回 corner_count 字段的偏移）。

    CS2 各版本头部长度不完全一样，所以不硬编码偏移，而是按"后面确实跟着
    一段干净的坐标浮点数"来判定。
    """
    n = len(blob)
    for p in range(12, min(n - 8, search_limit)):
        (count,) = struct.unpack_from("<I", blob, p)
        if not (100 <= count <= 200_000):
            continue
        end = p + 4 + count * 12
        if end + 4 > n:
            continue
        arr = np.frombuffer(blob, dtype="<f4", count=count * 3, offset=p + 4)
        if not np.isfinite(arr).all():
            continue
        if np.abs(arr).max() > 20_000:
            continue
        z = arr[2::3]
        if z.max() - z.min() > 4000:
            continue
        (poly_count,) = struct.unpack_from("<I", blob, end)
        if not (10 <= poly_count <= 200_000):
            continue
        return p
    raise ValueError("找不到导航网格的角点数组，可能是没见过的 .nav 版本")


def read_nav(path: Path) -> NavMesh:
    blob = Path(path).read_bytes()
    magic, version, sub_version = struct.unpack_from("<III", blob, 0)
    if magic != MAGIC:
        raise ValueError(f"不是 .nav 文件：magic=0x{magic:08X}")

    p = find_corner_block(blob)
    (corner_count,) = struct.unpack_from("<I", blob, p)
    corners = np.frombuffer(blob, dtype="<f4", count=corner_count * 3,
                            offset=p + 4).reshape(-1, 3).astype(np.float64)

    q = p + 4 + corner_count * 12
    (polygon_count,) = struct.unpack_from("<I", blob, q)
    q += 4

    polygons: list[np.ndarray] = []
    for _ in range(polygon_count):
        size = blob[q]
        q += 1
        if not 3 <= size <= 64:
            raise ValueError(f"多边形角点数异常：{size}（偏移 {q - 1}）")
        indices = np.frombuffer(blob, dtype="<u4", count=size, offset=q).astype(np.int32)
        if indices.max() >= corner_count:
            raise ValueError(f"角点索引越界：{indices.max()} >= {corner_count}")
        polygons.append(indices)
        q += size * 4 + 4

    return NavMesh(version=version, sub_version=sub_version,
                   corners=corners, polygons=polygons)


def polygon_centroids(nav: NavMesh) -> np.ndarray:
    return np.array([nav.corners[idx].mean(axis=0) for idx in nav.polygons])


def render_overlay(nav: NavMesh, radar_path: Path, overview: dict[str, float],
                   out_path: Path, transform=None) -> None:
    """把导航网格叠到雷达图上，用来看两者对不对得齐。"""
    from heatmap import MapTransform, detect_transform

    radar = Image.open(radar_path).convert("RGBA")
    tf = transform or detect_transform(overview, np.array(radar), nav)
    bg = Image.new("RGBA", radar.size, (255, 255, 255, 255))
    bg.alpha_composite(radar)
    line = Image.new("RGBA", radar.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(line)
    for indices in nav.polygons:
        pts3 = nav.corners[indices]
        px, py = tf.world_to_radar(pts3[:, 0], pts3[:, 1])
        pts = list(zip(px.tolist(), py.tolist()))
        dr.polygon(pts, fill=(255, 40, 40, 90), outline=(255, 255, 0, 200))
    bg.alpha_composite(line)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bg.convert("RGB").save(out_path)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("map", help="地图名，例如 de_dust2")
    ap.add_argument("--nav", type=Path, default=None, help=".nav 路径（默认 data/game_nav/<map>.nav）")
    ap.add_argument("--npz", action="store_true", help="同时另存一份 npz 缓存")
    ap.add_argument("--no-figure", action="store_true", help="不出对照图")
    args = ap.parse_args()

    nav_path = args.nav or Path("data/game_nav") / f"{args.map}.nav"
    nav = read_nav(nav_path)
    lo, hi = nav.bounds()
    print(f"{nav_path}: version={nav.version}.{nav.sub_version}")
    print(f"  角点 {len(nav.corners):,} 个，多边形 {len(nav.polygons):,} 个")
    print(f"  多边形角点数分布 {nav.polygon_sizes}")
    print(f"  世界坐标范围 x {lo[0]:.0f}..{hi[0]:.0f}  y {lo[1]:.0f}..{hi[1]:.0f} "
          f" z {lo[2]:.0f}..{hi[2]:.0f}")

    if args.npz:
        npz = Path("data/game_nav") / f"{args.map}_navmesh.npz"
        nav.to_npz(npz)
        print(f"  缓存 -> {npz}")

    if not args.no_figure:
        from heatmap import read_overview_all

        overview = read_overview_all(Path(f"data/{args.map}.txt"))
        render_overlay(nav, Path(f"data/{args.map}.png"), overview,
                       Path("out") / f"{args.map}_navmesh_overlay.png")


if __name__ == "__main__":
    main()
