"""Overlay CS2 event positions onto the map radar image.

The CS2 overview file gives the transform between world units and radar
pixels:

    radar_x = (world_x - pos_x) / scale
    radar_y = (pos_y - world_y) / scale      <- note the flipped Y axis

If the transform is right, kill markers land on walkable ground. Markers
inside walls or off the map mean the transform or the map file is wrong.
"""

from __future__ import annotations

import itertools
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
from PIL import Image
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_overview(path: Path) -> dict[str, float]:
    """Parse a CS2 overview .txt (Valve KeyValues) for the transform fields."""
    text = path.read_text(encoding="utf-8", errors="replace")
    values: dict[str, float] = {}
    for key in ("pos_x", "pos_y", "scale", "rotate"):
        match = re.search(rf'"{key}"\s+"(-?[\d.]+)"', text)
        if match:
            values[key] = float(match.group(1))
    if not {"pos_x", "pos_y", "scale"} <= values.keys():
        raise SystemExit(f"overview file missing transform fields: {path}")
    return values


def read_overview_all(path: Path) -> dict[str, float]:
    """Parse every numeric key out of a CS2 overview file.

    Besides the transform, this carries normalised positions for both team
    spawns and both bomb sites, which are useful for orienting a figure and
    for sanity-checking the transform.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        key: float(value)
        for key, value in re.findall(r'"(\w+)"\s+"(-?[\d.]+)"', text)
    }


def world_to_radar(
    x: np.ndarray, y: np.ndarray, overview: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    px = (x - overview["pos_x"]) / overview["scale"]
    py = (overview["pos_y"] - y) / overview["scale"]
    return px, py


# --------------------------------------------------------------------------
# 带旋转的坐标变换
#
# overview 文件里的 "rotate" 表示"这张雷达图在图像编辑时被转了 90 度"，也就是
# 图像坐标轴和世界坐标轴对调了。de_mirage 是 rotate=0，de_dust2 是 rotate=1。
# 与其照抄某份社区约定，这里把 8 种"旋转 + 翻转"组合都列出来，用数据自己选：
# 正确的那一种会让导航网格几乎完全落在雷达图的有内容区域内，错的那种只有
# 随机命中率（dust2 约 48%）。
# --------------------------------------------------------------------------

# 归一化坐标 (u, v) 的 8 种变换写法，u 对应图像横轴、v 对应纵轴
_AXIS_FORMS = {
    "uv":   lambda u, v: (u, v),
    "1-uv": lambda u, v: (1.0 - u, v),
    "u1-v": lambda u, v: (u, 1.0 - v),
    "1-u1-v": lambda u, v: (1.0 - u, 1.0 - v),
    "vu":   lambda u, v: (v, u),
    "1-vu": lambda u, v: (1.0 - v, u),
    "v1-u": lambda u, v: (v, 1.0 - u),
    "1-v1-u": lambda u, v: (1.0 - v, 1.0 - u),
}


@dataclass
class MapTransform:
    """世界坐标 <-> 雷达像素的变换，支持 8 种轴向组合。"""

    pos_x: float
    pos_y: float
    scale: float
    width: int
    height: int
    form: str = "uv"
    overview: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_overview(
        cls, overview: dict[str, float], size: tuple[int, int] = (1024, 1024),
        form: str = "uv",
    ) -> "MapTransform":
        return cls(pos_x=overview["pos_x"], pos_y=overview["pos_y"],
                   scale=overview["scale"], width=size[0], height=size[1],
                   form=form, overview=overview)

    def normalized(self, x: np.ndarray, y: np.ndarray):
        """世界坐标 -> [0,1] 归一化图像坐标（未做轴向变换）。"""
        u = (np.asarray(x, dtype=np.float64) - self.pos_x) / (self.scale * self.width)
        v = (self.pos_y - np.asarray(y, dtype=np.float64)) / (self.scale * self.height)
        return u, v

    def world_to_radar(self, x: np.ndarray, y: np.ndarray):
        u, v = self.normalized(x, y)
        uu, vv = _AXIS_FORMS[self.form](u, v)
        return uu * self.width, vv * self.height

    def radar_to_world(self, px: np.ndarray, py: np.ndarray):
        """像素 -> 世界坐标（用于把网格边界写回世界单位）。"""
        u = np.asarray(px, dtype=np.float64) / self.width
        v = np.asarray(py, dtype=np.float64) / self.height
        # 反解：先还原 (u, v)，再按定义反推世界坐标
        if self.form in ("uv", "1-uv", "u1-v", "1-u1-v"):
            a, b = u, v
            if self.form in ("1-uv", "1-u1-v"):
                a = 1.0 - a
            if self.form in ("u1-v", "1-u1-v"):
                b = 1.0 - b
        else:
            a, b = v, u
            if self.form in ("1-vu", "1-v1-u"):
                a = 1.0 - a
            if self.form in ("v1-u", "1-v1-u"):
                b = 1.0 - b
        x = a * self.scale * self.width + self.pos_x
        y = self.pos_y - b * self.scale * self.height
        return x, y


def detect_transform(
    overview: dict[str, float], radar_rgba, nav, verbose: bool = True
) -> MapTransform:
    """用导航网格和雷达图内容区自动挑出正确的轴向组合。

    判据：把导航网格按每种组合栅格化到图像上，正确的那种会几乎全部落在
    雷达图有内容的像素里（地图footprint之内）。
    """
    from PIL import Image, ImageDraw

    height, width = radar_rgba.shape[:2]
    content = radar_rgba[:, :, 3] > 128
    scored = []
    for form in _AXIS_FORMS:
        tf = MapTransform.from_overview(overview, (width, height), form=form)
        canvas = Image.new("L", (width, height), 0)
        dr = ImageDraw.Draw(canvas)
        for indices in nav.polygons:
            pts3 = nav.corners[indices]
            px, py = tf.world_to_radar(pts3[:, 0], pts3[:, 1])
            dr.polygon(list(zip(px.tolist(), py.tolist())), fill=255)
        mask = np.array(canvas) > 0
        if mask.sum() == 0:
            continue
        hit = float(content[mask].mean())
        scored.append((hit, form))
    if not scored:
        raise SystemExit("无法确定坐标变换：导航网格栅格化后为空")
    scored.sort(reverse=True)
    best_hit, best_form = scored[0]
    if verbose:
        print("坐标变换候选（导航网格落在雷达图内容区的比例）:")
        for hit, form in scored:
            mark = "  <-- 选用" if form == best_form else ""
            print(f"  {form:<8} {hit*100:6.1f}%{mark}")
    if best_hit < 0.85:
        print(f"警告：最佳组合命中率只有 {best_hit*100:.1f}%，坐标变换可能仍有问题")
    return MapTransform.from_overview(overview, (width, height), form=best_form)


def main() -> None:
    demo_map = sys.argv[1] if len(sys.argv) > 1 else "de_mirage"
    kills_path = Path("out/player_death.parquet")
    radar_path = Path(f"data/{demo_map}.png")
    overview_path = Path(f"data/{demo_map}.txt")

    for path in (kills_path, radar_path, overview_path):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    overview = read_overview(overview_path)
    print(f"overview: pos_x={overview['pos_x']} pos_y={overview['pos_y']} "
          f"scale={overview['scale']}")

    radar = Image.open(radar_path).convert("RGBA")
    width, height = radar.size
    print(f"radar: {width}x{height}")

    kills = pl.read_parquet(kills_path)
    x = kills["attacker_X"].cast(pl.Float64).to_numpy()
    y = kills["attacker_Y"].cast(pl.Float64).to_numpy()

    px, py = world_to_radar(x, y, overview)

    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    print(f"kills: {len(px)} total, {inside.sum()} inside radar bounds")
    if not inside.all():
        print(f"  out-of-bounds x range: {px.min():.1f}..{px.max():.1f}")
        print(f"  out-of-bounds y range: {py.min():.1f}..{py.max():.1f}")

    # Density grid, smoothed into a heat surface.
    grid, _, _ = np.histogram2d(
        py[inside], px[inside],
        bins=[height, width],
        range=[[0, height], [0, width]],
    )
    density = gaussian_filter(grid, sigma=18)
    if density.max() > 0:
        density = density / density.max()

    fig, ax = plt.subplots(figsize=(10, 10), dpi=110)
    ax.imshow(radar, extent=[0, width, height, 0])
    ax.imshow(
        np.ma.masked_where(density < 0.05, density),
        extent=[0, width, height, 0],
        cmap="inferno",
        alpha=0.55,
    )
    ax.scatter(px[inside], py[inside], s=9, c="cyan", edgecolors="black",
               linewidths=0.3, alpha=0.75, label="kill position")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title(f"{demo_map} - kill positions ({inside.sum()} events)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_axis_off()

    out_path = Path("out") / f"{demo_map}_kills_heatmap.png"
    fig.tight_layout()
    fig.savefig(out_path, facecolor="black")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
