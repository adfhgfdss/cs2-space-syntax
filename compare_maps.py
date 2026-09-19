r"""多图构形对比：把两张（或更多）地图的空间句法指标放在一起看。

这一步是"实验二"的起点：同一套方法跑在不同地图上，比较它们的构形差异。
它不需要 demo，只要有雷达图和导航网格就能跑，所以 de_dust2 也能进对比。

用法：
    .venv\\Scripts\\python.exe compare_maps.py de_dust2 de_mirage

    收敛为单图研究后，被降级为附录的图（例如 de_dust2）其句法表放在
    `out\\appendix_dust2\\`，本脚本会自动去找；用 `--out-dir` 指定结果写去哪。

输出：
    out\\compare_syntax_summary.csv   各图指标汇总
    out\\compare_syntax_maps.png      指标分布 + 两张图的整合度/选择度对照
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRICS = ("connectivity", "isovist_area_px", "integration_hh_t",
           "integration_hh_m", "choice_t")


def summarise(map_name: str, root: Path) -> tuple[pl.DataFrame, dict]:
    path = find_grid(map_name, root)
    df = pl.read_csv(path)
    row = {"map": map_name, "cells": df.height}
    for metric in METRICS:
        values = df[metric].to_numpy().astype(float)
        row[f"{metric}_median"] = round(float(np.median(values)), 4)
        row[f"{metric}_mean"] = round(float(values.mean()), 4)
        row[f"{metric}_p95"] = round(float(np.percentile(values, 95)), 4)
    # 结构特征：拓扑与米制整合度的一致性（高 = 尺度均匀，低 = 有明显的环形/长距离结构）
    row["t_vs_m_integration_corr"] = round(float(np.corrcoef(
        df["integration_hh_t"].to_numpy().astype(float),
        df["integration_hh_m"].to_numpy().astype(float))[0, 1]), 4)
    # 选择度的集中度：前 5% 格子占了多少"经过量"
    choice = np.sort(df["choice_t"].to_numpy().astype(float))[::-1]
    top = max(int(len(choice) * 0.05), 1)
    row["choice_top5pct_share"] = round(float(choice[:top].sum() / choice.sum()), 4)
    return df, row


def find_grid(map_name: str, root: Path) -> Path:
    """主线图在 out\\ 下，附录图在 out\\appendix_*\\ 下，两处都找。"""
    candidates = [root / "out" / f"{map_name}_syntax_grid.csv"]
    candidates += sorted(root.glob(f"out/appendix_*/{map_name}_syntax_grid.csv"))
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(
        f"找不到 {map_name}_syntax_grid.csv（out\\ 与 out\\appendix_*\\ 都找过了）。"
        f"先跑 space_syntax.py {map_name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("maps", nargs="+", help="要对比的地图名")
    ap.add_argument("--out-dir", type=Path, default=Path("out"),
                    help="结果写到哪个目录（附录对比建议写 out/appendix_dust2）")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, rows = {}, []
    for name in args.maps:
        df, row = summarise(name, root)
        frames[name] = df
        rows.append(row)

    summary = pl.DataFrame(rows)
    out_csv = out_dir / "compare_syntax_summary.csv"
    summary.write_csv(out_csv)
    print(f"wrote {out_csv}")
    with pl.Config(tbl_cols=12, tbl_width_chars=200, fmt_str_lengths=30):
        print(summary.select(
            "map", "cells", "connectivity_median", "integration_hh_t_median",
            "integration_hh_m_median", "choice_t_median",
            "t_vs_m_integration_corr", "choice_top5pct_share"))

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    n = len(args.maps)
    fig, axes = plt.subplots(2, n + 2, figsize=(5 * (n + 2), 11), dpi=100)

    for i, name in enumerate(args.maps):
        df = frames[name]
        radar = np.array(Image.open(root / "data" / f"{name}.png").convert("RGBA"))
        cols = int(df["col"].max()) + 1
        rows_n = int(df["row"].max()) + 1
        cell = int(df["px_left"].unique().sort().to_numpy()[1]
                   - df["px_left"].unique().sort().to_numpy()[0])
        for j, (metric, title, cmap) in enumerate((
            ("integration_hh_t", "整合度 Integration[HH]", "turbo"),
            ("choice_t", "选择度 Choice", "magma"),
        )):
            ax = axes[j][i]
            layer = np.full((rows_n, cols), np.nan)
            layer[df["row"].to_numpy(), df["col"].to_numpy()] = \
                df[metric].to_numpy().astype(float)
            ax.imshow(radar, extent=[0, cols * cell, rows_n * cell, 0])
            lo, hi = np.nanpercentile(layer, [2, 98])
            ax.imshow(np.ma.masked_invalid(layer),
                      extent=[0, cols * cell, rows_n * cell, 0],
                      cmap=cmap, alpha=0.78, vmin=lo, vmax=hi)
            ax.set_title(f"{name} — {title}")
            ax.set_axis_off()

    # 分布对比：整合度与选择度（选择度取对数，跨度太大）
    for j, (metric, title, log) in enumerate((
        ("integration_hh_t", "整合度分布", False),
        ("choice_t", "选择度分布（对数）", True),
    )):
        ax = axes[j][n]
        for name in args.maps:
            values = frames[name][metric].to_numpy().astype(float)
            if log:
                values = np.log10(values - values.min() + 1.0)
            ax.hist(values, bins=40, alpha=0.55, label=name, density=True)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.2)

    # 汇总条形图
    ax = axes[0][n + 1]
    width = 0.35
    xs = np.arange(len(args.maps))
    ax.bar(xs - width / 2, [r["integration_hh_m_median"] for r in rows], width,
           label="米制整合度中位")
    ax.bar(xs + width / 2, [r["t_vs_m_integration_corr"] for r in rows], width,
           label="拓扑/米制整合度一致性")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["map"] for r in rows])
    ax.set_title("指标可比性：请用同一口径跨图比较")
    ax.grid(alpha=0.2, axis="y")
    ax.legend()

    ax = axes[1][n + 1]
    ax.bar(xs - width / 2, [r["choice_top5pct_share"] for r in rows], width,
           label="前 5% 格子的选择度占比")
    ax.bar(xs + width / 2, [r["connectivity_median"] for r in rows], width,
           label="连通度中位")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["map"] for r in rows])
    ax.set_title("结构差异：选择度越集中，越像「少数通道承担全部穿行」")
    ax.grid(alpha=0.2, axis="y")
    ax.legend()

    fig.tight_layout()
    out_png = out_dir / "compare_syntax_maps.png"
    fig.savefig(out_png, facecolor="black")
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
