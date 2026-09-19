r"""把两个实验做成直观的曲线图，并把曲线上的点导出成表（可核对）。

产出（都在 out\ 下）：
    exp1_regression.png        实验一：整合度-交火散点+回归线，以及按整合度分位数的曲线
    exp1_cumulative.png        实验一：累计曲线（整合度最高的前 X% 轴线承担多少交火）
    exp2_regression.png        实验二：连通度-整合度散点+回归线（R²）
    exp1_curve_data.csv        分位数曲线上的确切数值
    exp1_cumulative_data.csv   累计曲线上的确切数值
    exp2_points_axial.csv      实验二散点（轴线口径）的原始数据
    exp2_points_vga_sample.csv 实验二散点（VGA 口径）的抽样数据（画图用）

用法：
    .venv\Scripts\python.exe exp_curves.py
"""

from __future__ import annotations

from pathlib import Path

import json

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
MAPS = ("de_dust2", "de_mirage")
LABEL = {"de_dust2": "Dust2", "de_mirage": "Mirage"}


def fit_text(x, y, fit) -> str:
    a, b = fit.intercept, fit.slope
    rho = stats.spearmanr(x, y)
    r = stats.pearsonr(x, y)
    return (f"n={len(x)}   ρ={rho.statistic:+.3f} (p={rho.pvalue:.2e})\n"
            f"r={r.statistic:+.3f}  R²={r.statistic ** 2:.3f}  p={r.pvalue:.2e}\n"
            f"拟合：y={a:.3f}{b:+.4f}x")


def decile_curve(x: np.ndarray, y: np.ndarray, bins: int = 10):
    order = np.argsort(x)
    groups = np.array_split(order, bins)
    xs = np.array([x[g].mean() for g in groups])
    ys = np.array([y[g].mean() for g in groups])
    se = np.array([y[g].std(ddof=1) / np.sqrt(len(g)) for g in groups])
    return xs, ys, se, groups


def valid_mask(df: pl.DataFrame) -> np.ndarray:
    """depthmapX 对孤立轴线会给 integration=-1 / choice=-1，这些必须剔除。"""
    integ = df["integration_hh"].to_numpy().astype(float)
    conn = df["connectivity"].to_numpy().astype(float)
    choice = df["choice"].to_numpy().astype(float)
    return (integ > 0) & (conn > 0) & (choice >= 0)


def figure_exp1(data: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 11), dpi=110)
    rows = []
    for row, map_name in enumerate(MAPS):
        df = data[map_name]
        keep = valid_mask(df)
        dropped = int((~keep).sum())
        df = df.filter(pl.Series(keep))
        x = df["integration_hh"].to_numpy().astype(float)
        y = df["crossfire_per_m"].to_numpy().astype(float)
        fit = stats.linregress(x, y)
        ax = axes[row][0]
        ax.scatter(x, y, s=34, c="#1f77b4", alpha=0.85, edgecolors="white", linewidths=0.5)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, fit.intercept + fit.slope * xs, color="#d62728", linewidth=2.4,
                label="最小二乘拟合")
        ax.set_xlabel("轴线整合度 Integration[HH]（depthmapX）")
        ax.set_ylabel("交火密度（次/米）")
        ax.set_title(f"{LABEL[map_name]}：{df.height} 条轴线，"
                     f"{int(df['crossfire'].sum())} 次交火"
                     + (f"（剔除 {dropped} 条孤立轴线）" if dropped else ""))
        ax.text(0.03, 0.97, fit_text(x, y, fit), transform=ax.transAxes,
                va="top", ha="left", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=9)

        ax2 = axes[row][1]
        xs2, ys2, se2, groups = decile_curve(x, y)
        ax2.errorbar(xs2, ys2, yerr=se2, marker="o", markersize=7, capsize=4,
                     color="#2ca02c", linewidth=2.2)
        for xi, yi in zip(xs2, ys2):
            ax2.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                         xytext=(0, 9), ha="center", fontsize=8)
        ax2.set_xlabel("整合度（按十分位数分箱后的组均值）")
        ax2.set_ylabel("该组平均交火密度（次/米）")
        ax2.set_title(f"{LABEL[map_name]}：按整合度分 10 组的交火密度曲线"
                      f"（误差棒=标准误）")
        ax2.grid(alpha=0.25)
        for i, (xi, yi, ei, g) in enumerate(zip(xs2, ys2, se2, groups)):
            rows.append({"map": map_name, "decile": i + 1, "lines": len(g),
                         "integration_mean": round(float(xi), 4),
                         "crossfire_per_m_mean": round(float(yi), 4),
                         "crossfire_per_m_se": round(float(ei), 4),
                         "crossfire_sum": int(df["crossfire"].to_numpy()[g].sum())})
    fig.tight_layout()
    path = OUT / "exp1_regression.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    pl.DataFrame(rows).write_csv(OUT / "exp1_curve_data.csv")
    print(f"wrote {path} 与 exp1_curve_data.csv")


def figure_exp1_cumulative(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 7.5), dpi=110)
    rows = []
    for map_name, color in zip(MAPS, ("#d62728", "#1f77b4")):
        df = data[map_name]
        df = df.filter(pl.Series(valid_mask(df)))
        x = df["integration_hh"].to_numpy().astype(float)
        y = df["crossfire"].to_numpy().astype(float)
        order = np.argsort(-x)
        cum = np.cumsum(y[order]) / y.sum()
        pct = np.arange(1, len(cum) + 1) / len(cum) * 100.0
        ax.plot(pct, cum * 100, linewidth=2.4, color=color,
                label=f"{LABEL[map_name]}（{len(cum)} 条轴线）")
        for p in (10, 20, 30, 50):
            idx = max(int(round(len(cum) * p / 100)) - 1, 0)
            rows.append({"map": map_name, "top_pct_of_lines": p,
                         "crossfire_share_pct": round(float(cum[idx] * 100), 2)})
    ax.plot([0, 100], [0, 100], linestyle="--", color="grey", linewidth=1.5,
            label="随机分布（对角线）")
    ax.set_xlabel("整合度最高的前 X% 轴线")
    ax.set_ylabel("这些轴线承担的累计交火占比（%）")
    ax.set_title("累计曲线：交火是否集中在高整合度轴线上")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = OUT / "exp1_cumulative.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    pl.DataFrame(rows).write_csv(OUT / "exp1_cumulative_data.csv")
    print(f"wrote {path} 与 exp1_cumulative_data.csv")


def figure_exp2(axial: dict, vga: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=110)
    for col, map_name in enumerate(MAPS):
        df = axial[map_name]
        x = df["connectivity"].to_numpy().astype(float)
        y = df["integration_hh"].to_numpy().astype(float)
        ax = axes[0][col]
        ax.scatter(x, y, s=34, c="#1f77b4", alpha=0.85, edgecolors="white", linewidths=0.5)
        fit = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, fit.intercept + fit.slope * xs, color="#d62728", linewidth=2.4)
        ax.set_xlabel("连通度 Connectivity（局部）")
        ax.set_ylabel("全局整合度 Integration[HH]")
        ax.set_title(f"{LABEL[map_name]} 轴线口径：R²={fit.rvalue ** 2:.4f}")
        ax.text(0.03, 0.97,
                f"n={len(x)}\nr={fit.rvalue:+.4f}\nR²={fit.rvalue ** 2:.4f}\n"
                f"p={fit.pvalue:.2e}\n斜率={fit.slope:.4f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
        ax.grid(alpha=0.25)
        df.write_csv(OUT / f"exp2_points_axial_{map_name}.csv")

        grid = vga[map_name]
        gx = grid["connectivity"].to_numpy().astype(float)
        gy = grid["integration_hh_t"].to_numpy().astype(float)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(gx), size=min(1200, len(gx)), replace=False)
        ax2 = axes[1][col]
        ax2.scatter(gx[idx], gy[idx], s=14, c="#2ca02c", alpha=0.5, edgecolors="none")
        fit2 = stats.linregress(gx, gy)
        xs2 = np.linspace(np.percentile(gx, 1), np.percentile(gx, 99), 50)
        ax2.plot(xs2, fit2.intercept + fit2.slope * xs2, color="#d62728", linewidth=2.4)
        ax2.set_xlabel("可见格子数（VGA 连通度）")
        ax2.set_ylabel("Integration[HH]（拓扑）")
        ax2.set_title(f"{LABEL[map_name]} VGA 口径：R²={fit2.rvalue ** 2:.4f}")
        ax2.text(0.03, 0.97,
                 f"n={len(gx)}\nr={fit2.rvalue:+.4f}\nR²={fit2.rvalue ** 2:.4f}\n"
                 f"p={fit2.pvalue:.2e}\n斜率={fit2.slope:.6f}",
                 transform=ax2.transAxes, va="top", ha="left", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
        ax2.grid(alpha=0.25)
        pl.DataFrame({"connectivity": gx[idx], "integration_hh_t": gy[idx]}).write_csv(
            OUT / f"exp2_points_vga_sample_{map_name}.csv")
    fig.tight_layout()
    path = OUT / "exp2_regression.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    axial = {m: pl.read_csv(OUT / f"{m}_axial_lines.csv") for m in MAPS}
    vga = {m: pl.read_csv(OUT / f"{m}_syntax_grid.csv") for m in MAPS}
    figure_exp1(axial)
    figure_exp1_cumulative(axial)
    figure_exp2(axial, vga)

    # 把回归的确切参数单独存一份，报告里直接引用
    fits = {}
    for map_name in MAPS:
        df = axial[map_name]
        keep_lines = valid_mask(df)
        df = df.filter(pl.Series(keep_lines))
        integ = df["integration_hh"].to_numpy().astype(float)
        dens = df["crossfire_per_m"].to_numpy().astype(float)
        counts = df["crossfire"].to_numpy().astype(float)
        choice = df["choice"].to_numpy().astype(float)
        conn = df["connectivity"].to_numpy().astype(float)
        entry = {"n_lines": int(df.height),
                 "crossfire_total": int(counts.sum())}
        for tag, x, y in (("integration_vs_density", integ, dens),
                          ("choice_vs_density", choice, dens),
                          ("integration_vs_count", integ, counts),
                          ("connectivity_vs_integration", conn, integ)):
            fit = stats.linregress(x, y)
            rho = stats.spearmanr(x, y)
            entry[tag] = {
                "n": int(len(x)),
                "spearman_rho": round(float(rho.statistic), 4),
                "spearman_p": float(rho.pvalue),
                "pearson_r": round(float(fit.rvalue), 4),
                "r2": round(float(fit.rvalue ** 2), 4),
                "pearson_p": float(fit.pvalue),
                "slope": float(fit.slope),
                "intercept": float(fit.intercept),
                "stderr": float(fit.stderr),
            }
        fits[map_name] = entry
    (OUT / "exp_fit_exact.json").write_text(
        json.dumps(fits, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote exp_fit_exact.json")


if __name__ == "__main__":
    main()
