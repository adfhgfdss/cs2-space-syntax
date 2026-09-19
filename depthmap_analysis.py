r"""两个实验的主脚本：构形敏感性（实验一）与可理解度（实验二）。

实验一：轴线图（depthmapX 计算）→ 整合度 / 选择度 → 与真实比赛的交火热点对照。
实验二：可理解度 R² = 连通度与全局整合度之间的决定系数；Dust2 vs Mirage 对比。

依赖：`tools/depthmapXcli.exe`（官方 release 的命令行版）。

用法：
    .venv\Scripts\python.exe depthmap_analysis.py            # 两张图都做
    .venv\Scripts\python.exe depthmap_analysis.py de_dust2   # 只做一张

产出（都在 out\ 下）：
    <地图>_axial.dxf / _axial_preview.png    轴线图（几何）
    <地图>_axial_lines.csv                   每条轴线：depthmapX 指标 + 交火计数 + 密度
    <地图>_axial_map.png                     整合度着色 + 交火热点叠加
    <地图>_axial_scatter.png                 实验一/实验二的散点图
    exp1_configuration.json                  实验一结果（相关系数、热点线）
    exp2_intelligibility.json                实验二结果（两种口径的 R²）
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
from PIL import Image
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import axial_map  # noqa: E402
from heatmap import detect_transform, read_overview_all  # noqa: E402
from navmesh import NavMesh  # noqa: E402

ROOT = Path(__file__).resolve().parent
CLI = ROOT / "tools" / "depthmapXcli.exe"


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(CLI), *args], capture_output=True, text=True,
                          errors="replace")


def depthmapx_axial(map_name: str, dxf: Path) -> Path:
    """IMPORT → MAPCONVERT(axial) → AXIAL → EXPORT csv，全程用 depthmapX 本体。"""
    work = ROOT / "out" / "depthmap" / map_name
    work.mkdir(parents=True, exist_ok=True)
    graph = work / "drawing.graph"
    axial = work / "axial.graph"
    analysed = work / "axial_analysed.graph"
    csv_out = ROOT / "out" / f"{map_name}_axial_lines.csv"
    steps = [
        ("导入 DXF", ["-m", "IMPORT", "-f", str(dxf), "-o", str(graph),
                   "-it", "drawing"]),
        ("转成轴线图", ["-m", "MAPCONVERT", "-f", str(graph), "-o", str(axial),
                    "-co", "axial", "-con", "axial"]),
        ("轴向分析", ["-m", "AXIAL", "-f", str(axial), "-o", str(analysed),
                   "-xa", "n", "-xac", "-xal", "-xar"]),
        ("导出 CSV", ["-m", "EXPORT", "-f", str(analysed), "-o", str(csv_out),
                   "-em", "shapegraph-map-csv"]),
    ]
    for label, args in steps:
        proc = run_cli(*args)
        ok = proc.returncode == 0
        print(f"    depthmapX {label}: {'ok' if ok else 'FAILED'}")
        if not ok:
            raise SystemExit(f"depthmapX 步骤失败：{label}\n{proc.stdout}\n{proc.stderr}")
    return csv_out


def load_lines(csv_path: Path) -> pl.DataFrame:
    df = pl.read_csv(csv_path, infer_schema_length=2000)
    # depthmapX 导出的列名带空格与方括号，统一转成 snake_case，
    # 例如 "Integration [HH]" -> integration_hh、"Choice [Norm]" -> choice_norm
    def snake(name: str) -> str:
        s = name.strip().lower()
        s = s.replace(" ", "_").replace("[", "_").replace("]", "").replace("-", "_")
        while "__" in s:
            s = s.replace("__", "_")
        return s.strip("_")

    return df.rename({col: snake(col) for col in df.columns})


def kills_per_line(df: pl.DataFrame, map_name: str) -> pl.DataFrame:
    """把比赛里的交火点（攻守位置中点）分配给最近的轴线，得到每条线的交火密度。"""
    kills = pl.read_parquet(ROOT / "out" / "behaviour" / f"{map_name}_player_death.parquet")
    nav = NavMesh.from_npz(ROOT / "data" / "game_nav" / f"{map_name}_navmesh.npz")
    overview = read_overview_all(ROOT / "data" / f"{map_name}.txt")
    radar = Image.open(ROOT / "data" / f"{map_name}.png").convert("RGBA")
    tf = detect_transform(overview, np.array(radar), nav, verbose=False)

    vx, vy = tf.world_to_radar(kills["user_X"].cast(pl.Float64).to_numpy(),
                               kills["user_Y"].cast(pl.Float64).to_numpy())
    ax, ay = tf.world_to_radar(kills["attacker_X"].cast(pl.Float64).to_numpy(),
                               kills["attacker_Y"].cast(pl.Float64).to_numpy())
    px = (np.asarray(vx, dtype=float) + np.asarray(ax, dtype=float)) / 2.0
    py = (np.asarray(vy, dtype=float) + np.asarray(ay, dtype=float)) / 2.0

    x1 = df["x1"].to_numpy().astype(float)
    y1 = df["y1"].to_numpy().astype(float)
    x2 = df["x2"].to_numpy().astype(float)
    y2 = df["y2"].to_numpy().astype(float)
    counts = np.zeros(len(df), dtype=float)
    for i in range(len(px)):
        dx, dy = x2 - x1, y2 - y1
        seg_len = np.hypot(dx, dy)
        seg_len[seg_len == 0] = 1e-9
        t = np.clip(((px[i] - x1) * dx + (py[i] - y1) * dy) / (seg_len ** 2), 0, 1)
        dist = np.hypot(px[i] - (x1 + t * dx), py[i] - (y1 + t * dy))
        counts[int(np.argmin(dist))] += 1
    length_m = df["line_length"].to_numpy().astype(float) * 0.01905
    length_m[length_m <= 0] = 1e-9
    return df.with_columns(
        pl.Series("crossfire", counts),
        pl.Series("crossfire_per_m", counts / length_m),
        pl.Series("length_m", length_m),
    )


def intelligibility(df: pl.DataFrame) -> dict:
    conn = df["connectivity"].to_numpy().astype(float)
    integ = df["integration_hh"].to_numpy().astype(float)
    ok = np.isfinite(conn) & np.isfinite(integ) & (conn > 0) & (integ > 0)
    r = float(np.corrcoef(conn[ok], integ[ok])[0, 1])
    r_log = float(np.corrcoef(np.log(conn[ok]), np.log(integ[ok]))[0, 1])
    slope, intercept = np.polyfit(conn[ok], integ[ok], 1)
    return {
        "lines": int(ok.sum()),
        "r": round(r, 4),
        "r2": round(r ** 2, 4),
        "r_log": round(r_log, 4),
        "r2_log": round(r_log ** 2, 4),
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "p_value": float(stats.pearsonr(conn[ok], integ[ok])[1]),
    }


def vga_intelligibility(map_name: str) -> dict:
    grid = pl.read_csv(ROOT / "out" / f"{map_name}_syntax_grid.csv")
    conn = grid["connectivity"].to_numpy().astype(float)
    integ = grid["integration_hh_t"].to_numpy().astype(float)
    ok = np.isfinite(conn) & np.isfinite(integ) & (conn > 0) & (integ > 0)
    r = float(np.corrcoef(conn[ok], integ[ok])[0, 1])
    return {"cells": int(ok.sum()), "r": round(r, 4), "r2": round(r ** 2, 4)}


def configuration_experiment(df: pl.DataFrame) -> dict:
    integ = df["integration_hh"].to_numpy().astype(float)
    choice = df["choice"].to_numpy().astype(float)
    dens = df["crossfire_per_m"].to_numpy().astype(float)
    counts = df["crossfire"].to_numpy().astype(float)
    ok = np.isfinite(integ) & np.isfinite(dens)
    n = len(integ)
    order = np.argsort(integ)
    low = order[: max(n // 5, 1)]
    high = order[-max(n // 5, 1):]
    ratio = float("nan")
    if dens[low].mean() > 0:
        ratio = float(dens[high].mean() / dens[low].mean())
    return {
        "lines": int(ok.sum()),
        "crossfire_total": int(counts.sum()),
        "integration_vs_crossfire_rho": round(float(stats.spearmanr(integ[ok], dens[ok])[0]), 4),
        "integration_vs_crossfire_r": round(float(np.corrcoef(integ[ok], dens[ok])[0, 1]), 4),
        "choice_vs_crossfire_rho": round(float(stats.spearmanr(choice[ok], dens[ok])[0]), 4),
        "integration_vs_count_rho": round(float(stats.spearmanr(integ[ok], counts[ok])[0]), 4),
        "top20pct_mean_crossfire_per_m": round(float(dens[high].mean()), 3),
        "bottom20pct_mean_crossfire_per_m": round(float(dens[low].mean()), 3),
        "ratio_top_over_bottom": round(ratio, 3) if np.isfinite(ratio) else None,
    }


def figure_axial_map(map_name: str, df: pl.DataFrame, result: dict) -> None:
    radar = result["radar"]
    canvas = Image.new("RGBA", radar.size, (255, 255, 255, 255))
    canvas.alpha_composite(radar)
    base = np.array(canvas.convert("RGB"))
    integ = df["integration_hh"].to_numpy().astype(float)
    norm = (integ - np.nanmin(integ)) / (np.nanmax(integ) - np.nanmin(integ) + 1e-9)
    cmap = plt.get_cmap("turbo")
    dens = df["crossfire_per_m"].to_numpy().astype(float)
    dnorm = dens / (np.nanmax(dens) + 1e-9)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), dpi=100)
    for ax, color_by, title in (
        (axes[0], "integration", "轴线整合度 Integration[HH]（depthmapX 计算）"),
        (axes[1], "crossfire", "交火热点（线宽/颜色 = 每米交火次数）"),
    ):
        ax.imshow(base, extent=[0, radar.size[0], radar.size[1], 0])
        for i, row in enumerate(df.iter_rows(named=True)):
            if color_by == "integration":
                color, width = cmap(float(norm[i])), 3.0
            else:
                color, width = plt.get_cmap("inferno")(float(dnorm[i])), 1.5 + 5.0 * float(dnorm[i])
            ax.plot([row["x1"], row["x2"]], [row["y1"], row["y2"]],
                    color=color, linewidth=width, solid_capstyle="round")
        ax.set_title(f"{map_name} — {title}")
        ax.set_xlim(0, radar.size[0])
        ax.set_ylim(radar.size[1], 0)
        ax.set_axis_off()
    fig.tight_layout()
    path = ROOT / "out" / f"{map_name}_axial_map.png"
    fig.savefig(path, facecolor="black")
    plt.close(fig)
    print(f"  wrote {path}")


def figure_scatter(map_name: str, df: pl.DataFrame, intel: dict, conf: dict) -> None:
    integ = df["integration_hh"].to_numpy().astype(float)
    conn = df["connectivity"].to_numpy().astype(float)
    dens = df["crossfire_per_m"].to_numpy().astype(float)
    choice = df["choice"].to_numpy().astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), dpi=100)
    axes[0].scatter(conn, integ, s=26, c="#1f77b4")
    axes[0].set_xlabel("连通度 Connectivity（局部）")
    axes[0].set_ylabel("全局整合度 Integration[HH]")
    axes[0].set_title(f"{map_name} 可理解度 R²={intel['r2']:.3f}（对数 {intel['r2_log']:.3f}）")
    axes[1].scatter(integ, dens, s=26, c="#d62728")
    axes[1].set_xlabel("Integration[HH]")
    axes[1].set_ylabel("交火密度（次/米）")
    axes[1].set_title(f"实验一：整合度 × 交火 ρ={conf['integration_vs_crossfire_rho']:+.3f}")
    axes[2].scatter(choice, dens, s=26, c="#2ca02c")
    axes[2].set_xlabel("Choice（选择度）")
    axes[2].set_ylabel("交火密度（次/米）")
    axes[2].set_title(f"实验一：选择度 × 交火 ρ={conf['choice_vs_crossfire_rho']:+.3f}")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    path = ROOT / "out" / f"{map_name}_axial_scatter.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


def analyse_map(map_name: str, cell: int = 4) -> dict:
    print(f"\n=== {map_name} ===")
    dxf = ROOT / "out" / f"{map_name}_axial.dxf"
    result = axial_map.build(map_name, cell)
    axial_map.write_dxf(result["segments"], dxf)
    axial_map.draw_preview(result, ROOT / "out" / f"{map_name}_axial_preview.png", map_name)
    print(f"  轴线 {len(result['segments'])} 条 → 交给 depthmapX")
    csv_path = depthmapx_axial(map_name, dxf)
    df = kills_per_line(load_lines(csv_path), map_name)
    df.write_csv(csv_path)
    intel = intelligibility(df)
    vga = vga_intelligibility(map_name)
    conf = configuration_experiment(df)
    figure_axial_map(map_name, df, result)
    figure_scatter(map_name, df, intel, conf)
    print(f"  R²={intel['r2']}（对数 {intel['r2_log']}）｜"
          f"整合度×交火 ρ={conf['integration_vs_crossfire_rho']:+.3f}｜"
          f"选择度×交火 ρ={conf['choice_vs_crossfire_rho']:+.3f}")
    return {"map": map_name, "axial_lines": len(df), "intelligibility": intel,
            "intelligibility_vga": vga, "configuration": conf}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("maps", nargs="*", default=["de_dust2", "de_mirage"])
    ap.add_argument("--cell", type=int, default=4)
    args = ap.parse_args()
    if not CLI.exists():
        raise SystemExit(f"缺少 {CLI}（depthmapX 命令行版）")
    results = [analyse_map(m, args.cell) for m in args.maps]
    exp1 = {r["map"]: r["configuration"] for r in results}
    exp2 = {r["map"]: {"axial": r["intelligibility"], "vga": r["intelligibility_vga"],
                       "axial_lines": r["axial_lines"]} for r in results}
    (ROOT / "out" / "exp1_configuration.json").write_text(
        json.dumps(exp1, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "out" / "exp2_intelligibility.json").write_text(
        json.dumps(exp2, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n实验一 -> out/exp1_configuration.json")
    print("实验二 -> out/exp2_intelligibility.json")
    figure_comparison(results)


def figure_comparison(results: list[dict]) -> None:
    """两图并列：实验一的相关性 + 实验二两种口径的 R²。"""
    maps = [r["map"] for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), dpi=100)
    x = np.arange(len(maps))
    w = 0.35
    axes[0].bar(x - w / 2, [r["configuration"]["integration_vs_crossfire_rho"] for r in results],
                w, label="整合度 × 交火密度")
    axes[0].bar(x + w / 2, [r["configuration"]["choice_vs_crossfire_rho"] for r in results],
                w, label="选择度 × 交火密度")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(maps)
    axes[0].axhline(0, color="k", linewidth=0.8)
    axes[0].set_ylabel("Spearman ρ")
    axes[0].set_title("实验一：构形指标与交火密度的关系")
    axes[0].legend()
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar(x - w / 2, [r["intelligibility"]["r2"] for r in results], w, label="轴向图 R²")
    axes[1].bar(x + w / 2, [r["intelligibility"]["r2_log"] for r in results], w,
                label="轴向图 R²（对数）")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(maps)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("R²")
    axes[1].set_title("实验二：可理解度（轴线图口径）")
    axes[1].legend()
    axes[1].grid(alpha=0.25, axis="y")

    axes[2].bar(x, [r["intelligibility_vga"]["r2"] for r in results], 0.5, color="#8c564b")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(maps)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("R²")
    axes[2].set_title("实验二：可理解度（VGA 栅格口径，交叉验证）")
    axes[2].grid(alpha=0.25, axis="y")

    fig.tight_layout()
    path = ROOT / "out" / "exp_comparison.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
