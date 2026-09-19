r"""把空间句法指标和真实行为数据对上，做相关分析。

这是"实验一"的核心一步：构形指标（整合度/选择度/等视域面积）来自几何，
行为指标来自 demo。两者按同一套格子对齐后，就能检验
"构形上更整合的空间是否真的被用得更多"。

行为指标有两种口径，分开看才有意义：
  * 全回合占位   —— 包含包点、卡点、残局这些被战术强制的位置；
  * 开局 15 秒占位 —— 战术尚未展开，更接近"自然运动"。

用法：
    .venv\\Scripts\\python.exe analyse_behaviour.py de_mirage
    .venv\\Scripts\\python.exe analyse_behaviour.py de_mirage --window 20

输出：
    out\\<map>_behaviour_grid.csv    格子级的句法 + 行为表
    out\\<map>_behaviour.png         散点/分箱曲线/对照图
    out\\<map>_behaviour.json        相关系数与统计摘要
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
from PIL import Image
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from heatmap import detect_transform, read_overview_all  # noqa: E402
from navmesh import NavMesh  # noqa: E402

METRIC_COLUMNS = ("connectivity", "isovist_area_px", "isovist_perimeter",
                  "integration_hh_t", "integration_hh_m", "choice_t")


def load_syntax(map_name: str, root: Path) -> pl.DataFrame:
    path = root / "out" / f"{map_name}_syntax_grid.csv"
    if not path.exists():
        raise SystemExit(f"缺少 {path}，先跑 space_syntax.py {map_name}")
    return pl.read_csv(path)


def load_behaviour(map_name: str, root: Path,
                   behaviour_dir: Path | None = None) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """读行为数据：优先用 ingest_demos.py 按地图分好的表，其次退回单 demo 的旧表。

    两种布局都会校验地图是否对得上——这是之前踩过的坑（拿 Mirage 的轨迹去画
    dust2 的格子，结果不会报错，只会静默出错）。
    """
    beh = behaviour_dir or (root / "out" / "behaviour")
    per_map = {
        "tracks": beh / f"{map_name}_tracks.parquet",
        "kills": beh / f"{map_name}_player_death.parquet",
        "meta": beh / f"{map_name}_meta.json",
    }
    if per_map["tracks"].exists():
        paths = per_map
    else:
        paths = {
            "tracks": root / "out" / "tracks.parquet",
            "kills": root / "out" / "player_death.parquet",
            "meta": root / "out" / "tracks_meta.json",
        }
    if not paths["tracks"].exists():
        raise SystemExit(
            f"没有 {map_name} 的行为数据。\n"
            f"把 demo 放进 demos\\ 后跑：\n"
            f"  .venv\\Scripts\\python.exe ingest_demos.py --only {map_name}\n"
            f"几何层的分析不受影响，可以先跑 space_syntax.py {map_name}。"
        )

    meta = {}
    if paths["meta"].exists():
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        demo_map = meta.get("map_name")
        if demo_map and demo_map != map_name:
            raise SystemExit(
                f"地图不一致：行为数据来自 {demo_map}，但要分析的是 {map_name}。\n"
                f"用 ingest_demos.py --only {map_name} 重新抽一遍。"
            )
    demo_count = meta.get("demo_count", 1 if meta.get("demos") else 0)
    rounds = meta.get("rounds", meta.get("rounds_total", "?"))
    print(f"  行为数据：{demo_count} 场 demo / {rounds} 回合"
          f"（{paths['tracks'].relative_to(root)}）")
    return pl.read_parquet(paths["tracks"]), pl.read_parquet(paths["kills"]), meta


def map_points_to_cells(frame: pl.DataFrame, tf, cell: int, cols: int, rows: int,
                        xcol: str = "X", ycol: str = "Y"):
    px, py = tf.world_to_radar(
        frame[xcol].cast(pl.Float64).to_numpy(),
        frame[ycol].cast(pl.Float64).to_numpy(),
    )
    col = np.floor(px / cell).astype(np.int64)
    row = np.floor(py / cell).astype(np.int64)
    inside = (col >= 0) & (col < cols) & (row >= 0) & (row < rows)
    return col, row, inside


def aggregate(col: np.ndarray, row: np.ndarray, mask: np.ndarray,
              cols: int, rows: int) -> np.ndarray:
    grid = np.zeros((rows, cols), dtype=np.float64)
    np.add.at(grid, (row[mask], col[mask]), 1.0)
    return grid


def spearman(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 8 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {"rho": None, "p": None, "n": int(len(x))}
    rho, p = stats.spearmanr(x, y)
    return {"rho": round(float(rho), 4), "p": float(p), "n": int(len(x))}


def pearson(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 8 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {"r": None, "p": None}
    r, p = stats.pearsonr(x, y)
    return {"r": round(float(r), 4), "p": float(p)}


def hotspot_profile(table: pl.DataFrame, metric: str, behaviour: str,
                    quantile: float = 0.9) -> dict:
    """对比"行为热点格"和"构形热点格"是不是同一批格子。

    秩相关只刻画单调关系，而战术游戏里更常见的是"少数格子被反复使用"，
    所以额外给一个热点对比：行为前 10% 的格子，其中位整合度是多少；
    反过来构形前 10% 的格子，其中位占位又是多少。
    """
    m = table[metric].to_numpy().astype(float)
    b = table[behaviour].to_numpy().astype(float)
    thr_b = np.quantile(b, quantile)
    thr_m = np.quantile(m, quantile)
    hot_b = b >= thr_b
    hot_m = m >= thr_m
    return {
        "behaviour_top10pct_cells": int(hot_b.sum()),
        "behaviour_top10pct_median_metric": round(float(np.median(m[hot_b])), 3),
        "all_cells_median_metric": round(float(np.median(m)), 3),
        "metric_top10pct_median_behaviour": round(float(np.median(b[hot_m])), 3),
        "all_cells_median_behaviour": round(float(np.median(b)), 3),
        "overlap_cells": int((hot_b & hot_m).sum()),
        "overlap_expected_if_independent": round(float(hot_b.sum() * hot_m.sum()
                                                      / len(b)), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("map", help="地图名，必须和 demos/ 里那份 demo 的地图一致")
    ap.add_argument("--window", type=float, default=15.0, help="开局窗口（秒），默认 15")
    ap.add_argument("--cell", type=int, default=None, help="格子边长，默认沿用句法表的设置")
    ap.add_argument("--behaviour-dir", type=Path, default=Path("out/behaviour"),
                    help="行为数据目录（默认 out/behaviour；做稳健性检查时可指向 "
                         "out/behaviour_pug）")
    ap.add_argument("--tag", default="",
                    help="给输出文件名加后缀，避免覆盖主线结果（例如 --tag pug）")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    syntax = load_syntax(args.map, root)

    cell = args.cell
    if cell is None:
        lefts = syntax["px_left"].unique().sort().to_numpy()
        cell = int(lefts[1] - lefts[0])
    cols = int(syntax["col"].max()) + 1
    rows = int(syntax["row"].max()) + 1
    print(f"{args.map}: 句法栅格 {cols}×{rows}，格子 {cell}px，"
          f"{syntax.height} 个可通行格子")

    overview = read_overview_all(root / "data" / f"{args.map}.txt")
    radar = Image.open(root / "data" / f"{args.map}.png").convert("RGBA")
    nav = NavMesh.from_npz(root / "data" / "game_nav" / f"{args.map}_navmesh.npz")
    tf = detect_transform(overview, np.array(radar), nav, verbose=False)

    behaviour_dir = args.behaviour_dir if args.behaviour_dir.is_absolute() else root / args.behaviour_dir
    tracks, kills, demo_meta = load_behaviour(args.map, root, behaviour_dir)

    col, row, inside = map_points_to_cells(tracks, tf, cell, cols, rows)
    print(f"轨迹样本 {len(tracks):,}，落在画面内 {inside.mean()*100:.2f}%")

    valid_keys = set((syntax["row"] * cols + syntax["col"]).to_list())
    track_key = row * cols + col
    in_grid = inside & np.isin(track_key, np.fromiter(valid_keys, dtype=np.int64))
    print(f"  其中落在句法格子内 {in_grid.mean()*100:.2f}%（其余多在贴墙一格，被腐蚀掉）")

    occ_all = aggregate(col, row, in_grid, cols, rows)

    ct = tracks.filter(pl.col("side") == "CT")
    t_only = tracks.filter(pl.col("side") == "T")
    ccol, crow, cinside = map_points_to_cells(ct, tf, cell, cols, rows)
    tcol, trow, tinside = map_points_to_cells(t_only, tf, cell, cols, rows)
    cok = cinside & np.isin(crow * cols + ccol, np.fromiter(valid_keys, dtype=np.int64))
    tok = tinside & np.isin(trow * cols + tcol, np.fromiter(valid_keys, dtype=np.int64))
    occ_ct = aggregate(ccol, crow, cok, cols, rows)
    occ_t_grid = aggregate(tcol, trow, tok, cols, rows)

    early = tracks.filter(
        (pl.col("seconds_after_freeze") >= 0) & (pl.col("seconds_after_freeze") <= args.window)
    )
    ecol, erow, einside = map_points_to_cells(early, tf, cell, cols, rows)
    eok = einside & np.isin(erow * cols + ecol, np.fromiter(valid_keys, dtype=np.int64))
    occ_early = aggregate(ecol, erow, eok, cols, rows)
    print(f"开局 {args.window:.0f}s 内样本 {len(early):,}，"
          f"落在句法格子内 {eok.mean()*100:.2f}%")

    kcol, krow, kinside = map_points_to_cells(kills, tf, cell, cols, rows,
                                              xcol="attacker_X", ycol="attacker_Y")
    kok = kinside & np.isin(krow * cols + kcol, np.fromiter(valid_keys, dtype=np.int64))
    kills_grid = aggregate(kcol, krow, kok, cols, rows)

    side_kills = {}
    for name in ("CT", "T"):
        sub = kills.filter(pl.col("attacker_side") == name)
        scol, srow, sinside = map_points_to_cells(sub, tf, cell, cols, rows,
                                                  xcol="attacker_X", ycol="attacker_Y")
        sok = sinside & np.isin(srow * cols + scol, np.fromiter(valid_keys, dtype=np.int64))
        side_kills[name] = aggregate(scol, srow, sok, cols, rows)

    # 通行流量：同一玩家在同一回合内相邻采样之间的移动，统计进/出该格子的次数
    flow = np.zeros((rows, cols), dtype=np.float64)
    visits_rounds = np.zeros((rows, cols), dtype=np.float64)
    cell_flat = row * cols + col
    sel = in_grid
    sid_all = tracks["steamid"].to_numpy()
    rnd_all = tracks["round_index"].to_numpy()
    tick_all = tracks["tick"].to_numpy()
    sid, rnd, tick, key = sid_all[sel], rnd_all[sel], tick_all[sel], cell_flat[sel]
    if key.size > 1:
        order = np.lexsort((tick, rnd, sid))
        sid, rnd, tick, key = sid[order], rnd[order], tick[order], key[order]
        same = (sid[1:] == sid[:-1]) & (rnd[1:] == rnd[:-1])
        gap = np.diff(tick)
        # 采样步长 16 tick；超过 2 倍说明中间有缺口（死亡/换局），不算一步移动
        step_ok = (gap > 0) & (gap <= 32)
        keep = same & step_ok
        moved = keep & (key[1:] != key[:-1])
        flow_grid_flat = np.bincount(key[1:][moved], minlength=rows * cols) + \
            np.bincount(key[:-1][moved], minlength=rows * cols)
        flow = flow_grid_flat.reshape(rows, cols)
        uniq = np.unique(rnd * (rows * cols) + key)
        visits_flat = np.bincount(uniq % (rows * cols), minlength=rows * cols)
        visits_rounds = visits_flat.reshape(rows, cols)
        print(f"通行流量：{int(moved.sum()):,} 次格间移动，"
              f"覆盖 {int((flow > 0).sum())} 个格子")

    gr, gc = syntax["row"].to_numpy(), syntax["col"].to_numpy()
    table = syntax.with_columns(
        pl.Series("occ_all", occ_all[gr, gc]),
        pl.Series("occ_ct", occ_ct[gr, gc]),
        pl.Series("occ_t", occ_t_grid[gr, gc]),
        pl.Series("occ_early", occ_early[gr, gc]),
        pl.Series("kills", kills_grid[gr, gc]),
        pl.Series("kills_ct", side_kills["CT"][gr, gc]),
        pl.Series("kills_t", side_kills["T"][gr, gc]),
        pl.Series("flow", flow[gr, gc]),
        pl.Series("visits_rounds", visits_rounds[gr, gc]),
    )
    suffix = f"_{args.tag}" if args.tag else ""
    grid_path = root / "out" / f"{args.map}_behaviour_grid{suffix}.csv"
    table.write_csv(grid_path)
    print(f"wrote {grid_path.relative_to(root)}")

    results = {}
    behaviour_columns = ("occ_all", "occ_early", "flow", "visits_rounds", "kills",
                         "occ_ct", "occ_t", "kills_ct", "kills_t")
    for behaviour in behaviour_columns:
        y = table[behaviour].to_numpy().astype(float)
        results[behaviour] = {}
        for metric in METRIC_COLUMNS:
            x = table[metric].to_numpy().astype(float)
            results[behaviour][metric] = {**spearman(x, y), **pearson(x, y)}

    print("\n分阵营 Spearman（指标 -> 行为）:")
    print(f"{'指标':<20}{'CT占位':>10}{'T占位':>10}{'CT击杀':>10}{'T击杀':>10}")
    for metric in METRIC_COLUMNS:
        cells = []
        for behaviour in ("occ_ct", "occ_t", "kills_ct", "kills_t"):
            rho = results[behaviour][metric]["rho"]
            cells.append("  n/a" if rho is None else f"{rho:+.3f}")
        print(f"{metric:<20}{cells[0]:>10}{cells[1]:>10}{cells[2]:>10}{cells[3]:>10}")

    results["_hotspot_vs_integration"] = {
        "occ_all": hotspot_profile(table, "integration_hh_t", "occ_all"),
        "occ_early": hotspot_profile(table, "integration_hh_t", "occ_early"),
        "flow": hotspot_profile(table, "integration_hh_t", "flow"),
        "choice_t": hotspot_profile(table, "choice_t", "occ_all"),
    }
    results["_note"] = ("Spearman 秩相关，n = 可通行格子数。空间数据存在自相关，"
                        "p 值只能当参考，不能当独立样本的显著性。")
    results["_samples"] = {
        "tracks_total": int(len(tracks)),
        "tracks_in_grid": int(in_grid.sum()),
        "tracks_early": int(len(early)),
        "early_window_seconds": args.window,
        "kills_total": int(len(kills)),
        "kills_in_grid": int(kok.sum()),
        "cells": int(syntax.height),
        "cell_px": int(cell),
    }

    print("\nSpearman 相关（指标 -> 行为）:")
    print(f"{'指标':<20}{'全回合占位':>12}{'开局占位':>12}{'通行流量':>12}"
          f"{'出现回合数':>12}{'击杀':>10}")
    for metric in METRIC_COLUMNS:
        cells = []
        for behaviour in ("occ_all", "occ_early", "flow", "visits_rounds", "kills"):
            rho = results[behaviour][metric]["rho"]
            cells.append("  n/a" if rho is None else f"{rho:+.3f}")
        print(f"{metric:<20}{cells[0]:>12}{cells[1]:>12}{cells[2]:>12}"
              f"{cells[3]:>12}{cells[4]:>10}")

    print("\nPearson 线性相关:")
    print(f"{'指标':<20}{'全回合占位':>12}{'开局占位':>12}{'通行流量':>12}")
    for metric in METRIC_COLUMNS:
        cells = []
        for behaviour in ("occ_all", "occ_early", "flow"):
            r = results[behaviour][metric]["r"]
            cells.append("  n/a" if r is None else f"{r:+.3f}")
        print(f"{metric:<20}{cells[0]:>12}{cells[1]:>12}{cells[2]:>12}")

    print("\n热点对比（行为前 10% 的格子 vs 构形前 10% 的格子）:")
    for key, prof in results["_hotspot_vs_integration"].items():
        print(f"  {key}: 行为热点格中位整合度 {prof['behaviour_top10pct_median_metric']}"
              f" / 全体中位 {prof['all_cells_median_metric']}；"
              f"构形热点格中位行为 {prof['metric_top10pct_median_behaviour']}"
              f" / 全体中位 {prof['all_cells_median_behaviour']}；"
              f"重合 {prof['overlap_cells']} 格"
              f"（独立假设下期望 {prof['overlap_expected_if_independent']} 格）")

    with open(root / "out" / f"{args.map}_behaviour{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    draw_figure(args.map, table, np.array(radar), root, args.window, cell, rows, cols, suffix)


def draw_figure(map_name: str, table: pl.DataFrame, radar_arr: np.ndarray, root: Path,
                window: float, cell: int, rows: int, cols: int, suffix: str = "") -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    integ = table["integration_hh_t"].to_numpy().astype(float)
    occ = table["occ_all"].to_numpy().astype(float)
    occ_e = table["occ_early"].to_numpy().astype(float)
    flow = table["flow"].to_numpy().astype(float)

    fig = plt.figure(figsize=(26, 12), dpi=100)
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.scatter(integ, occ, s=9, alpha=0.5, c="#1f77b4", edgecolors="none")
    ax1.set_xlabel("Integration[HH]（拓扑）")
    ax1.set_ylabel("全回合占位样本数")
    ax1.set_title(f"{map_name}: 整合度 vs 全回合占位")

    ax2 = fig.add_subplot(2, 4, 2)
    ax2.scatter(integ, occ_e, s=9, alpha=0.5, c="#d62728", edgecolors="none")
    ax2.set_xlabel("Integration[HH]（拓扑）")
    ax2.set_ylabel(f"冻结后 {window:.0f}s 内占位样本数")
    ax2.set_title("整合度 vs 开局占位（更接近自然运动）")

    ax3 = fig.add_subplot(2, 4, 3)
    for y, label, color in ((occ, "全回合占位", "#1f77b4"),
                            (occ_e, f"冻结后 {window:.0f}s 占位", "#d62728"),
                            (flow, "通行流量", "#2ca02c")):
        order = np.argsort(integ)
        edges = np.array_split(order, 10)
        xs = [integ[idx].mean() for idx in edges]
        ys = [y[idx].mean() for idx in edges]
        err = [y[idx].std() / max(np.sqrt(len(idx)), 1) for idx in edges]
        ax3.errorbar(xs, ys, yerr=err, marker="o", capsize=3, label=label, color=color)
    ax3.set_xlabel("Integration[HH]（分箱均值）")
    ax3.set_ylabel("平均强度（各自量纲，看趋势）")
    ax3.set_title("按整合度分箱：三种行为指标的变化")
    ax3.legend()

    ax4 = fig.add_subplot(2, 4, 4)
    ax4.scatter(integ, flow, s=9, alpha=0.5, c="#2ca02c", edgecolors="none")
    ax4.set_xlabel("Integration[HH]（拓扑）")
    ax4.set_ylabel("通行流量（进+出次数）")
    ax4.set_title("整合度 vs 通行流量（移动经济）")

    def grid_layer(name: str) -> np.ndarray:
        layer = np.full((rows, cols), np.nan)
        if name in table.columns:
            layer[table["row"].to_numpy(), table["col"].to_numpy()] = \
                table[name].to_numpy().astype(float)
        return layer

    for idx, (name, title, cmap) in enumerate((
        ("integration_hh_t", "整合度", "turbo"),
        ("occ_all", "全回合占位", "magma"),
        ("occ_early", f"冻结后 {window:.0f}s 占位", "viridis"),
        ("flow", "通行流量", "cividis"),
    ), start=5):
        ax = fig.add_subplot(2, 4, idx)
        layer = grid_layer(name)
        ax.imshow(radar_arr, extent=[0, cols * cell, rows * cell, 0])
        finite = np.isfinite(layer)
        if finite.any():
            lo, hi = np.nanpercentile(layer, [2, 98])
            ax.imshow(np.ma.masked_invalid(layer),
                      extent=[0, cols * cell, rows * cell, 0],
                      cmap=cmap, alpha=0.75, vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.set_axis_off()

    fig.tight_layout()
    path = root / "out" / f"{map_name}_behaviour{suffix}.png"
    fig.savefig(path, facecolor="black")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
