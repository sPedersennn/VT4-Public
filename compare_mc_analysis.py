"""
compare_mc_analysis.py
======================
Runs both the DES (GA-scheduled) and SOT Monte Carlo simulations and produces
combined comparison plots showing how the two scheduling approaches differ
across all KPIs.

Outputs (all in the working directory)
  compare_summary_<dist>_<n>jobs.png    — overlaid KPI histograms (4 panels)
  compare_throughput_<dist>_<n>jobs.png — weekly throughput comparison (2 panels)
  compare_violin_<n>jobs.png            — violin comparison across distributions
  compare_convergence_<n>jobs.png       — GA convergence curve
"""
from __future__ import annotations

import argparse
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import mc_analysis_SA          as mc_des
import mc_analysis_current_sot as mc_sot

from ga_simulationBC_current import (
    load_jobs      as load_jobs_des,
    run_ga,
    plot_convergence,
    N_WORKERS_B,
    N_WORKERS_C,
    GA_POP_SIZE,
    GA_GENERATIONS,
    MAX_GENERATED_JOBS,
    WEEK_HOURS,
)
from sot_simulationBC_current import Job as SotJob
from constraintsBC import CHANGEOVER_TIME

DIST_NAMES    = mc_des.DIST_NAMES
DES_COLOR     = "#5BA4CF"
SOT_COLOR     = "#F5A623"
# SOT scatter colormap: orange (low fitness = good) → blue (high fitness = bad)
_SOT_CMAP = LinearSegmentedColormap.from_list("sot_scatter", [SOT_COLOR, "#2C7FB8"])


# ---------------------------------------------------------------------------
# Shared oven-utilization helper
# ---------------------------------------------------------------------------

def _oven_util(results: list[dict]) -> np.ndarray:
    return np.array([
        sum(j.oven_end - j.oven_start + CHANGEOVER_TIME
            for j in r["_jobs"]
            if j.oven_start is not None and j.oven_end is not None)
        / r["makespan_h"] * 100.0
        for r in results
    ])


# ---------------------------------------------------------------------------
# Shared weekly-arrays helper
# ---------------------------------------------------------------------------

def _weekly_arrays(results: list[dict]) -> np.ndarray:
    """Return a (n_reps × max_weeks) array of per-week blade counts."""
    all_counts = []
    for r in results:
        done = [j for j in r["_jobs"] if j.oven_end is not None]
        if not done:
            all_counts.append(np.zeros(1, dtype=int))
            continue
        ms    = max(j.oven_end for j in done)
        n_wks = max(1, int(np.ceil(ms / WEEK_HOURS)))
        wk_counts = np.zeros(n_wks, dtype=int)
        for j in done:
            wk = min(int(j.oven_end / WEEK_HOURS), n_wks - 1)
            wk_counts[wk] += 1
        all_counts.append(wk_counts)
    max_wks = max(len(c) for c in all_counts)
    padded  = np.zeros((len(all_counts), max_wks), dtype=float)
    for i, c in enumerate(all_counts):
        padded[i, : len(c)] = c
    return padded


def _pad_to(arr: np.ndarray, w: int) -> np.ndarray:
    if arr.shape[1] < w:
        arr = np.hstack([arr, np.zeros((arr.shape[0], w - arr.shape[1]))])
    return arr


# ---------------------------------------------------------------------------
# 4-panel summary comparison
# ---------------------------------------------------------------------------

def plot_comparison_summary(
    des_results: list[dict],
    sot_results: list[dict],
    dist:        str,
    n_jobs:      int,
    save_path:   Optional[str] = None,
) -> None:
    """
    4-panel figure overlaying DES and SOT distributions:
      Panel 1 — Makespan histograms
      Panel 2 — SC-2 track blocking histograms
      Panel 3 — Oven utilization histograms
      Panel 4 — Delay vs makespan scatter (DES: RdYlGn_r, SOT: RdYlGn)
    """
    des_ms   = np.array([r["makespan_h"]   for r in des_results])
    sot_ms   = np.array([r["makespan_h"]   for r in sot_results])
    des_sc2  = np.array([r["sc2"]          for r in des_results])
    sot_sc2  = np.array([r["sc2"]          for r in sot_results])
    des_util = _oven_util(des_results)
    sot_util = _oven_util(sot_results)

    des_pad  = _weekly_arrays(des_results)
    sot_pad  = _weekly_arrays(sot_results)
    max_wks  = max(des_pad.shape[1], sot_pad.shape[1])
    des_pad  = _pad_to(des_pad, max_wks)
    sot_pad  = _pad_to(sot_pad, max_wks)
    x_pos_s  = np.arange(max_wks)
    wk_lbl_s = [f"Wk {w + 1}" for w in range(max_wks)]
    rot_s    = 45 if max_wks > 8 else 0
    sot_fit  = np.array([r["fitness"]      for r in sot_results])

    n_rep = len(des_results)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"DES (GA) vs SOT — {n_rep} replications each  |  "
        f"Delay: {dist.upper()}  |  {n_jobs} jobs",
        fontsize=13, fontweight="bold",
    )

    def _hist_panel(ax, des_arr, sot_arr, xlabel, title):
        bins = np.histogram_bin_edges(np.concatenate([des_arr, sot_arr]), bins=30)
        ax.hist(des_arr, bins=bins, color=DES_COLOR, alpha=0.60,
                edgecolor="white", linewidth=0.4, label="DES (GA)")
        ax.hist(sot_arr, bins=bins, color=SOT_COLOR, alpha=0.60,
                edgecolor="white", linewidth=0.4, label="SOT")
        ax.axvline(des_arr.mean(), color=DES_COLOR, linestyle="--", linewidth=1.8,
                   label=f"DES mean {des_arr.mean():.1f}")
        ax.axvline(sot_arr.mean(), color=SOT_COLOR, linestyle="--", linewidth=1.8,
                   label=f"SOT mean {sot_arr.mean():.1f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    _hist_panel(axes[0, 0], des_ms,   sot_ms,   "Makespan (h)",           "Makespan distribution")
    _hist_panel(axes[0, 1], des_sc2,  sot_sc2,  "SC-2 track blocking (h)", "SC-2 track blocking distribution")
    _hist_panel(axes[1, 0], des_util, sot_util, "Oven utilization (%)",   "Oven utilization distribution")

    # Panel 4 — per-week throughput
    des_mean_wk_s = des_pad.mean(axis=0)
    sot_mean_wk_s = sot_pad.mean(axis=0)
    des_std_wk_s  = des_pad.std(axis=0)
    sot_std_wk_s  = sot_pad.std(axis=0)
    ax = axes[1, 1]
    ax.fill_between(x_pos_s, des_mean_wk_s - des_std_wk_s, des_mean_wk_s + des_std_wk_s,
                    alpha=0.20, color=DES_COLOR)
    ax.fill_between(x_pos_s, sot_mean_wk_s - sot_std_wk_s, sot_mean_wk_s + sot_std_wk_s,
                    alpha=0.20, color=SOT_COLOR)
    ax.plot(x_pos_s, des_mean_wk_s, color=DES_COLOR, linewidth=2.0, marker="o", markersize=4,
            label=f"DES (GA)  avg={des_mean_wk_s.mean():.1f} bl/wk")
    ax.plot(x_pos_s, sot_mean_wk_s, color=SOT_COLOR, linewidth=2.0, marker="s", markersize=4,
            label=f"SOT  avg={sot_mean_wk_s.mean():.1f} bl/wk")
    ax.set_xticks(x_pos_s)
    ax.set_xticklabels(wk_lbl_s, fontsize=8, rotation=rot_s, ha="right")
    ax.set_xlabel(f"Simulation week  (1 week = {WEEK_HOURS:.0f} h)")
    ax.set_ylabel("Blades completed")
    ax.set_title("Per-week throughput  (mean ± 1σ)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        save_path = f"compare_summary_{dist}_{n_jobs}jobs.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Summary comparison saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Weekly throughput comparison
# ---------------------------------------------------------------------------

def plot_comparison_throughput(
    des_results: list[dict],
    sot_results: list[dict],
    dist:        str,
    n_jobs:      int,
    save_path:   Optional[str] = None,
) -> None:
    """
    2-panel figure: cumulative throughput and delay vs makespan scatter.
      Left  — Cumulative throughput (mean lines with P5/P95 dotted)
      Right — Delay vs makespan scatter (DES: RdYlGn_r, SOT: RdYlGn)
    """
    des_pad = _weekly_arrays(des_results)
    sot_pad = _weekly_arrays(sot_results)

    max_weeks = max(des_pad.shape[1], sot_pad.shape[1])
    des_pad   = _pad_to(des_pad, max_weeks)
    sot_pad   = _pad_to(sot_pad, max_weeks)

    x_pos       = np.arange(max_weeks)
    week_labels = [f"Wk {w + 1}" for w in range(max_weeks)]
    rot         = 45 if max_weeks > 8 else 0

    des_cum    = des_pad.cumsum(axis=1)
    sot_cum    = sot_pad.cumsum(axis=1)
    des_mean_c = des_cum.mean(axis=0)
    sot_mean_c = sot_cum.mean(axis=0)
    des_std_c  = des_cum.std(axis=0)
    sot_std_c  = sot_cum.std(axis=0)
    des_p5_c   = np.percentile(des_cum, 5,  axis=0)
    des_p95_c  = np.percentile(des_cum, 95, axis=0)
    sot_p5_c   = np.percentile(sot_cum, 5,  axis=0)
    sot_p95_c  = np.percentile(sot_cum, 95, axis=0)

    des_ms  = np.array([r["makespan_h"]   for r in des_results])
    sot_ms  = np.array([r["makespan_h"]   for r in sot_results])
    des_dly = np.array([r["mean_delay_h"] for r in des_results])
    sot_dly = np.array([r["mean_delay_h"] for r in sot_results])
    des_fit = np.array([r["fitness"]      for r in des_results])
    sot_fit = np.array([r["fitness"]      for r in sot_results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Cumulative throughput & delay — DES (GA) vs SOT  |  {len(des_results)} reps each  |  "
        f"Delay: {dist.upper()}  |  {n_jobs} jobs",
        fontsize=13, fontweight="bold",
    )

    # Left — cumulative throughput with P5/P95
    ax1.fill_between(x_pos, des_mean_c - des_std_c, des_mean_c + des_std_c,
                     alpha=0.18, color=DES_COLOR)
    ax1.fill_between(x_pos, sot_mean_c - sot_std_c, sot_mean_c + sot_std_c,
                     alpha=0.18, color=SOT_COLOR)
    ax1.plot(x_pos, des_mean_c, color=DES_COLOR, linewidth=2.0,
             marker="o", markersize=4,
             label=f"DES (GA)  P5={des_p5_c[-1]:.0f}  P95={des_p95_c[-1]:.0f}")
    ax1.plot(x_pos, sot_mean_c, color=SOT_COLOR, linewidth=2.0,
             marker="s", markersize=4,
             label=f"SOT  P5={sot_p5_c[-1]:.0f}  P95={sot_p95_c[-1]:.0f}")
    ax1.plot(x_pos, des_p5_c,  color=DES_COLOR, linestyle=":", linewidth=1.0)
    ax1.plot(x_pos, des_p95_c, color=DES_COLOR, linestyle=":", linewidth=1.0)
    ax1.plot(x_pos, sot_p5_c,  color=SOT_COLOR, linestyle=":", linewidth=1.0)
    ax1.plot(x_pos, sot_p95_c, color=SOT_COLOR, linestyle=":", linewidth=1.0)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(week_labels, fontsize=8, rotation=rot, ha="right")
    ax1.set_xlabel(f"Simulation week  (1 week = {WEEK_HOURS:.0f} h)")
    ax1.set_ylabel("Cumulative blades completed")
    ax1.set_title("Cumulative throughput  (mean ± 1σ, P5/P95 dotted)")
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Right — delay vs makespan scatter
    # DES: RdYlGn_r  (red = high fitness / bad, green = low / good)
    # SOT: orange→blue  (orange = low fitness / good, blue = high / bad)
    sc_des = ax2.scatter(des_dly, des_ms, c=des_fit, cmap="RdYlGn_r",
                         alpha=0.50, s=18, edgecolors="none", marker="o",
                         vmin=des_fit.min(), vmax=des_fit.max())
    sc_sot = ax2.scatter(sot_dly, sot_ms, c=sot_fit, cmap=_SOT_CMAP,
                         alpha=0.50, s=18, edgecolors="none", marker="^",
                         vmin=sot_fit.min(), vmax=sot_fit.max())
    cb1 = fig.colorbar(sc_des, ax=ax2, fraction=0.046, pad=0.04)
    cb1.set_label("DES fitness", fontsize=7)
    cb2 = fig.colorbar(sc_sot, ax=ax2, fraction=0.046, pad=0.14)
    cb2.set_label("SOT fitness", fontsize=7)

    # Trendlines
    x_all = np.linspace(min(des_dly.min(), sot_dly.min()),
                        max(des_dly.max(), sot_dly.max()), 100)
    z_des = np.polyfit(des_dly, des_ms, 1)
    z_sot = np.polyfit(sot_dly, sot_ms, 1)
    ax2.plot(x_all, np.polyval(z_des, x_all),
             color=DES_COLOR, linestyle="--", linewidth=1.5,
             label=f"DES trend  slope={z_des[0]:.2f}")
    ax2.plot(x_all, np.polyval(z_sot, x_all),
             color=SOT_COLOR, linestyle="--", linewidth=1.5,
             label=f"SOT trend  slope={z_sot[0]:.2f}")

    ax2.scatter([], [], marker="o", color=DES_COLOR, alpha=0.7, label="DES (GA)")
    ax2.scatter([], [], marker="^", color=SOT_COLOR, alpha=0.7, label="SOT")
    ax2.set_xlabel("Mean per-job delay (h)")
    ax2.set_ylabel("Makespan (h)")
    ax2.set_title("Delay vs makespan  (colour = fitness)")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        save_path = f"compare_throughput_{dist}_{n_jobs}jobs.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Throughput comparison saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Cross-distribution violin comparison
# ---------------------------------------------------------------------------

def plot_comparison_violin(
    des_all:   dict[str, list[dict]],
    sot_all:   dict[str, list[dict]],
    n_jobs:    int,
    save_path: Optional[str] = None,
) -> None:
    """
    Side-by-side violin plots comparing DES vs SOT makespan and oven
    utilization for each delay distribution tested.
    """
    dists  = list(des_all.keys())
    n_dist = len(dists)
    n_rep  = len(next(iter(des_all.values())))

    des_ms   = [np.array([r["makespan_h"] for r in des_all[d]]) for d in dists]
    sot_ms   = [np.array([r["makespan_h"] for r in sot_all[d]]) for d in dists]
    des_util = [_oven_util(des_all[d]) for d in dists]
    sot_util = [_oven_util(sot_all[d]) for d in dists]

    x      = np.arange(n_dist)
    offset = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"DES (GA) vs SOT — distribution comparison  |  "
        f"{n_jobs} jobs  ({n_rep} reps each)",
        fontsize=13, fontweight="bold",
    )

    for ax, des_data, sot_data, ylabel, title in [
        (axes[0], des_ms,   sot_ms,   "Makespan (h)",         "Makespan by distribution"),
        (axes[1], des_util, sot_util, "Oven utilization (%)", "Oven utilization by distribution"),
    ]:
        vp_des = ax.violinplot(des_data, positions=x - offset,
                               widths=0.35, showmedians=True, showextrema=False)
        vp_sot = ax.violinplot(sot_data, positions=x + offset,
                               widths=0.35, showmedians=True, showextrema=False)
        for body in vp_des["bodies"]:
            body.set_facecolor(DES_COLOR); body.set_alpha(0.65)
        for body in vp_sot["bodies"]:
            body.set_facecolor(SOT_COLOR); body.set_alpha(0.65)
        vp_des["cmedians"].set_color("black"); vp_des["cmedians"].set_linewidth(1.5)
        vp_sot["cmedians"].set_color("black"); vp_sot["cmedians"].set_linewidth(1.5)

        ax.scatter(x - offset, [d.mean() for d in des_data],
                   marker="D", s=40, color=DES_COLOR, zorder=5,
                   edgecolors="black", linewidths=0.6, label="DES (GA) mean")
        ax.scatter(x + offset, [d.mean() for d in sot_data],
                   marker="D", s=40, color=SOT_COLOR, zorder=5,
                   edgecolors="black", linewidths=0.6, label="SOT mean")

        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in dists], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        save_path = f"compare_violin_{n_jobs}jobs.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Violin comparison saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Console comparison summary
# ---------------------------------------------------------------------------

def print_comparison(des_results: list[dict], sot_results: list[dict],
                     dist: str) -> None:
    fmt = "  {:<26} {:>9.2f}  {:>9.2f}  {:>+9.2f}"
    print(f"\n{'='*64}")
    print(f"  DES (GA) vs SOT  —  {len(des_results)} reps  dist={dist.upper()}")
    print(f"  {'Metric':<26} {'DES':>9}   {'SOT':>9}   {'Diff':>9}")
    print(f"  {'-'*60}")
    for label, key in [
        ("Makespan (h)",        "makespan_h"),
        ("Fitness",             "fitness"),
        ("Mean job delay (h)",  "mean_delay_h"),
    ]:
        des_v = float(np.mean([r[key] for r in des_results]))
        sot_v = float(np.mean([r[key] for r in sot_results]))
        print(fmt.format(label, des_v, sot_v, des_v - sot_v))
    des_u = float(_oven_util(des_results).mean())
    sot_u = float(_oven_util(sot_results).mean())
    print(fmt.format("Oven utilization (%)", des_u, sot_u, des_u - sot_u))
    print(f"{'='*64}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare DES (GA) vs SOT Monte Carlo simulations"
    )
    parser.add_argument("--workers_b",    type=int, default=N_WORKERS_B)
    parser.add_argument("--workers_c",    type=int, default=N_WORKERS_C)
    parser.add_argument("--jobs",         type=int, default=None,
                        help="Limit jobs loaded (default: all)")
    parser.add_argument("--generations",  type=int, default=GA_GENERATIONS)
    parser.add_argument("--popsize",      type=int, default=GA_POP_SIZE)
    parser.add_argument("--replications", type=int, default=200,
                        help="MC replications per distribution per scheduler (default: 200)")
    parser.add_argument("--dist",         type=str, default="pert",
                        choices=DIST_NAMES,
                        help="Delay distribution (default: pert)")
    parser.add_argument("--all_dists",    action="store_true",
                        help="Run all 5 distributions and produce a violin comparison")
    parser.add_argument("--base_seed",    type=int, default=0)
    args = parser.parse_args()

    pool_B = args.workers_b
    pool_C = args.workers_c

    # Load jobs once — SOT receives the same set converted to its Job type
    jobs_des = load_jobs_des()
    if args.jobs is not None:
        jobs_des = jobs_des[:args.jobs]
    jobs_sot = [
        SotJob(
            job_id      = j.job_id,
            estimated_h = j.estimated_h,
            job_index   = j.job_index,
            is_buffered = j.is_buffered,
            skip_phase1 = j.skip_phase1,
        )
        for j in jobs_des
    ]

    n = len(jobs_des) + MAX_GENERATED_JOBS
    print(f"\nJobs: {len(jobs_des)} loaded + {MAX_GENERATED_JOBS} max generated = {n} total")
    print(f"  B workers: {pool_B}  |  C workers: {pool_C}  |  Replications: {args.replications}")

    # Run GA once — chromosome shared across all DES replications
    print(f"\nRunning GA  (pop={args.popsize}, gen={args.generations}) ...")
    best_chrom, best_fit, history = run_ga(
        jobs_des, pool_B=pool_B, pool_C=pool_C,
        pop_size=args.popsize, n_gen=args.generations,
    )
    print(f"  Best GA fitness: {best_fit:.1f}")
    plot_convergence(history, save_path=f"compare_convergence_{n}jobs.png")

    dists_to_run = DIST_NAMES if args.all_dists else [args.dist]
    des_all: dict[str, list[dict]] = {}
    sot_all: dict[str, list[dict]] = {}

    for dist in dists_to_run:
        print(f"\n{'-'*50}")
        print(f"  Distribution: {dist.upper()}")
        print(f"{'-'*50}")

        des_results = mc_des.run_monte_carlo(
            jobs_des, best_chrom, pool_B, pool_C,
            n_replications=args.replications,
            dist=dist, base_seed=args.base_seed,
        )
        sot_results = mc_sot.run_monte_carlo(
            jobs_sot, pool_B, pool_C,
            n_replications=args.replications,
            dist=dist, base_seed=args.base_seed,
        )

        mc_des.print_summary(des_results, dist)
        mc_sot.print_summary(sot_results, dist)
        print_comparison(des_results, sot_results, dist)

        plot_comparison_summary(des_results, sot_results, dist, n)
        plot_comparison_throughput(des_results, sot_results, dist, n)

        des_all[dist] = des_results
        sot_all[dist] = sot_results

    if args.all_dists and len(des_all) > 1:
        plot_comparison_violin(des_all, sot_all, n)
