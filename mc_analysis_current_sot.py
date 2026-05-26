"""
mc_analysis_current_sot.py
==========================
Monte Carlo simulation: N independent replications of the SOT blade repair
simulation (sot_simulationBC_current.py), comparing five delay distributions
applied to each job's phase-2 duration.

Distributions tested:
  pert        Modified PERT fitted to data_real.csv  (matches base simulation)
  normal      Normal(mu, sigma) fitted to data_real.csv
  uniform     Uniform[a, b] spanning data_real.csv delay range
  triangular  Triangular(a, mode, b) from data_real.csv
  lognormal   Shifted LogNormal fitted to data_real.csv

How it works
  Jobs are loaded from current_status.csv (primary + buffer blades).
  Before each replication a fresh copy of the template jobs is made and a
  per-job phase-2 delay is sampled from the chosen distribution and applied
  directly to job.phase2_h.  simulate_sot_bc() then runs the full SOT
  simulation with the perturbed phase durations.  SC-1/2/3 are returned
  directly by the simulation and require no post-processing.

No GA chromosome is required — worker allocation follows the SOT heuristic
(guarantee 1 worker per active job, then top-up shortest-operation-time first),
so replications differ only in their delay seed.

Outputs (all in the working directory)
  mc_results_sot_<dist>_<n>jobs.csv      per-replication KPIs
  mc_summary_sot_<dist>_<n>jobs.png      4-panel KPI distribution plots
  mc_comparison_sot_<n>jobs.png          violin comparison across all dists
  mc_throughput_sot_<dist>_<n>jobs.png   weekly throughput distribution
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as _scipy_beta

from sot_simulationBC_current import (
    Job,
    load_jobs,
    simulate_sot_bc,
    plot_gantt,
    N_WORKERS_B,
    N_WORKERS_C,
    WEEK_HOURS,
    SC2_WEIGHT,
    SC3_WEIGHT,
    MAX_GENERATED_JOBS,
)
from constraintsBC import CHANGEOVER_TIME

# ---------------------------------------------------------------------------
# Quick-change run settings  ← edit these to control the run
# ---------------------------------------------------------------------------
MC_N_JOBS         = None  # None = all jobs from current_status.csv
MC_N_REPLICATIONS = 1000   # MC replications per distribution
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Distribution parameter fitting (data_real.csv, cached after first read)
# ---------------------------------------------------------------------------

_DIST_PARAMS_CACHE: Optional[dict] = None


def _load_dist_params() -> dict:
    """Read Actual-Estimated delays from data_real.csv and fit all distribution params."""
    csv_path = os.path.join(os.path.dirname(__file__), "data_real.csv")
    delays: list[float] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            try:
                delays.append(float(row["Actual"]) - float(row["Estimated"]))
            except (KeyError, ValueError):
                pass

    arr = np.array(delays, dtype=float)
    a   = float(arr.min())
    b   = float(arr.max())
    counts, edges = np.histogram(arr, bins="auto")
    mode = float((edges[np.argmax(counts)] + edges[np.argmax(counts) + 1]) / 2)
    mu    = float(arr.mean())
    sigma = float(arr.std(ddof=1))

    shift    = -a + 1e-6
    shifted  = arr + shift
    ln_mu    = float(np.log(shifted).mean())
    ln_sigma = float(np.log(shifted).std(ddof=1))

    return {
        "n":    len(delays),
        "a":    a,   "b":    b,   "mode": mode, "gamma": 60.0,
        "mu":   mu,  "sigma": sigma,
        "ln_shift": shift, "ln_mu": ln_mu, "ln_sigma": ln_sigma,
    }


def _get_dist_params() -> dict:
    global _DIST_PARAMS_CACHE
    if _DIST_PARAMS_CACHE is None:
        _DIST_PARAMS_CACHE = _load_dist_params()
    return _DIST_PARAMS_CACHE


# ---------------------------------------------------------------------------
# Delay samplers — all share signature  (rng: random.Random) -> float
# ---------------------------------------------------------------------------

def _sampler_pert(rng: random.Random) -> float:
    p = _get_dist_params()
    a, mode, b, gamma = p["a"], p["mode"], p["b"], p["gamma"]
    alpha1 = 1.0 + gamma * (mode - a) / (b - a)
    alpha2 = 1.0 + gamma * (b - mode) / (b - a)
    z = float(_scipy_beta.ppf(rng.random(), alpha1, alpha2))
    return a + z * (b - a)


def _sampler_normal(rng: random.Random) -> float:
    p = _get_dist_params()
    return rng.gauss(p["mu"], p["sigma"])


def _sampler_uniform(rng: random.Random) -> float:
    p = _get_dist_params()
    return rng.uniform(p["a"], p["b"])


def _sampler_triangular(rng: random.Random) -> float:
    p = _get_dist_params()
    return rng.triangular(p["a"], p["b"], p["mode"])


def _sampler_lognormal(rng: random.Random) -> float:
    p = _get_dist_params()
    return rng.lognormvariate(p["ln_mu"], p["ln_sigma"]) - p["ln_shift"]


DISTRIBUTIONS: dict[str, Callable[[random.Random], float]] = {
    "pert":       _sampler_pert,
    "normal":     _sampler_normal,
    "uniform":    _sampler_uniform,
    "triangular": _sampler_triangular,
    "lognormal":  _sampler_lognormal,
}
DIST_NAMES = list(DISTRIBUTIONS.keys())


# ---------------------------------------------------------------------------
# Fresh-jobs helper
# ---------------------------------------------------------------------------

def _fresh_jobs_sot(template: list[Job]) -> list[Job]:
    """Return a list of clean Job copies ready for a new simulation run."""
    return [Job(
        job_id      = j.job_id,
        estimated_h = j.estimated_h,
        job_index   = j.job_index,
        is_buffered = j.is_buffered,
        skip_phase1 = j.skip_phase1,
    ) for j in template]


# ---------------------------------------------------------------------------
# Per-replication runner
# ---------------------------------------------------------------------------

def run_one(
    jobs_template: list[Job],
    pool_B:        int,
    pool_C:        int,
    delay_seed:    int,
    dist:          str,
) -> dict:
    """Run a single SOT replication and return a KPI dict."""
    jobs_copy = _fresh_jobs_sot(jobs_template)
    for j in jobs_copy:
        j.mc_delay = 0.0

    sampler   = DISTRIBUTIONS[dist]
    staff_log, sc1, sc2, sc3 = simulate_sot_bc(
        jobs_copy,
        pool_B        = pool_B,
        pool_C        = pool_C,
        gen_seed      = delay_seed,
        delay_sampler = sampler,
        delay_seed    = delay_seed,
    )
    done      = [j for j in jobs_copy if j.oven_end is not None]
    mc_delays = [j.mc_delay for j in done]
    makespan = max((j.oven_end for j in done), default=float("inf"))

    fitness = makespan + SC2_WEIGHT * sc2 + SC3_WEIGHT * sc3

    return {
        "dist":          dist,
        "makespan_h":    round(makespan, 3),
        "fitness":       round(fitness, 3),
        "sc1":           round(sc1, 3),
        "sc2":           round(sc2, 3),
        "sc3":           round(sc3, 3),
        "n_completed":   len(done),
        "mean_delay_h":  round(float(np.mean(mc_delays))  if mc_delays else 0.0, 4),
        "total_delay_h": round(float(np.sum(mc_delays))   if mc_delays else 0.0, 4),
        "n_delayed":     len(mc_delays),
        "_jobs":      jobs_copy,
        "_staff_log": staff_log,
        "_pool_B":    pool_B,
        "_pool_C":    pool_C,
    }


# ---------------------------------------------------------------------------
# Monte Carlo loop
# ---------------------------------------------------------------------------

def run_monte_carlo(
    jobs_template:  list[Job],
    pool_B:         int,
    pool_C:         int,
    n_replications: int  = 200,
    dist:           str  = "pert",
    base_seed:      int  = 0,
    verbose:        bool = True,
) -> list[dict]:
    """
    Run n_replications independent SOT simulations with delay distribution dist.

    Each replication i uses delay_seed = base_seed + i so the delay RNG is
    unique per replication while the SOT heuristic is deterministic given the
    same jobs and pool sizes.
    """
    results: list[dict] = []

    if verbose:
        print(f"\n  Running {n_replications} replications  [dist={dist.upper()}] ...")

    for rep in range(n_replications):
        r = run_one(jobs_template, pool_B, pool_C,
                    delay_seed=base_seed + rep, dist=dist)
        r["replication"] = rep + 1
        results.append(r)

        if verbose and (rep + 1) % 25 == 0:
            ms_arr = np.array([x["makespan_h"] for x in results])
            print(f"    rep {rep+1:>3}/{n_replications}  "
                  f"last={r['makespan_h']:.1f}h  "
                  f"mean={ms_arr.mean():.1f}h  std={ms_arr.std():.1f}h")

    return results


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict], dist: str) -> None:
    fmt = "  {:<24} {:>9.2f}  ±{:>7.2f}  [{:>7.1f}, {:>7.1f}]"
    print(f"\n{'='*60}")
    print(f"  Monte Carlo summary (SOT) — {len(results)} reps  dist={dist.upper()}")
    print(f"  {'Metric':<24} {'Mean':>9}   {'Std':>7}   {'[Min':>7}, {'Max]':>7}")
    print(f"  {'-'*56}")
    for label, key in [
        ("Makespan (h)",          "makespan_h"),
        ("Fitness",               "fitness"),
        ("SC-1 (worker band)",    "sc1"),
        ("SC-2 (track blocking)", "sc2"),
        ("SC-3 (oven idle)",      "sc3"),
        ("Mean job delay (h)",    "mean_delay_h"),
    ]:
        vals = np.array([r[key] for r in results])
        print(fmt.format(label, vals.mean(), vals.std(), vals.min(), vals.max()))
    ms = np.array([r["makespan_h"] for r in results])
    print(f"\n  Makespan percentiles:  "
          f"P5={np.percentile(ms,5):.1f}h  "
          f"P50={np.percentile(ms,50):.1f}h  "
          f"P95={np.percentile(ms,95):.1f}h")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(results: list[dict], n_jobs: int, dist: str) -> str:
    path   = f"mc_results_sot_{dist}_{n_jobs}jobs.csv"
    fields = [k for k in results[0] if not k.startswith("_")]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fields})
    print(f"  CSV saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# Single-distribution summary plot
# ---------------------------------------------------------------------------

def plot_summary(
    results:   list[dict],
    dist:      str,
    n_jobs:    int,
    save_path: Optional[str] = None,
) -> None:
    """4-panel figure: makespan hist, SC2 hist, oven utilization hist, delay/makespan scatter."""
    ms  = np.array([r["makespan_h"]   for r in results])
    fit = np.array([r["fitness"]      for r in results])
    sc2 = np.array([r["sc2"]          for r in results])
    dly = np.array([r["mean_delay_h"] for r in results])

    oven_util = np.array([
        sum(j.oven_end - j.oven_start + CHANGEOVER_TIME
            for j in r["_jobs"]
            if j.oven_start is not None and j.oven_end is not None)
        / r["makespan_h"] * 100.0
        for r in results
    ])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Monte Carlo SOT — {len(results)} replications  |  "
        f"Delay: {dist.upper()}  |  {n_jobs} jobs",
        fontsize=13, fontweight="bold",
    )

    # ---- Makespan histogram ----
    ax = axes[0, 0]
    ax.hist(ms, bins=30, color="#5BA4CF", edgecolor="white", linewidth=0.5)
    ax.axvline(ms.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean {ms.mean():.1f} h")
    ax.axvline(np.percentile(ms, 5),  color="orange", linestyle=":", linewidth=1.2,
               label=f"P5={np.percentile(ms,5):.1f} h")
    ax.axvline(np.percentile(ms, 95), color="orange", linestyle=":", linewidth=1.2,
               label=f"P95={np.percentile(ms,95):.1f} h")
    ax.set_xlabel("Makespan (h)")
    ax.set_ylabel("Frequency")
    ax.set_title("Makespan distribution")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ---- SC-2 histogram ----
    ax = axes[0, 1]
    ax.hist(sc2, bins=30, color="#F5A623", edgecolor="white", linewidth=0.5)
    ax.axvline(sc2.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean {sc2.mean():.1f} h")
    ax.axvline(np.percentile(sc2, 5),  color="orange", linestyle=":", linewidth=1.2,
               label=f"P5={np.percentile(sc2,5):.1f} h")
    ax.axvline(np.percentile(sc2, 95), color="orange", linestyle=":", linewidth=1.2,
               label=f"P95={np.percentile(sc2,95):.1f} h")
    ax.set_xlabel("SC-2 track blocking (h)")
    ax.set_ylabel("Frequency")
    ax.set_title("SC-2 (track blocking) distribution")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ---- Oven utilization histogram ----
    ax = axes[1, 0]
    ax.hist(oven_util, bins=30, color="#7EC850", edgecolor="white", linewidth=0.5)
    ax.axvline(oven_util.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean {oven_util.mean():.1f} %")
    ax.axvline(np.percentile(oven_util, 5),  color="orange", linestyle=":", linewidth=1.2,
               label=f"P5={np.percentile(oven_util,5):.1f} %")
    ax.axvline(np.percentile(oven_util, 95), color="orange", linestyle=":", linewidth=1.2,
               label=f"P95={np.percentile(oven_util,95):.1f} %")
    ax.set_xlabel("Oven utilization (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Oven utilization distribution")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ---- Mean delay vs makespan scatter, coloured by fitness ----
    ax = axes[1, 1]
    sc_plot = ax.scatter(dly, ms, c=fit, cmap="RdYlGn_r",
                         alpha=0.55, s=20, edgecolors="none")
    plt.colorbar(sc_plot, ax=ax, label="Fitness score")
    z = np.polyfit(dly, ms, 1)
    x_line = np.linspace(dly.min(), dly.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line),
            color="navy", linestyle="--", linewidth=1.2,
            label=f"Trend (slope {z[0]:.2f})")
    ax.set_xlabel("Mean per-job delay (h)")
    ax.set_ylabel("Makespan (h)")
    ax.set_title("Delay vs makespan  (colour = fitness score)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        save_path = f"mc_summary_sot_{dist}_{n_jobs}jobs.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Summary plot saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Cross-distribution comparison plot
# ---------------------------------------------------------------------------

def plot_comparison(
    all_results: dict[str, list[dict]],
    n_jobs:      int,
    save_path:   Optional[str] = None,
) -> None:
    """Violin + mean-marker comparison of makespan and SC-2 across distributions."""
    dists     = list(all_results.keys())
    makespans = [np.array([r["makespan_h"] for r in all_results[d]]) for d in dists]
    sc2s      = [np.array([r["sc2"]        for r in all_results[d]]) for d in dists]
    n_reps    = len(next(iter(all_results.values())))

    colors = ["#5BA4CF", "#F5A623", "#7EC850", "#E05C2A", "#9B59B6"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Delay-distribution comparison (SOT) — {n_jobs} jobs  ({n_reps} reps each)",
        fontsize=13, fontweight="bold",
    )

    for ax, datasets, ylabel, title in [
        (axes[0], makespans, "Makespan (h)",            "Makespan by distribution"),
        (axes[1], sc2s,      "SC-2 track blocking (h)", "SC-2 by distribution"),
    ]:
        vp = ax.violinplot(datasets, positions=range(len(dists)),
                           showmedians=True, showextrema=True)
        for body, col in zip(vp["bodies"], colors):
            body.set_facecolor(col)
            body.set_alpha(0.60)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(2)

        means = [float(d.mean()) for d in datasets]
        ax.scatter(range(len(dists)), means,
                   marker="D", s=45, color="red", zorder=5, label="Mean")

        for i, m in enumerate(means):
            ax.text(i, m + 0.01 * (max(means) - min(means) + 1e-6),
                    f"{m:.0f}", ha="center", va="bottom", fontsize=8, color="red")

        ax.set_xticks(range(len(dists)))
        ax.set_xticklabels([d.upper() for d in dists], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        save_path = f"mc_comparison_sot_{n_jobs}jobs.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Comparison plot saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Weekly throughput distribution plot
# ---------------------------------------------------------------------------

def plot_weekly_throughput_distribution(
    results:   list[dict],
    dist:      str,
    n_jobs:    int,
    save_path: Optional[str] = None,
) -> None:
    """
    Two-panel figure showing the distribution of weekly throughput across all
    MC replications.

    Left panel  — Box plots per week: each box summarises how many blades
                  completed (oven_end) in that simulation week across every
                  replication.  A red dashed mean line and an orange P5–P95
                  shaded band are overlaid.

    Right panel — Cumulative throughput: the mean cumulative blades completed
                  per week with a ±1σ shaded band, and P5/P95 boundary lines.
    """
    all_counts: list[np.ndarray] = []
    for r in results:
        jobs = r["_jobs"]
        done = [j for j in jobs if j.oven_end is not None]
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

    max_weeks = max(len(c) for c in all_counts)
    padded = np.zeros((len(all_counts), max_weeks), dtype=float)
    for i, c in enumerate(all_counts):
        padded[i, : len(c)] = c

    week_labels = [f"Wk {w + 1}" for w in range(max_weeks)]
    x_pos       = np.arange(max_weeks)

    mean_wk = padded.mean(axis=0)
    p5_wk   = np.percentile(padded, 5,  axis=0)
    p95_wk  = np.percentile(padded, 95, axis=0)

    cumulative = padded.cumsum(axis=1)
    mean_cum   = cumulative.mean(axis=0)
    std_cum    = cumulative.std(axis=0)
    p5_cum     = np.percentile(cumulative, 5,  axis=0)
    p95_cum    = np.percentile(cumulative, 95, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Weekly throughput distribution (SOT MC) — {len(results)} replications  |  "
        f"Delay: {dist.upper()}  |  {n_jobs} jobs",
        fontsize=13, fontweight="bold",
    )

    bp = ax1.boxplot(
        [padded[:, w] for w in range(max_weeks)],
        positions=x_pos,
        widths=0.55,
        patch_artist=True,
        boxprops=dict(facecolor="#D4E6F1", color="#2C7FB8"),
        medianprops=dict(color="red", linewidth=2),
        whiskerprops=dict(color="#2C7FB8"),
        capprops=dict(color="#2C7FB8"),
        flierprops=dict(marker="o", markersize=2.5, alpha=0.35, color="#555555"),
        manage_ticks=False,
    )
    _ = bp

    ax1.fill_between(x_pos, p5_wk, p95_wk,
                     alpha=0.18, color="orange", label="P5–P95 band")
    ax1.plot(x_pos, mean_wk, color="red", linestyle="--", linewidth=1.8,
             marker="D", markersize=4, label=f"Mean ({mean_wk.mean():.1f} blades/wk)")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(week_labels, fontsize=8,
                        rotation=45 if max_weeks > 8 else 0, ha="right")
    ax1.set_xlabel("Simulation week  (1 week = 144 h)")
    ax1.set_ylabel("Blades completed")
    ax1.set_title("Per-week throughput distribution")
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)

    ax2.fill_between(x_pos, mean_cum - std_cum, mean_cum + std_cum,
                     alpha=0.25, color="#5BA4CF", label="Mean ± 1σ")
    ax2.fill_between(x_pos, p5_cum, p95_cum,
                     alpha=0.13, color="orange", label="P5–P95 band")
    ax2.plot(x_pos, mean_cum, color="#1A5C8C", linewidth=2.0,
             marker="o", markersize=4, label="Mean cumulative")
    ax2.plot(x_pos, p5_cum,  color="orange", linestyle=":", linewidth=1.2,
             label=f"P5={p5_cum[-1]:.0f} blades")
    ax2.plot(x_pos, p95_cum, color="orange", linestyle=":", linewidth=1.2,
             label=f"P95={p95_cum[-1]:.0f} blades")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(week_labels, fontsize=8,
                        rotation=45 if max_weeks > 8 else 0, ha="right")
    ax2.set_xlabel("Simulation week  (1 week = 144 h)")
    ax2.set_ylabel("Cumulative blades completed")
    ax2.set_title("Cumulative throughput distribution")
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        save_path = f"mc_throughput_sot_{dist}_{n_jobs}jobs.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Weekly throughput distribution saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monte Carlo SOT — replications with pluggable delay distributions"
    )
    parser.add_argument("--workers_b",    type=int, default=N_WORKERS_B)
    parser.add_argument("--workers_c",    type=int, default=N_WORKERS_C)
    parser.add_argument("--jobs",         type=int, default=MC_N_JOBS,
                        help="Limit jobs loaded from current_status.csv (default: all)")
    parser.add_argument("--replications", type=int, default=MC_N_REPLICATIONS,
                        help=f"MC replications per distribution (default: {MC_N_REPLICATIONS})")
    parser.add_argument("--dist",         type=str, default="pert",
                        choices=DIST_NAMES,
                        help="Delay distribution (default: pert)")
    parser.add_argument("--all_dists",    action="store_true",
                        help="Run all 5 distributions and produce a comparison plot")
    parser.add_argument("--base_seed",    type=int, default=0,
                        help="Base seed; replication i uses base_seed + i")
    parser.add_argument("--gantt",        action="store_true",
                        help="Plot Gantt from the final replication")
    args = parser.parse_args()

    pool_B = args.workers_b
    pool_C = args.workers_c

    jobs_template = load_jobs()
    if args.jobs is not None:
        jobs_template = jobs_template[:args.jobs]
    n = len(jobs_template) + MAX_GENERATED_JOBS

    n_primary = sum(1 for j in jobs_template if not j.is_buffered)
    n_buf     = sum(1 for j in jobs_template if j.is_buffered)
    ests      = [j.estimated_h for j in jobs_template]

    print(f"\nLoaded {n} jobs from current_status.csv")
    print(f"  Primary (buffer=0): {n_primary}  |  Buffer (buffer=1): {n_buf}")
    print(f"  B workers: {pool_B}  |  C workers: {pool_C}")
    print(f"  Remaining hours: min={min(ests):.1f}h  max={max(ests):.1f}h  avg={sum(ests)/n:.1f}h")

    p = _get_dist_params()
    print(f"\nDelay params fitted from data_real.csv (n={p['n']} observations)")
    print(f"  PERT / Triangular : a={p['a']:.2f}  mode={p['mode']:.2f}  b={p['b']:.2f}")
    print(f"  Normal            : mu={p['mu']:.2f}  sigma={p['sigma']:.2f}")
    print(f"  Uniform           : [{p['a']:.2f}, {p['b']:.2f}]")
    print(f"  Lognormal shift   : {p['ln_shift']:.4f}  ln_mu={p['ln_mu']:.4f}  "
          f"ln_sigma={p['ln_sigma']:.4f}")

    print(f"\nNo GA needed — SOT heuristic is deterministic given pool sizes.")
    print(f"  B pool: {pool_B}  |  C pool: {pool_C}  |  Initial jobs: {n}")

    dists_to_run = DIST_NAMES if args.all_dists else [args.dist]
    all_results: dict[str, list[dict]] = {}

    for dist in dists_to_run:
        results = run_monte_carlo(
            jobs_template, pool_B, pool_C,
            n_replications=args.replications,
            dist=dist,
            base_seed=args.base_seed,
        )
        print_summary(results, dist)
        export_csv(results, n, dist)
        plot_summary(results, dist, n)
        plot_weekly_throughput_distribution(results, dist, n)
        all_results[dist] = results

    if args.all_dists and len(all_results) > 1:
        plot_comparison(all_results, n)

    if args.gantt:
        last      = all_results[args.dist][-1]
        jobs_last = last["_jobs"]
        staff_log = last["_staff_log"]
        plot_gantt(jobs_last, staff_log,
                   pool_B=pool_B, pool_C=pool_C,
                   save_path=f"gantt_sot_mc_{n}jobs.png")
