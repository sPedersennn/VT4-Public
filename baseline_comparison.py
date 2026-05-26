"""
baseline_comparison.py
======================
Compares des_ga_change_baseline.py (GA+SA+DES) vs des_sot_baseline.py (SOT+DES).

Each replication samples a fresh random set of n_jobs rows from data_real.csv
(1174 rows available).  Both simulators receive the same sampled jobs and the
same per-job PERT delays for that replication.

Replication i uses seed = base_seed + i for both job sampling and delay
sampling, guaranteeing paired stochastic scenarios.  The GA is re-optimised
from scratch on each replication's job sample.

Outputs (saved in working directory)
  baseline_compare_summary_<n>jobs.png    — 4-panel KPI comparison
  baseline_compare_throughput_<n>jobs.png — cumulative throughput & delay scatter
"""
from __future__ import annotations

import argparse
import csv as _csv
import random
from typing import Optional

from scipy.stats import beta as _scipy_beta

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# GA-DES side
import ga_simulationBC_current as _ga_mod
import des_simulation_SA_change as _des_mod
from ga_simulationBC_current import (
    Job,
    _fresh_jobs,
    run_ga,
    N_WORKERS_B,
    N_WORKERS_C,
    SC2_WEIGHT,
    SC3_WEIGHT,
    GA_POP_SIZE,
    GA_GENERATIONS,
    WEEK_HOURS,
    DATA_REAL_CSV,
)
from des_simulation_SA_change import simulate_des_bc, _get_pert_params
from des_ga_change_baseline import N_JOBS

# SOT-DES side
from sot_simulationBC_baseline import Job as SotJob
from des_sot_baseline import simulate_des_sot

from constraintsBC import CHANGEOVER_TIME, MAX_REPAIR_TRACKS

# Disable buffer job generation for all runs in this module
_ga_mod.MAX_GENERATED_JOBS = 0
_des_mod.MAX_GENERATED_JOBS = 0

DES_COLOR = "#5BA4CF"
SOT_COLOR = "#F5A623"
_SOT_CMAP = LinearSegmentedColormap.from_list("sot_scatter", [SOT_COLOR, "#2C7FB8"])

N_JOB_SUBSETS   = 10   # distinct job samples drawn from data_real.csv
N_DELAY_REPS    = 50   # delay replications per job subset
N_JOBS_BASELINE = N_JOBS  # 50 jobs per subset


# ---------------------------------------------------------------------------
# Full-dataset loaders (all rows from data_real.csv)
# ---------------------------------------------------------------------------

def load_all_jobs_baseline(csv_path: str = DATA_REAL_CSV) -> list:
    """Load every row from data_real.csv as GA Job objects (no N_JOBS cap)."""
    jobs = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = _csv.DictReader(f, delimiter=";")
        for i, row in enumerate(reader):
            try:
                estimated = float(row["Estimated"])
            except (KeyError, ValueError):
                continue
            jobs.append(Job(
                job_id      = f"JOB-{i + 1:04d}",
                estimated_h = estimated,
                job_index   = i,
                is_buffered = False,
                skip_phase1 = i < MAX_REPAIR_TRACKS,
            ))
    return jobs


def load_all_sot_jobs(csv_path: str = DATA_REAL_CSV) -> list[SotJob]:
    """Load every row from data_real.csv as SotJob objects (no N_JOBS cap)."""
    jobs = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = _csv.DictReader(f, delimiter=";")
        for i, row in enumerate(reader):
            try:
                estimated = float(row["Estimated"])
            except (KeyError, ValueError):
                continue
            jobs.append(SotJob(
                job_id      = f"JOB-{i + 1:04d}",
                estimated_h = estimated,
                job_index   = i,
                skip_phase1 = i < MAX_REPAIR_TRACKS,
            ))
    return jobs


# ---------------------------------------------------------------------------
# Re-index helpers — reset job_index to 0..n-1 after random sampling.
# The GA chromosome is a permutation of range(n), and the DES delay_applied
# set uses job_index as a unique key.  Both break if job_index values are
# arbitrary CSV row numbers (0..1173) instead of 0..n-1.
# ---------------------------------------------------------------------------

def _reindex_ga_sample(jobs_sample: list) -> list:
    """Return GA Job copies with job_index = position in sample (0..n-1)."""
    return [
        Job(
            job_id      = j.job_id,
            estimated_h = j.estimated_h,
            job_index   = new_idx,
            is_buffered = j.is_buffered,
            skip_phase1 = j.skip_phase1,
        )
        for new_idx, j in enumerate(jobs_sample)
    ]


def _reindex_sot_sample(jobs_sample: list[SotJob]) -> list[SotJob]:
    """Return SotJob copies with job_index = position in sample (0..n-1)."""
    return [
        SotJob(
            job_id      = j.job_id,
            estimated_h = j.estimated_h,
            job_index   = new_idx,
            skip_phase1 = j.skip_phase1,
        )
        for new_idx, j in enumerate(jobs_sample)
    ]


# ---------------------------------------------------------------------------
# Fresh SOT jobs copy (no simulation state, preserves job_index)
# ---------------------------------------------------------------------------

def _fresh_sot(template: list[SotJob]) -> list[SotJob]:
    return [SotJob(
        job_id      = j.job_id,
        estimated_h = j.estimated_h,
        job_index   = j.job_index,
        skip_phase1 = j.skip_phase1,
    ) for j in template]


# ---------------------------------------------------------------------------
# Shared KPI helpers (mirrors compare_mc_analysis.py)
# ---------------------------------------------------------------------------

def _oven_util(results: list[dict]) -> np.ndarray:
    return np.array([
        sum(j.oven_end - j.oven_start + CHANGEOVER_TIME
            for j in r["_jobs"]
            if j.oven_start is not None and j.oven_end is not None)
        / r["makespan_h"] * 100.0
        for r in results
    ])


def _weekly_arrays(results: list[dict]) -> np.ndarray:
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
# Pre-generate delays by job_id so both simulators receive identical
# per-job delays regardless of scheduling order or worker allocation.
# ---------------------------------------------------------------------------

def _pert_sample(rng: random.Random) -> float:
    a, mode, b, gamma = _get_pert_params()
    alpha1 = 1.0 + gamma * (mode - a) / (b - a)
    alpha2 = 1.0 + gamma * (b - mode) / (b - a)
    return a + float(_scipy_beta.ppf(rng.random(), alpha1, alpha2)) * (b - a)


def _build_delay_map(job_ids: list[str], delay_seed: int) -> dict[str, float]:
    """One PERT sample per job, keyed by job_id."""
    rng = random.Random(delay_seed)
    return {jid: _pert_sample(rng) for jid in job_ids}


def _preset_delays(jobs: list, delay_map: dict[str, float]) -> None:
    """
    Tag each job with its pre-assigned PERT delay so the engine returns it
    at the natural late-phase-2 trigger point instead of sampling randomly.
    Also initialises mc_delay=0.0 so jobs that never hit the delay block
    still have the attribute when KPIs are collected.
    """
    for j in jobs:
        j._preset_delay = delay_map.get(j.job_id, 0.0)
        j.mc_delay      = 0.0


# ---------------------------------------------------------------------------
# Single-run DES helpers (GA and SOT)
# Both receive an already-reindexed jobs_sample and a pre-computed chromosome.
# ---------------------------------------------------------------------------

def _run_des_ga(
    jobs_sample: list,
    best_chrom:  list[int],
    pool_B:      int,
    pool_C:      int,
    delay_seed:  int,
) -> dict:
    delay_map = _build_delay_map([j.job_id for j in jobs_sample], delay_seed)
    jobs_copy = _fresh_jobs(jobs_sample)
    _preset_delays(jobs_copy, delay_map)
    _, _, sc2, sc3 = simulate_des_bc(
        jobs_copy, best_chrom, pool_B, pool_C,
        use_sa=True, use_lookahead=True,
        use_delays=True,
        delay_seed=delay_seed,
        gen_seed=delay_seed,
        sa_history_out=None,
    )
    done      = [j for j in jobs_copy if j.oven_end is not None]
    makespan  = max((j.oven_end for j in done), default=float("inf"))
    fitness   = makespan + SC2_WEIGHT * sc2 + SC3_WEIGHT * sc3
    mc_delays = [j.mc_delay for j in done]
    return {
        "makespan_h":   round(makespan, 3),
        "fitness":      round(fitness,  3),
        "sc2":          round(sc2,      3),
        "sc3":          round(sc3,      3),
        "n_completed":  len(done),
        "mean_delay_h": round(float(np.mean(mc_delays)) if mc_delays else 0.0, 4),
        "_jobs":        jobs_copy,
    }


def _run_des_sot(
    jobs_sample: list[SotJob],
    pool_B:      int,
    pool_C:      int,
    delay_seed:  int,
) -> dict:
    delay_map = _build_delay_map([j.job_id for j in jobs_sample], delay_seed)
    jobs_copy = _fresh_sot(jobs_sample)
    _preset_delays(jobs_copy, delay_map)
    _, _, sc2, sc3 = simulate_des_sot(
        jobs_copy, pool_B=pool_B, pool_C=pool_C,
        use_delays=True, delay_seed=delay_seed,
    )
    done      = [j for j in jobs_copy if j.oven_end is not None]
    makespan  = max((j.oven_end for j in done), default=float("inf"))
    fitness   = makespan + SC2_WEIGHT * sc2 + SC3_WEIGHT * sc3
    mc_delays = [j.mc_delay for j in done]
    return {
        "makespan_h":   round(makespan, 3),
        "fitness":      round(fitness,  3),
        "sc2":          round(sc2,      3),
        "sc3":          round(sc3,      3),
        "n_completed":  len(done),
        "mean_delay_h": round(float(np.mean(mc_delays)) if mc_delays else 0.0, 4),
        "_jobs":        jobs_copy,
    }


# ---------------------------------------------------------------------------
# Monte Carlo loops
#   Outer loop: n_subsets distinct job samples
#   Inner loop: n_delay_reps different delay seeds per sample
#   GA runs once per subset; SOT runs n_delay_reps times per subset
# ---------------------------------------------------------------------------

def run_mc_ga(
    all_jobs_ga:  list,
    pool_B:       int,
    pool_C:       int,
    n_jobs:       int,
    n_subsets:    int = N_JOB_SUBSETS,
    n_delay_reps: int = N_DELAY_REPS,
    base_seed:    int = 0,
    pop_size:     int = GA_POP_SIZE,
    n_gen:        int = GA_GENERATIONS,
) -> list[dict]:
    results: list[dict] = []
    total = n_subsets * n_delay_reps
    print(f"\n  GA+SA — {n_subsets} job subsets × {n_delay_reps} delay reps = {total} runs")
    print(f"  ({n_jobs} jobs sampled from {len(all_jobs_ga)}, GA re-optimised per subset)")
    for sub in range(n_subsets):
        job_seed    = base_seed + sub
        jobs_sample = _reindex_ga_sample(
            random.Random(job_seed).sample(all_jobs_ga, n_jobs)
        )
        print(f"    Subset {sub + 1}/{n_subsets}: optimising GA ...", flush=True)
        best_chrom, _, _ = run_ga(
            jobs_sample, pool_B=pool_B, pool_C=pool_C,
            pop_size=pop_size, n_gen=n_gen,
        )
        for rep in range(n_delay_reps):
            delay_seed = base_seed + sub * n_delay_reps + rep
            r = _run_des_ga(jobs_sample, best_chrom, pool_B, pool_C, delay_seed)
            r["subset"]      = sub + 1
            r["replication"] = rep + 1
            results.append(r)
        sub_ms = np.mean([r["makespan_h"] for r in results[-n_delay_reps:]])
        print(f"    Subset {sub + 1} done — mean makespan {sub_ms:.1f} h")
    return results


def run_mc_sot(
    all_jobs_sot: list[SotJob],
    pool_B:       int,
    pool_C:       int,
    n_jobs:       int,
    n_subsets:    int = N_JOB_SUBSETS,
    n_delay_reps: int = N_DELAY_REPS,
    base_seed:    int = 0,
) -> list[dict]:
    results: list[dict] = []
    total = n_subsets * n_delay_reps
    print(f"\n  SOT — {n_subsets} job subsets × {n_delay_reps} delay reps = {total} runs")
    print(f"  ({n_jobs} jobs sampled from {len(all_jobs_sot)})")
    for sub in range(n_subsets):
        job_seed    = base_seed + sub
        jobs_sample = _reindex_sot_sample(
            random.Random(job_seed).sample(all_jobs_sot, n_jobs)
        )
        for rep in range(n_delay_reps):
            delay_seed = base_seed + sub * n_delay_reps + rep
            r = _run_des_sot(jobs_sample, pool_B, pool_C, delay_seed)
            r["subset"]      = sub + 1
            r["replication"] = rep + 1
            results.append(r)
        sub_ms = np.mean([r["makespan_h"] for r in results[-n_delay_reps:]])
        print(f"    Subset {sub + 1}/{n_subsets} done — mean makespan {sub_ms:.1f} h")
    return results


# ---------------------------------------------------------------------------
# Console comparison table
# ---------------------------------------------------------------------------

def print_comparison(des_results: list[dict], sot_results: list[dict],
                     n_jobs: int) -> None:
    fmt    = "  {:<26} {:>9.2f}  {:>9.2f}  {:>+9.2f}"
    n      = len(des_results)
    n_subs = len({r["subset"] for r in des_results})
    n_reps = n // n_subs
    print(f"\n{'='*64}")
    print(f"  DES (GA+SA) vs DES (SOT) — {n_subs} subsets × {n_reps} delay reps"
          f"  |  {n_jobs} jobs/subset")
    print(f"  {'Metric':<26} {'GA+SA':>9}   {'SOT':>9}   {'Diff':>9}")
    print(f"  {'-'*60}")
    for label, key in [
        ("Makespan (h)",       "makespan_h"),
        ("Fitness",            "fitness"),
        ("Mean job delay (h)", "mean_delay_h"),
        ("SC-2 (oven wait h)", "sc2"),
    ]:
        ga_v  = float(np.mean([r[key] for r in des_results]))
        sot_v = float(np.mean([r[key] for r in sot_results]))
        print(fmt.format(label, ga_v, sot_v, ga_v - sot_v))
    ga_u  = float(_oven_util(des_results).mean())
    sot_u = float(_oven_util(sot_results).mean())
    print(fmt.format("Oven utilization (%)", ga_u, sot_u, ga_u - sot_u))
    print(f"{'='*64}")


# ---------------------------------------------------------------------------
# 4-panel summary comparison
# ---------------------------------------------------------------------------

def plot_comparison_summary(
    des_results: list[dict],
    sot_results: list[dict],
    n_jobs:      int,
    save_path:   Optional[str] = None,
) -> None:
    des_ms   = np.array([r["makespan_h"] for r in des_results])
    sot_ms   = np.array([r["makespan_h"] for r in sot_results])
    des_sc2  = np.array([r["sc2"]        for r in des_results])
    sot_sc2  = np.array([r["sc2"]        for r in sot_results])
    des_util = _oven_util(des_results)
    sot_util = _oven_util(sot_results)

    des_pad  = _weekly_arrays(des_results)
    sot_pad  = _weekly_arrays(sot_results)
    max_wks  = max(des_pad.shape[1], sot_pad.shape[1])
    des_pad  = _pad_to(des_pad, max_wks)
    sot_pad  = _pad_to(sot_pad, max_wks)
    x_pos    = np.arange(max_wks)
    wk_lbls  = [f"Wk {w + 1}" for w in range(max_wks)]
    rot      = 45 if max_wks > 8 else 0
    n_rep    = len(des_results)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"DES (GA+SA) vs DES (SOT) — {n_rep} replications each  |  "
        f"{n_jobs} jobs/rep sampled from data_real.csv  |  PERT delays",
        fontsize=13, fontweight="bold",
    )

    def _hist(ax, des_arr, sot_arr, xlabel, title):
        bins = np.histogram_bin_edges(np.concatenate([des_arr, sot_arr]), bins="auto")
        ax.hist(des_arr, bins=bins, color=DES_COLOR, alpha=0.60,
                edgecolor="white", linewidth=0.4, label="DES (GA+SA)")
        ax.hist(sot_arr, bins=bins, color=SOT_COLOR, alpha=0.60,
                edgecolor="white", linewidth=0.4, label="DES (SOT)")
        ax.axvline(des_arr.mean(), color=DES_COLOR, linestyle="--", linewidth=1.8,
                   label=f"GA+SA mean {des_arr.mean():.1f}")
        ax.axvline(sot_arr.mean(), color=SOT_COLOR, linestyle="--", linewidth=1.8,
                   label=f"SOT mean {sot_arr.mean():.1f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    _hist(axes[0, 0], des_ms,   sot_ms,   "Makespan (h)",            "Makespan distribution")
    _hist(axes[0, 1], des_sc2,  sot_sc2,  "SC-2 track blocking (h)", "SC-2 track blocking distribution")
    _hist(axes[1, 0], des_util, sot_util, "Oven utilization (%)",    "Oven utilization distribution")

    des_wk_mean = des_pad.mean(axis=0)
    sot_wk_mean = sot_pad.mean(axis=0)
    des_wk_std  = des_pad.std(axis=0)
    sot_wk_std  = sot_pad.std(axis=0)
    ax = axes[1, 1]
    ax.fill_between(x_pos, des_wk_mean - des_wk_std, des_wk_mean + des_wk_std,
                    alpha=0.20, color=DES_COLOR)
    ax.fill_between(x_pos, sot_wk_mean - sot_wk_std, sot_wk_mean + sot_wk_std,
                    alpha=0.20, color=SOT_COLOR)
    ax.plot(x_pos, des_wk_mean, color=DES_COLOR, linewidth=2.0, marker="o", markersize=4,
            label=f"DES (GA+SA)  avg={des_wk_mean.mean():.1f} bl/wk")
    ax.plot(x_pos, sot_wk_mean, color=SOT_COLOR, linewidth=2.0, marker="s", markersize=4,
            label=f"SOT  avg={sot_wk_mean.mean():.1f} bl/wk")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(wk_lbls, fontsize=8, rotation=rot, ha="right")
    ax.set_xlabel(f"Simulation week  (1 week = {WEEK_HOURS:.0f} h)")
    ax.set_ylabel("Blades completed")
    ax.set_title("Per-week throughput  (mean ± 1σ)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        save_path = f"baseline_compare_summary_{n_jobs}jobs.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Summary comparison saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Cumulative throughput & delay scatter
# ---------------------------------------------------------------------------

def plot_comparison_throughput(
    des_results: list[dict],
    sot_results: list[dict],
    n_jobs:      int,
    save_path:   Optional[str] = None,
) -> None:
    des_pad   = _weekly_arrays(des_results)
    sot_pad   = _weekly_arrays(sot_results)
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
        f"Cumulative throughput & delay — DES (GA+SA) vs DES (SOT)  |  "
        f"{len(des_results)} reps each  |  {n_jobs} jobs/rep",
        fontsize=13, fontweight="bold",
    )

    ax1.fill_between(x_pos, des_mean_c - des_std_c, des_mean_c + des_std_c,
                     alpha=0.18, color=DES_COLOR)
    ax1.fill_between(x_pos, sot_mean_c - sot_std_c, sot_mean_c + sot_std_c,
                     alpha=0.18, color=SOT_COLOR)
    ax1.plot(x_pos, des_mean_c, color=DES_COLOR, linewidth=2.0, marker="o", markersize=4,
             label=f"DES (GA+SA)  P5={des_p5_c[-1]:.0f}  P95={des_p95_c[-1]:.0f}")
    ax1.plot(x_pos, sot_mean_c, color=SOT_COLOR, linewidth=2.0, marker="s", markersize=4,
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

    sc_des = ax2.scatter(des_dly, des_ms, c=des_fit, cmap="RdYlGn_r",
                         alpha=0.65, s=60, edgecolors="none", marker="o",
                         vmin=des_fit.min(), vmax=des_fit.max())
    sc_sot = ax2.scatter(sot_dly, sot_ms, c=sot_fit, cmap=_SOT_CMAP,
                         alpha=0.65, s=60, edgecolors="none", marker="^",
                         vmin=sot_fit.min(), vmax=sot_fit.max())
    cb1 = fig.colorbar(sc_des, ax=ax2, fraction=0.046, pad=0.04)
    cb1.set_label("DES fitness", fontsize=7)
    cb2 = fig.colorbar(sc_sot, ax=ax2, fraction=0.046, pad=0.14)
    cb2.set_label("SOT fitness", fontsize=7)

    all_dly = np.concatenate([des_dly, sot_dly])
    x_line  = np.linspace(all_dly.min(), all_dly.max(), 100)
    if len(des_dly) > 1 and des_dly.std() > 1e-9:
        z_des = np.polyfit(des_dly, des_ms, 1)
        ax2.plot(x_line, np.polyval(z_des, x_line),
                 color=DES_COLOR, linestyle="--", linewidth=1.5,
                 label=f"DES trend  slope={z_des[0]:.2f}")
    if len(sot_dly) > 1 and sot_dly.std() > 1e-9:
        z_sot = np.polyfit(sot_dly, sot_ms, 1)
        ax2.plot(x_line, np.polyval(z_sot, x_line),
                 color=SOT_COLOR, linestyle="--", linewidth=1.5,
                 label=f"SOT trend  slope={z_sot[0]:.2f}")

    ax2.scatter([], [], marker="o", color=DES_COLOR, alpha=0.8, label="DES (GA+SA)")
    ax2.scatter([], [], marker="^", color=SOT_COLOR, alpha=0.8, label="SOT")
    ax2.set_xlabel("Mean per-job delay (h)")
    ax2.set_ylabel("Makespan (h)")
    ax2.set_title("Delay vs makespan  (colour = fitness)")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        save_path = f"baseline_compare_throughput_{n_jobs}jobs.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Throughput comparison saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compare DES (GA+SA) vs DES (SOT) — "
            "n_subsets job samples × n_delay_reps delay scenarios each."
        )
    )
    parser.add_argument("--workers_b",    type=int, default=N_WORKERS_B,
                        help=f"B worker pool (default: {N_WORKERS_B})")
    parser.add_argument("--workers_c",    type=int, default=N_WORKERS_C,
                        help=f"C worker pool (default: {N_WORKERS_C})")
    parser.add_argument("--jobs",         type=int, default=N_JOBS_BASELINE,
                        help=f"Jobs per subset (default: {N_JOBS_BASELINE})")
    parser.add_argument("--subsets",      type=int, default=N_JOB_SUBSETS,
                        help=f"Distinct job subsets (default: {N_JOB_SUBSETS})")
    parser.add_argument("--delay_reps",   type=int, default=N_DELAY_REPS,
                        help=f"Delay replications per subset (default: {N_DELAY_REPS})")
    parser.add_argument("--generations",  type=int, default=GA_GENERATIONS,
                        help=f"GA generations per subset (default: {GA_GENERATIONS})")
    parser.add_argument("--popsize",      type=int, default=GA_POP_SIZE,
                        help=f"GA population size (default: {GA_POP_SIZE})")
    parser.add_argument("--base_seed",    type=int, default=0,
                        help="Base RNG seed (default: 0)")
    args = parser.parse_args()

    pool_B = args.workers_b
    pool_C = args.workers_c
    n_jobs = args.jobs

    # ------------------------------------------------------------------
    # 1. Load full dataset once
    # ------------------------------------------------------------------
    all_jobs_ga  = load_all_jobs_baseline()
    all_jobs_sot = load_all_sot_jobs()

    print(f"\nLoaded {len(all_jobs_ga)} jobs from data_real.csv")
    print(f"  {args.subsets} job subsets × {args.delay_reps} delay reps "
          f"= {args.subsets * args.delay_reps} runs per scheduler")
    print(f"  {n_jobs} jobs per subset  |  B workers: {pool_B}  |  C workers: {pool_C}")
    print(f"  GA: pop={args.popsize}, gen={args.generations}  |  base_seed: {args.base_seed}")

    # ------------------------------------------------------------------
    # 2. Monte Carlo runs
    # ------------------------------------------------------------------
    des_results = run_mc_ga(
        all_jobs_ga, pool_B, pool_C, n_jobs,
        n_subsets=args.subsets,
        n_delay_reps=args.delay_reps,
        base_seed=args.base_seed,
        pop_size=args.popsize,
        n_gen=args.generations,
    )
    sot_results = run_mc_sot(
        all_jobs_sot, pool_B, pool_C, n_jobs,
        n_subsets=args.subsets,
        n_delay_reps=args.delay_reps,
        base_seed=args.base_seed,
    )

    # ------------------------------------------------------------------
    # 3. Console summary
    # ------------------------------------------------------------------
    print_comparison(des_results, sot_results, n_jobs)

    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------
    plot_comparison_summary(des_results, sot_results, n_jobs)
    plot_comparison_throughput(des_results, sot_results, n_jobs)
