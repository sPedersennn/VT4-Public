"""
robust_analysis_SA.py
=====================
Sample Average Approximation (SAA) + robustness analysis of the SA blade-repair
DES (des_simulation_SA_change.py).

SAA procedure
-------------
1. Run the GA once to obtain the best chromosome.
2. Repeat M times ("batches"):
     - Draw N independent replications, each with a unique delay seed.
     - Compute per-batch averages of all KPIs.
3. E[KPI] ≈ grand mean of batch averages; 95% CI via t-distribution.

Robustness metrics  (computed over all M*N individual replications)
-------------------------------------------------------------------
Two operational metrics drive robustness — not fitness:

  Oven utilization (higher is better)
    CVaR_5_low  = mean of the LOWEST 5 % of replication values
                  (worst-case utilization scenarios)
    Util robustness score = CVaR_5_low / E[util]
      → 1.0 = tail as good as mean (robust)
      → < 0.85 = fragile (tail much worse than mean)

  Track blocking / SC-2 (lower is better)
    CVaR_95_high = mean of the HIGHEST 5 % of replication values
                   (worst-case blocking scenarios)
    Blocking robustness score = CVaR_95_high / E[sc2]
      → 1.0 = tail as bad as mean (robust)
      → > 1.15 = fragile (tail much worse than mean)

Stress testing
--------------
  Re-run the SAA with delay draws scaled by factors s ∈ {1.0, 1.5, 2.0, 2.5}.
  Tracks how E[util] and CVaR_5_low(util) / E[sc2] and CVaR_95(sc2) degrade.

Outputs
  robust_sa_batches_<dist>_<n>jobs.csv       per-batch KPI averages
  robust_sa_reps_<dist>_<n>jobs.csv          all individual replication KPIs
  robust_sa_convergence_<dist>_<n>jobs.png   E[util] + E[sc2] convergence
  robust_sa_tail_<dist>_<n>jobs.png          tail risk / CVaR visualisation
  robust_sa_stress_<dist>_<n>jobs.png        robustness metrics vs delay scale
  robust_sa_dashboard_<dist>_<n>jobs.png     4-panel robustness dashboard
  robust_sa_ev_comparison_<n>jobs.png        E[util] & E[sc2] across distributions
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats as _scipy_stats

from constraintsBC import CHANGEOVER_TIME
from mc_analysis_SA import (
    DIST_NAMES,
    DISTRIBUTIONS,
    _get_dist_params,
    run_one,
)
from ga_simulationBC_current import (
    _fresh_jobs,
    load_jobs,
    run_ga,
    N_WORKERS_B,
    N_WORKERS_C,
    SC2_WEIGHT,
    SC3_WEIGHT,
    GA_POP_SIZE,
    GA_GENERATIONS,
    MAX_GENERATED_JOBS,
)
from des_simulation_SA_change import simulate_des_bc, plot_gantt_des
import des_simulation_SA_change as _des_mod
import ga_simulationBC_current as _ga_mod


# ---------------------------------------------------------------------------
# Oven utilization helper
# ---------------------------------------------------------------------------

def _oven_util_pct(jobs: list, makespan_h: float) -> float:
    """Oven utilization as a percentage of makespan."""
    oven_h = sum(
        j.oven_end - j.oven_start + CHANGEOVER_TIME
        for j in jobs
        if j.oven_start is not None and j.oven_end is not None
    )
    return (oven_h / makespan_h * 100.0) if makespan_h > 0 else 0.0


# ---------------------------------------------------------------------------
# Stress-scaled sampler patching
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _patch_scaled_sampler(dist: str, scale: float):
    """Patch _sample_pert_delay with a version whose output is multiplied by scale."""
    base     = DISTRIBUTIONS[dist]
    original = _des_mod._sample_pert_delay
    _des_mod._sample_pert_delay = lambda rng: base(rng) * scale
    try:
        yield
    finally:
        _des_mod._sample_pert_delay = original


def run_one_stress(
    jobs_template: list,
    chromosome:    list[int],
    pool_B:        int,
    pool_C:        int,
    delay_seed:    int,
    dist:          str,
    scale:         float,
) -> dict:
    """Single replication with delays scaled by `scale`."""
    jobs_copy = _fresh_jobs(jobs_template)
    with _patch_scaled_sampler(dist, scale):
        _, _, sc2, sc3 = simulate_des_bc(
            jobs_copy, chromosome, pool_B, pool_C,
            use_sa=True, use_lookahead=True,
            use_delays=True,
            delay_seed=delay_seed,
            gen_seed=delay_seed,
        )
    done     = [j for j in jobs_copy if j.oven_end is not None]
    makespan = max((j.oven_end for j in done), default=float("inf"))
    fitness  = makespan + SC2_WEIGHT * sc2 + SC3_WEIGHT * sc3
    mc_delays = [d for j in done if (d := getattr(j, "mc_delay", None)) is not None]
    return {
        "makespan_h":    round(makespan, 3),
        "fitness":       round(fitness, 3),
        "sc2":           round(sc2, 3),
        "sc3":           round(sc3, 3),
        "mean_delay_h":  round(float(np.mean(mc_delays)) if mc_delays else 0.0, 4),
        "oven_util_pct": round(_oven_util_pct(jobs_copy, makespan), 4),
    }


# ---------------------------------------------------------------------------
# SAA batch runner  (returns batch avg + raw reps)
# ---------------------------------------------------------------------------

def run_saa_batch(
    jobs_template: list,
    chromosome:    list[int],
    pool_B:        int,
    pool_C:        int,
    batch_idx:     int,
    batch_size:    int,
    dist:          str,
    base_seed:     int   = 0,
    scale:         float = 1.0,
) -> tuple[dict, list[dict]]:
    """
    Run one SAA batch of `batch_size` replications.
    Seed scheme: base_seed + batch_idx * batch_size + rep  (unique across all batches).
    Returns (batch_avg_dict, list_of_rep_dicts).
    """
    seed_offset = base_seed + batch_idx * batch_size
    reps: list[dict] = []

    for rep in range(batch_size):
        if scale == 1.0:
            r = run_one(
                jobs_template, chromosome, pool_B, pool_C,
                delay_seed=seed_offset + rep, dist=dist,
            )
            # compute oven util from stored jobs
            ms = r["makespan_h"]
            r["oven_util_pct"] = round(_oven_util_pct(r["_jobs"], ms), 4)
        else:
            r = run_one_stress(
                jobs_template, chromosome, pool_B, pool_C,
                delay_seed=seed_offset + rep, dist=dist, scale=scale,
            )
        r["batch"]       = batch_idx + 1
        r["replication"] = rep + 1
        r["scale"]       = scale
        reps.append(r)

    kpi_keys = ["makespan_h", "fitness", "sc2", "sc3", "mean_delay_h", "oven_util_pct"]
    batch_avg = {k: float(np.mean([r[k] for r in reps])) for k in kpi_keys}
    batch_avg["batch"]      = batch_idx + 1
    batch_avg["batch_size"] = batch_size
    batch_avg["dist"]       = dist
    batch_avg["scale"]      = scale
    return batch_avg, reps


# ---------------------------------------------------------------------------
# Full SAA run
# ---------------------------------------------------------------------------

def run_saa(
    jobs_template:     list,
    chromosome:        list[int],
    pool_B:            int,
    pool_C:            int,
    n_batches:         int          = 20,
    batch_size:        int          = 50,
    dist:              str          = "pert",
    base_seed:         int          = 0,
    scale:             float        = 1.0,
    verbose:           bool         = True,
    batch_templates:   Optional[list] = None,
    batch_chromosomes: Optional[list] = None,
) -> tuple[list[dict], list[dict]]:
    """
    Run `n_batches` SAA batches.
    Returns (batch_results, all_reps) where all_reps is the flat list of every
    individual replication result across all batches.

    If `batch_templates` / `batch_chromosomes` are provided (one entry per CSV
    file), batch index b maps to pair_idx = b // 2, so each file and its
    GA-optimised chromosome cover exactly two consecutive batches.
    """
    batch_results: list[dict] = []
    all_reps:      list[dict] = []
    label = f"dist={dist.upper()}  scale={scale:.1f}x"

    if verbose:
        print(f"\n  SAA: {n_batches} batches x {batch_size} reps = "
              f"{n_batches * batch_size} total  [{label}]")
        if batch_templates is not None:
            print(f"       using {len(batch_templates)} templates/chromosomes "
                  f"(1 GA run per file, 2 batches each)")

    for b in range(n_batches):
        pair_idx = min(b // 2, (len(batch_templates) - 1) if batch_templates else 0)
        tmpl  = batch_templates[pair_idx]   if batch_templates   is not None else jobs_template
        chrom = batch_chromosomes[pair_idx] if batch_chromosomes is not None else chromosome
        bavg, reps = run_saa_batch(
            tmpl, chrom, pool_B, pool_C,
            batch_idx=b, batch_size=batch_size,
            dist=dist, base_seed=base_seed, scale=scale,
        )
        batch_results.append(bavg)
        all_reps.extend(reps)

        if verbose:
            util_arr = np.array([x["oven_util_pct"] for x in batch_results])
            sc2_arr  = np.array([x["sc2"]           for x in batch_results])
            print(f"    batch {b+1:>3}/{n_batches}  "
                  f"util={bavg['oven_util_pct']:.1f}%  "
                  f"E[util]={util_arr.mean():.1f}%  "
                  f"sc2={bavg['sc2']:.1f}h  "
                  f"E[sc2]={sc2_arr.mean():.1f}h")

    return batch_results, all_reps


# ---------------------------------------------------------------------------
# Expected value + CI  (from batch averages)
# ---------------------------------------------------------------------------

def compute_expected_value(
    batch_results: list[dict],
    kpi:           str   = "oven_util_pct",
    confidence:    float = 0.95,
) -> dict:
    z      = np.array([b[kpi] for b in batch_results])
    m      = len(z)
    mean   = float(z.mean())
    std    = float(z.std(ddof=1))
    se     = std / np.sqrt(m)
    t_crit = float(_scipy_stats.t.ppf((1 + confidence) / 2, df=m - 1))
    return {
        "kpi": kpi, "n_batches": m,
        "E_f": mean, "std": std, "se": se,
        "ci_lo": mean - t_crit * se,
        "ci_hi": mean + t_crit * se,
        "ci_width": 2 * t_crit * se,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# CVaR — supports upper tail (bad = high) and lower tail (bad = low)
# ---------------------------------------------------------------------------

def compute_cvar(
    all_reps:  list[dict],
    kpi:       str   = "sc2",
    alpha:     float = 0.95,
    tail:      str   = "upper",
) -> dict:
    """
    tail="upper"  →  CVaR of the worst (1-alpha) HIGH values  (bad = high, e.g. sc2)
    tail="lower"  →  CVaR of the worst (1-alpha) LOW  values  (bad = low,  e.g. oven_util_pct)

    Robustness score:
      upper tail:  CVaR / E[f]   — close to 1 = robust, > 1.15 = fragile
      lower tail:  CVaR / E[f]   — close to 1 = robust, < 0.85 = fragile
    """
    z  = np.array([r[kpi] for r in all_reps])
    ef = float(z.mean())

    if tail == "upper":
        cutoff = float(np.percentile(z, alpha * 100))
        tail_z = z[z >= cutoff]
    else:
        cutoff = float(np.percentile(z, (1 - alpha) * 100))
        tail_z = z[z <= cutoff]

    cvar = float(tail_z.mean())
    return {
        "kpi":              kpi,
        "tail":             tail,
        "alpha":            alpha,
        "n_total":          len(z),
        "n_tail":           len(tail_z),
        "VaR":              cutoff,
        "CVaR":             cvar,
        "E_f":              ef,
        "robustness_score": cvar / ef if ef != 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Tail risk percentile table
# ---------------------------------------------------------------------------

def compute_tail_risk(all_reps: list[dict], kpi: str) -> dict:
    z = np.array([r[kpi] for r in all_reps])
    return {
        "kpi":  kpi,
        "n":    len(z),
        "mean": float(z.mean()),
        "std":  float(z.std(ddof=1)),
        "min":  float(z.min()),
        "P5":   float(np.percentile(z, 5)),
        "P25":  float(np.percentile(z, 25)),
        "P50":  float(np.percentile(z, 50)),
        "P75":  float(np.percentile(z, 75)),
        "P95":  float(np.percentile(z, 95)),
        "P99":  float(np.percentile(z, 99)),
        "max":  float(z.max()),
    }


# ---------------------------------------------------------------------------
# Robustness score helpers
# ---------------------------------------------------------------------------

def _util_robustness(all_reps: list[dict]) -> dict:
    """Oven utilization robustness: CVaR_5_low / E[util]. Fragile if < 0.85."""
    return compute_cvar(all_reps, kpi="oven_util_pct", alpha=0.95, tail="lower")


def _blocking_robustness(all_reps: list[dict]) -> dict:
    """Track blocking robustness: CVaR_95_high / E[sc2]. Fragile if > 1.15."""
    return compute_cvar(all_reps, kpi="sc2", alpha=0.95, tail="upper")


def _makespan_robustness(all_reps: list[dict]) -> dict:
    """Makespan robustness: CVaR_95_high / E[makespan]. Fragile if > 1.15."""
    return compute_cvar(all_reps, kpi="makespan_h", alpha=0.95, tail="upper")


# ---------------------------------------------------------------------------
# Stress test
# ---------------------------------------------------------------------------

def run_stress_test(
    jobs_template: list,
    chromosome:    list[int],
    pool_B:        int,
    pool_C:        int,
    scales:        list[float],
    dist:          str  = "pert",
    n_batches:     int  = 10,
    batch_size:    int  = 30,
    base_seed:     int  = 0,
    verbose:       bool = True,
) -> list[dict]:
    """
    For each delay scale run a mini SAA and record E[util], CVaR_5_low(util),
    E[sc2], and CVaR_95(sc2).
    """
    summaries: list[dict] = []
    for scale in scales:
        if verbose:
            print(f"\n  Stress test  scale={scale:.1f}x ...")
        batch_results, all_reps = run_saa(
            jobs_template, chromosome, pool_B, pool_C,
            n_batches=n_batches, batch_size=batch_size,
            dist=dist, base_seed=base_seed, scale=scale, verbose=verbose,
        )
        ev_util  = compute_expected_value(batch_results, kpi="oven_util_pct")
        ev_sc2   = compute_expected_value(batch_results, kpi="sc2")
        cvar_util = _util_robustness(all_reps)
        cvar_sc2  = _blocking_robustness(all_reps)
        summaries.append({
            "scale":             scale,
            "E_util":            ev_util["E_f"],
            "ci_lo_util":        ev_util["ci_lo"],
            "ci_hi_util":        ev_util["ci_hi"],
            "CVaR5_util":        cvar_util["CVaR"],
            "util_rob_score":    cvar_util["robustness_score"],
            "E_sc2":             ev_sc2["E_f"],
            "ci_lo_sc2":         ev_sc2["ci_lo"],
            "ci_hi_sc2":         ev_sc2["ci_hi"],
            "CVaR95_sc2":        cvar_sc2["CVaR"],
            "blocking_rob_score": cvar_sc2["robustness_score"],
        })
    return summaries


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_robustness_summary(
    batch_results: list[dict],
    all_reps:      list[dict],
    dist:          str,
) -> None:
    m     = len(batch_results)
    n     = batch_results[0]["batch_size"]
    scale = batch_results[0].get("scale", 1.0)

    cvar_util = _util_robustness(all_reps)
    cvar_sc2  = _blocking_robustness(all_reps)
    tail_util = compute_tail_risk(all_reps, "oven_util_pct")
    tail_sc2  = compute_tail_risk(all_reps, "sc2")
    ev_util   = compute_expected_value(batch_results, "oven_util_pct")
    ev_sc2    = compute_expected_value(batch_results, "sc2")

    print(f"\n{'='*72}")
    print(f"  Robustness Summary - {m} batches x {n} reps = {m*n} total  "
          f"dist={dist.upper()}  scale={scale:.1f}x")
    print(f"  {'='*68}")

    print(f"\n  OVEN UTILIZATION  (higher is better)")
    print(f"  E[util]          = {ev_util['E_f']:.2f} %")
    print(f"  95% CI           = [{ev_util['ci_lo']:.2f},  {ev_util['ci_hi']:.2f}] %")
    print(f"  P50 / P5         = {tail_util['P50']:.2f} %  /  {tail_util['P5']:.2f} %")
    print(f"  CVaR_5_low       = {cvar_util['CVaR']:.2f} %  "
          f"(mean of lowest {cvar_util['n_tail']} reps out of {cvar_util['n_total']})")
    util_score   = cvar_util["robustness_score"]
    util_verdict = "ROBUST" if util_score >= 0.85 else "FRAGILE"
    print(f"  Robustness score = CVaR_5 / E[util] = {util_score:.4f}  ({util_verdict})")

    print(f"\n  TRACK BLOCKING / SC-2  (lower is better)")
    print(f"  E[sc2]           = {ev_sc2['E_f']:.2f} h")
    print(f"  95% CI           = [{ev_sc2['ci_lo']:.2f},  {ev_sc2['ci_hi']:.2f}] h")
    print(f"  P50 / P95        = {tail_sc2['P50']:.2f} h  /  {tail_sc2['P95']:.2f} h")
    print(f"  CVaR_95_high     = {cvar_sc2['CVaR']:.2f} h  "
          f"(mean of highest {cvar_sc2['n_tail']} reps out of {cvar_sc2['n_total']})")
    sc2_score    = cvar_sc2["robustness_score"]
    sc2_verdict  = "ROBUST" if sc2_score <= 1.15 else "FRAGILE"
    print(f"  Robustness score = CVaR_95 / E[sc2] = {sc2_score:.4f}  ({sc2_verdict})")

    print(f"{'='*72}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv_batches(batch_results: list[dict], n_jobs: int, dist: str) -> None:
    path = f"robust_sa_batches_{dist}_{n_jobs}jobs.csv"
    fields = [k for k in batch_results[0] if not k.startswith("_")]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for row in batch_results:
            writer.writerow({k: row[k] for k in fields})
    print(f"  Batch CSV saved -> {path}")


def export_csv_reps(all_reps: list[dict], n_jobs: int, dist: str) -> None:
    path = f"robust_sa_reps_{dist}_{n_jobs}jobs.csv"
    fields = [k for k in all_reps[0] if not k.startswith("_")]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for row in all_reps:
            writer.writerow({k: row[k] for k in fields})
    print(f"  Reps  CSV saved -> {path}")


# ---------------------------------------------------------------------------
# Plot: E[util] and E[sc2] convergence
# ---------------------------------------------------------------------------

def plot_saa_convergence(
    batch_results: list[dict],
    dist:          str,
    n_jobs:        int,
    save_path:     Optional[str] = None,
) -> None:
    kpis = [
        ("oven_util_pct", "Oven utilization (%)", "#7EC850", True),
        ("sc2",           "Track blocking (h)",   "#E05C2A", False),
        ("makespan_h",    "Makespan (h)",          "#5BA4CF", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    fig.suptitle(
        f"SAA Convergence — Oven Util, Track Blocking & Makespan  |  "
        f"dist={dist.upper()}  |  {n_jobs} jobs",
        fontsize=13, fontweight="bold",
    )
    for ax, (key, label, color, higher_better) in zip(axes, kpis):
        z  = np.array([b[key] for b in batch_results])
        xs = np.arange(1, len(z) + 1)
        run_mean = np.array([z[:i].mean() for i in xs])
        run_std  = np.array([z[:i].std(ddof=1) if i > 1 else 0.0 for i in xs])
        run_se   = np.where(xs > 1, run_std / np.sqrt(xs), 0.0)
        t_crits  = np.array([
            float(_scipy_stats.t.ppf(0.975, df=i - 1)) if i > 1 else 0.0
            for i in xs
        ])
        ci_half = t_crits * run_se
        ax.fill_between(xs, run_mean - ci_half, run_mean + ci_half,
                        alpha=0.25, color=color, label="95% CI")
        ax.plot(xs, run_mean, color=color, linewidth=2.0, label="Running E[f]")
        ax.scatter(xs, z, s=18, color="grey", alpha=0.55, zorder=3, label="Batch avg")
        ax.axhline(float(run_mean[-1]), color="red", linestyle="--", linewidth=1.2,
                   label=f"Final E[f] = {run_mean[-1]:.2f}")
        direction = "(↑ better)" if higher_better else "(↓ better)"
        ax.set_xlabel("Batches accumulated")
        ax.set_ylabel(label)
        ax.set_title(f"{label} {direction} — convergence")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path is None:
        save_path = f"robust_sa_convergence_{dist}_{n_jobs}jobs.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Convergence plot saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Plot: tail risk / CVaR for oven util and track blocking
# ---------------------------------------------------------------------------

def plot_tail_risk(
    all_reps:  list[dict],
    dist:      str,
    n_jobs:    int,
    save_path: Optional[str] = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Tail Risk & CVaR — Oven Util & Track Blocking  |  "
        f"dist={dist.upper()}  |  {n_jobs} jobs  ({len(all_reps)} reps)",
        fontsize=13, fontweight="bold",
    )

    # --- Panel 1: Oven utilization (lower tail is bad) ---
    ax = axes[0]
    cvar_util = _util_robustness(all_reps)
    tail_util = compute_tail_risk(all_reps, "oven_util_pct")
    z = np.array([r["oven_util_pct"] for r in all_reps])
    _, edges, patches = ax.hist(
        z, bins=40, color="#7EC850", edgecolor="white", linewidth=0.4, alpha=0.75,
    )
    for patch, left in zip(patches, edges[:-1]):
        if left <= cvar_util["VaR"]:
            patch.set_facecolor("#C0392B")
            patch.set_alpha(0.85)
    ax.axvline(tail_util["mean"], color="navy",  linestyle="--", linewidth=1.5,
               label=f"E[util] = {tail_util['mean']:.1f} %")
    ax.axvline(tail_util["P5"],   color="orange", linestyle=":",  linewidth=1.3,
               label=f"P5 = {tail_util['P5']:.1f} %")
    ax.axvline(cvar_util["CVaR"], color="#8B0000", linestyle="-", linewidth=1.5,
               label=f"CVaR_5_low = {cvar_util['CVaR']:.1f} %")
    score   = cvar_util["robustness_score"]
    verdict = "ROBUST" if score >= 0.85 else "FRAGILE"
    ax.text(0.03, 0.95, f"Rob.Score = {score:.3f}\n{verdict}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="lightgreen" if score >= 0.85 else "lightsalmon",
                      alpha=0.8))
    red_patch = mpatches.Patch(color="#C0392B", alpha=0.85, label="Worst 5 % tail")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [red_patch], fontsize=8)
    ax.set_xlabel("Oven utilization (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Oven utilization — low tail risk  (↑ better)")
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 2: Track blocking (upper tail is bad) ---
    ax = axes[1]
    cvar_sc2 = _blocking_robustness(all_reps)
    tail_sc2 = compute_tail_risk(all_reps, "sc2")
    z = np.array([r["sc2"] for r in all_reps])
    _, edges, patches = ax.hist(
        z, bins=40, color="#E05C2A", edgecolor="white", linewidth=0.4, alpha=0.75,
    )
    for patch, left in zip(patches, edges[:-1]):
        if left >= cvar_sc2["VaR"]:
            patch.set_facecolor("#C0392B")
            patch.set_alpha(0.85)
    ax.axvline(tail_sc2["mean"], color="navy",   linestyle="--", linewidth=1.5,
               label=f"E[sc2] = {tail_sc2['mean']:.1f} h")
    ax.axvline(tail_sc2["P95"],  color="orange", linestyle=":",  linewidth=1.3,
               label=f"P95 = {tail_sc2['P95']:.1f} h")
    ax.axvline(cvar_sc2["CVaR"], color="#8B0000", linestyle="-", linewidth=1.5,
               label=f"CVaR_95_high = {cvar_sc2['CVaR']:.1f} h")
    score   = cvar_sc2["robustness_score"]
    verdict = "ROBUST" if score <= 1.15 else "FRAGILE"
    ax.text(0.97, 0.95, f"Rob.Score = {score:.3f}\n{verdict}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="lightgreen" if score <= 1.15 else "lightsalmon",
                      alpha=0.8))
    red_patch = mpatches.Patch(color="#C0392B", alpha=0.85, label="Worst 5 % tail")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [red_patch], fontsize=8)
    ax.set_xlabel("Track blocking — SC-2 (h)")
    ax.set_ylabel("Frequency")
    ax.set_title("Track blocking — high tail risk  (↓ better)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        save_path = f"robust_sa_tail_{dist}_{n_jobs}jobs.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Tail risk plot saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Plot: stress test
# ---------------------------------------------------------------------------

def plot_stress_test(
    stress_summaries: list[dict],
    dist:             str,
    n_jobs:           int,
    save_path:        Optional[str] = None,
) -> None:
    scales    = [s["scale"]             for s in stress_summaries]
    e_util    = [s["E_util"]            for s in stress_summaries]
    cvar_util = [s["CVaR5_util"]        for s in stress_summaries]
    util_rob  = [s["util_rob_score"]    for s in stress_summaries]
    e_sc2     = [s["E_sc2"]             for s in stress_summaries]
    cvar_sc2  = [s["CVaR95_sc2"]        for s in stress_summaries]
    sc2_rob   = [s["blocking_rob_score"] for s in stress_summaries]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Stress Test — Robustness vs Delay Scale  "
        f"(SA / dist={dist.upper()} / {n_jobs} jobs)",
        fontsize=13, fontweight="bold",
    )

    # Oven util: E[util] and CVaR_5_low
    ax = axes[0, 0]
    ax.plot(scales, e_util,    color="#7EC850", linewidth=2.0, marker="o", label="E[util]")
    ax.plot(scales, cvar_util, color="#C0392B", linewidth=2.0, marker="s",
            linestyle="--", label="CVaR_5_low (worst 5%)")
    ax.set_xlabel("Delay scale factor")
    ax.set_ylabel("Oven utilization (%)")
    ax.set_title("Oven util: E[util] and CVaR_5_low vs scale  (↑ better)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Oven util robustness score
    ax = axes[0, 1]
    bar_colors = ["#2ECC71" if r >= 0.85 else "#E74C3C" for r in util_rob]
    ax.bar(scales, util_rob, width=0.08, color=bar_colors, edgecolor="white")
    ax.axhline(1.0,  color="black",  linestyle="--", linewidth=1.0, label="Ideal (1.0)")
    ax.axhline(0.85, color="orange", linestyle=":",  linewidth=1.2, label="Fragility (0.85)")
    for x, r in zip(scales, util_rob):
        ax.text(x, r + 0.003, f"{r:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Delay scale factor")
    ax.set_ylabel("Util robustness score  (CVaR_5 / E[util])")
    ax.set_title("Oven util robustness score vs scale")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Track blocking: E[sc2] and CVaR_95
    ax = axes[1, 0]
    ax.plot(scales, e_sc2,    color="#E05C2A", linewidth=2.0, marker="o", label="E[sc2]")
    ax.plot(scales, cvar_sc2, color="#C0392B", linewidth=2.0, marker="s",
            linestyle="--", label="CVaR_95_high (worst 5%)")
    ax.set_xlabel("Delay scale factor")
    ax.set_ylabel("Track blocking (h)")
    ax.set_title("Track blocking: E[sc2] and CVaR_95 vs scale  (↓ better)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Blocking robustness score
    ax = axes[1, 1]
    bar_colors = ["#2ECC71" if r <= 1.15 else "#E74C3C" for r in sc2_rob]
    ax.bar(scales, sc2_rob, width=0.08, color=bar_colors, edgecolor="white")
    ax.axhline(1.0,  color="black",  linestyle="--", linewidth=1.0, label="Ideal (1.0)")
    ax.axhline(1.15, color="orange", linestyle=":",  linewidth=1.2, label="Fragility (1.15)")
    for x, r in zip(scales, sc2_rob):
        ax.text(x, r + 0.003, f"{r:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Delay scale factor")
    ax.set_ylabel("Blocking robustness score  (CVaR_95 / E[sc2])")
    ax.set_title("Track blocking robustness score vs scale")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        save_path = f"robust_sa_stress_{dist}_{n_jobs}jobs.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Stress test plot saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Plot: 4-panel robustness dashboard
# ---------------------------------------------------------------------------

def plot_robustness_dashboard(
    batch_results: list[dict],
    all_reps:      list[dict],
    dist:          str,
    n_jobs:        int,
    save_path:     Optional[str] = None,
) -> None:
    m = len(batch_results)
    n = batch_results[0]["batch_size"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Robustness Dashboard (SA)  —  dist={dist.upper()}  |  {n_jobs} jobs  "
        f"|  {m}×{n} = {m*n} reps",
        fontsize=13, fontweight="bold",
    )

    cvar_util     = _util_robustness(all_reps)
    cvar_sc2      = _blocking_robustness(all_reps)
    cvar_makespan = _makespan_robustness(all_reps)
    tail_util     = compute_tail_risk(all_reps, "oven_util_pct")
    tail_sc2      = compute_tail_risk(all_reps, "sc2")
    tail_makespan = compute_tail_risk(all_reps, "makespan_h")

    # Panel 1: Oven util convergence
    ax = axes[0, 0]
    z  = np.array([b["oven_util_pct"] for b in batch_results])
    xs = np.arange(1, len(z) + 1)
    run_mean = np.array([z[:i].mean() for i in xs])
    run_std  = np.array([z[:i].std(ddof=1) if i > 1 else 0.0 for i in xs])
    run_se   = np.where(xs > 1, run_std / np.sqrt(xs), 0.0)
    t_crits  = np.array([
        float(_scipy_stats.t.ppf(0.975, df=i - 1)) if i > 1 else 0.0 for i in xs
    ])
    ax.fill_between(xs, run_mean - t_crits * run_se, run_mean + t_crits * run_se,
                    alpha=0.25, color="#7EC850")
    ax.plot(xs, run_mean, color="#7EC850", linewidth=2.0, label="Running E[util]")
    ax.scatter(xs, z, s=15, color="grey", alpha=0.5)
    ax.axhline(float(run_mean[-1]), color="red", linestyle="--", linewidth=1.2,
               label=f"E[util] = {run_mean[-1]:.1f} %")
    ax.set_xlabel("Batches")
    ax.set_ylabel("Oven utilization (%)")
    ax.set_title("E[oven util] convergence  (↑ better)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: Oven util distribution with low-tail CVaR
    ax = axes[0, 1]
    z_all = np.array([r["oven_util_pct"] for r in all_reps])
    _, edges, patches = ax.hist(
        z_all, bins=40, color="#7EC850", edgecolor="white", linewidth=0.4, alpha=0.75,
    )
    for patch, left in zip(patches, edges[:-1]):
        if left <= cvar_util["VaR"]:
            patch.set_facecolor("#C0392B")
            patch.set_alpha(0.85)
    ax.axvline(tail_util["mean"], color="navy",   linestyle="--", linewidth=1.5,
               label=f"E[util]={tail_util['mean']:.1f}%")
    ax.axvline(cvar_util["CVaR"], color="#8B0000", linestyle="-", linewidth=1.5,
               label=f"CVaR_5={cvar_util['CVaR']:.1f}%")
    ax.axvline(tail_util["P5"],   color="orange", linestyle=":",  linewidth=1.3,
               label=f"P5={tail_util['P5']:.1f}%")
    score   = cvar_util["robustness_score"]
    verdict = "ROBUST" if score >= 0.85 else "FRAGILE"
    ax.text(0.03, 0.95, f"Rob.Score = {score:.3f}\n{verdict}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="lightgreen" if score >= 0.85 else "lightsalmon",
                      alpha=0.8))
    ax.set_title("Oven util distribution  (↑ better)")
    ax.set_xlabel("Oven utilization (%)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: Track blocking distribution with high-tail CVaR
    ax = axes[1, 0]
    z_sc2 = np.array([r["sc2"] for r in all_reps])
    _, edges, patches = ax.hist(
        z_sc2, bins=40, color="#E05C2A", edgecolor="white", linewidth=0.4, alpha=0.75,
    )
    for patch, left in zip(patches, edges[:-1]):
        if left >= cvar_sc2["VaR"]:
            patch.set_facecolor("#C0392B")
            patch.set_alpha(0.85)
    ax.axvline(tail_sc2["mean"], color="navy",   linestyle="--", linewidth=1.5,
               label=f"E[sc2]={tail_sc2['mean']:.1f}h")
    ax.axvline(cvar_sc2["CVaR"], color="#8B0000", linestyle="-", linewidth=1.5,
               label=f"CVaR_95={cvar_sc2['CVaR']:.1f}h")
    ax.axvline(tail_sc2["P95"],  color="orange", linestyle=":",  linewidth=1.3,
               label=f"P95={tail_sc2['P95']:.1f}h")
    score   = cvar_sc2["robustness_score"]
    verdict = "ROBUST" if score <= 1.15 else "FRAGILE"
    ax.text(0.97, 0.95, f"Rob.Score = {score:.3f}\n{verdict}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="lightgreen" if score <= 1.15 else "lightsalmon",
                      alpha=0.8))
    ax.set_title("Track blocking distribution  (↓ better)")
    ax.set_xlabel("Track blocking — SC-2 (h)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 4: Makespan distribution with high-tail CVaR
    ax = axes[1, 1]
    z_ms = np.array([r["makespan_h"] for r in all_reps])
    _, edges, patches = ax.hist(
        z_ms, bins=40, color="#5BA4CF", edgecolor="white", linewidth=0.4, alpha=0.75,
    )
    for patch, left in zip(patches, edges[:-1]):
        if left >= cvar_makespan["VaR"]:
            patch.set_facecolor("#C0392B")
            patch.set_alpha(0.85)
    ax.axvline(tail_makespan["mean"], color="navy",    linestyle="--", linewidth=1.5,
               label=f"E[makespan]={tail_makespan['mean']:.1f}h")
    ax.axvline(cvar_makespan["CVaR"], color="#8B0000", linestyle="-",  linewidth=1.5,
               label=f"CVaR_95={cvar_makespan['CVaR']:.1f}h")
    ax.axvline(tail_makespan["P95"],  color="orange",  linestyle=":",  linewidth=1.3,
               label=f"P95={tail_makespan['P95']:.1f}h")
    score   = cvar_makespan["robustness_score"]
    verdict = "ROBUST" if score <= 1.15 else "FRAGILE"
    ax.text(0.97, 0.95, f"Rob.Score = {score:.3f}\n{verdict}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="lightgreen" if score <= 1.15 else "lightsalmon",
                      alpha=0.8))
    red_patch = mpatches.Patch(color="#C0392B", alpha=0.85, label="Worst 5 % tail")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [red_patch], fontsize=8)
    ax.set_title(f"Makespan distribution  (rob.score={score:.3f} — {verdict})")
    ax.set_xlabel("Makespan (h)")
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        save_path = f"robust_sa_dashboard_{dist}_{n_jobs}jobs.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Dashboard saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Multi-GA convergence comparison
# ---------------------------------------------------------------------------

def plot_multi_ga_convergence(
    histories:  list[list[float]],
    csv_labels: list[str],
    n_jobs:     int,
    save_path:  Optional[str] = None,
) -> None:
    """
    Plot all GA convergence curves on one chart so the 10 runs can be compared.
    `histories[i]` is the best-fitness-per-generation list for csv_labels[i].
    """
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(
        f"GA Convergence — all {len(histories)} CSV files compared  |  {n_jobs} jobs",
        fontsize=13, fontweight="bold",
    )

    for i, (hist, label) in enumerate(zip(histories, csv_labels)):
        color = colors[i % len(colors)]
        ax.plot(hist, color=color, linewidth=1.6, alpha=0.85,
                label=f"{label}  (final={hist[-1]:.1f})")
        ax.scatter(len(hist) - 1, hist[-1], s=45, color=color, zorder=5)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness  (makespan + SC penalties)")
    ax.set_title("Lower = better")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = f"robust_sa_ga_all_convergence_{n_jobs}jobs.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Multi-GA convergence plot saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Cross-distribution E[util] and E[sc2] comparison
# ---------------------------------------------------------------------------

def plot_ev_comparison(
    all_batch: dict[str, list[dict]],
    n_jobs:    int,
    save_path: Optional[str] = None,
) -> None:
    dists  = list(all_batch.keys())
    colors = ["#5BA4CF", "#F5A623", "#7EC850", "#E05C2A", "#9B59B6"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Expected Oven Util, Track Blocking & Makespan by Distribution (SA)  —  {n_jobs} jobs",
        fontsize=13, fontweight="bold",
    )

    for ax, (kpi, label, higher_better) in zip(axes, [
        ("oven_util_pct", "E[Oven utilization (%)]", True),
        ("sc2",           "E[Track blocking (h)]",   False),
        ("makespan_h",    "E[Makespan (h)]",          False),
    ]):
        means  = [compute_expected_value(all_batch[d], kpi)["E_f"]    for d in dists]
        ci_lo  = [compute_expected_value(all_batch[d], kpi)["ci_lo"]  for d in dists]
        ci_hi  = [compute_expected_value(all_batch[d], kpi)["ci_hi"]  for d in dists]
        xs     = np.arange(len(dists))
        ax.bar(xs, means, color=[colors[i % len(colors)] for i in range(len(dists))],
               edgecolor="white", linewidth=0.5, alpha=0.8)
        ax.errorbar(xs, means,
                    yerr=[[m - lo for m, lo in zip(means, ci_lo)],
                          [hi - m for m, hi in zip(means, ci_hi)]],
                    fmt="none", color="black", capsize=6, linewidth=1.5)
        for i, (m, _, hi) in enumerate(zip(means, ci_lo, ci_hi)):
            y = hi
            ax.text(i, y + 0.2, f"{m:.1f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([d.upper() for d in dists], fontsize=10)
        ax.set_ylabel(label)
        direction = "(↑ better)" if higher_better else "(↓ better)"
        ax.set_title(f"{label} {direction}")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        save_path = f"robust_sa_ev_comparison_{n_jobs}jobs.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Comparison plot saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Per-job oven-start timing collection across SAA replications
# ---------------------------------------------------------------------------

def _collect_job_timing(
    all_reps: list[dict],
    job_ids:  list[str],
) -> dict[str, dict[str, list[float]]]:
    """
    For each job_id, collect oven_start, oven_end, repair_start, and mc_delay
    values across all reps that carry _jobs data.
    """
    data: dict[str, dict[str, list[float]]] = {
        jid: {"oven_start": [], "oven_end": [], "repair_start": [], "mc_delay": []}
        for jid in job_ids
    }
    for rep in all_reps:
        if "_jobs" not in rep:
            continue
        job_map = {j.job_id: j for j in rep["_jobs"]}
        for jid in job_ids:
            j = job_map.get(jid)
            if j is None:
                continue
            if j.oven_start is not None:
                data[jid]["oven_start"].append(j.oven_start)
            if j.oven_end is not None:
                data[jid]["oven_end"].append(j.oven_end)
            if j.repair_start is not None:
                data[jid]["repair_start"].append(j.repair_start)
            delay = getattr(j, "mc_delay", None)
            if delay is not None:
                data[jid]["mc_delay"].append(delay)
    return data


# ---------------------------------------------------------------------------
# Timing uncertainty plot — per-job oven-start whisker chart
# ---------------------------------------------------------------------------

def plot_robust_schedule_timing(
    timing_data:  dict[str, dict[str, list[float]]],
    selected_rep: dict,
    job_ids:      list[str],
    dist:         str,
    n_jobs:       int,
    coverage_pct: float,
    save_path:    Optional[str] = None,
) -> None:
    """
    Horizontal timeline chart: for each job shows the P5–P95 range and IQR of
    oven_start across all SAA reps, plus a coloured dot for the selected
    (coverage-target) replication.  Jobs are ordered by their median oven_start.
    """
    from matplotlib.lines import Line2D

    sel_job_map = {j.job_id: j for j in selected_rep["_jobs"]}

    ordered: list[tuple[float, str]] = []
    for jid in job_ids:
        ov = timing_data[jid]["oven_start"]
        if ov:
            ordered.append((float(np.median(ov)), jid))
    ordered.sort()
    ordered_ids = [jid for _, jid in ordered]

    n_rows = len(ordered_ids)
    n_reps = len(next(iter(timing_data.values()))["oven_start"])
    fig, ax = plt.subplots(figsize=(14, max(5, n_rows * 0.65 + 3.0)))
    fig.suptitle(
        f"Robust Schedule - Oven-Start Timing Uncertainty\n"
        f"dist={dist.upper()}  |  P{coverage_pct*100:.0f} coverage target  |  "
        f"{n_jobs} jobs  |  {n_reps} SAA reps",
        fontsize=11, fontweight="bold",
    )

    y_pos = {jid: i for i, jid in enumerate(reversed(ordered_ids))}

    for jid in ordered_ids:
        ov = np.array(timing_data[jid]["oven_start"])
        if len(ov) == 0:
            continue
        y = y_pos[jid]

        p5  = float(np.percentile(ov, 5))
        p25 = float(np.percentile(ov, 25))
        p50 = float(np.percentile(ov, 50))
        p75 = float(np.percentile(ov, 75))
        p95 = float(np.percentile(ov, 95))

        # P5–P95 whisker line
        ax.plot([p5, p95], [y, y], color="#5BA4CF", linewidth=1.0, alpha=0.55, zorder=2)
        for x_end in (p5, p95):
            ax.plot([x_end, x_end], [y - 0.12, y + 0.12],
                    color="#5BA4CF", linewidth=1.0, alpha=0.55, zorder=2)

        # IQR bar (P25–P75)
        ax.barh(y, p75 - p25, left=p25, height=0.52,
                color="#5BA4CF", alpha=0.65, zorder=3)

        # Median tick
        ax.plot([p50, p50], [y - 0.28, y + 0.28],
                color="navy", linewidth=2.2, zorder=4)

        # Selected-rep oven_start dot
        sel_j = sel_job_map.get(jid)
        if sel_j is not None and sel_j.oven_start is not None:
            ax.scatter([sel_j.oven_start], [y], s=65, color="#E05C2A",
                       zorder=5,
                       label=f"Selected rep (P{coverage_pct*100:.0f})"
                             if jid == ordered_ids[0] else "")

        # Percentile annotations (small)
        ax.text(p50, y + 0.32, f"{p50:.1f}h", ha="center", va="bottom",
                fontsize=6, color="navy")

        # Job label
        is_buf = sel_j.is_buffered if sel_j else ("BUF" in jid)
        lbl = f"{'[BUF] ' if is_buf else ''}{jid}"
        ax.text(p5 - 0.3, y, lbl, ha="right", va="center", fontsize=8,
                color="navy" if is_buf else "black")

    iqr_h = mpatches.Patch(color="#5BA4CF", alpha=0.65, label="IQR (P25–P75) oven start")
    wsk_h = Line2D([0], [0], color="#5BA4CF", alpha=0.55, linewidth=1.5,
                   label="P5–P95 range")
    med_h = Line2D([0], [0], color="navy", linewidth=2.2, label="Median oven start")
    sel_h = Line2D([0], [0], marker="o", color="w", markerfacecolor="#E05C2A",
                   markersize=9,
                   label=f"P{coverage_pct*100:.0f} rep - planning target")
    ax.legend(handles=[iqr_h, wsk_h, med_h, sel_h], fontsize=9, loc="lower right")

    ax.set_yticks([])
    ax.set_xlabel("Oven start time (h from simulation start)", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.subplots_adjust(top=0.88)

    if save_path is None:
        save_path = (f"robust_sa_schedule_timing_{dist}_"
                     f"P{int(coverage_pct*100)}_{n_jobs}jobs.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Timing uncertainty plot saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Robust schedule — select P-X replication, Gantt + timing uncertainty
# ---------------------------------------------------------------------------

def run_robust_schedule(
    pool_B:           int,
    pool_C:           int,
    all_reps:         list[dict],
    batch_results:    list[dict],
    dist:             str,
    n_jobs:           int,
    n_primary:        int   = 12,
    n_buffer:         int   = 3,
    coverage_target:  float = 0.75,
    save_gantt_path:  Optional[str] = None,
    save_timing_path: Optional[str] = None,
) -> dict:
    """
    Select and visualise the most robust achievable schedule for the first
    n_primary blades + n_buffer buffer blades after the SAA has been run.

    The "robust schedule" is the SAA replication whose composite score
    (normalised makespan + sc2 − oven_util) lies at the `coverage_target`
    percentile.  Concretely, `coverage_target` × 100 % of all SAA replications
    have an equal-or-better composite outcome on this chromosome, so a planner
    who uses this schedule as the operating target will achieve or beat it in
    that fraction of real-world delay scenarios.

    Steps
    -----
    1. Filter all_reps to scale=1.0 runs that carry _jobs / _staff_log data.
    2. Compute per-rep composite score: z(makespan) + z(sc2) − z(oven_util).
    3. Pick the replication closest to the coverage_target percentile of scores.
    4. Filter its jobs to the first n_primary primary + n_buffer real buffer jobs.
    5. Plot the Gantt (existing plot_gantt_des) annotated with coverage stats.
    6. Collect per-job oven-start distributions across all reps and plot the
       timing uncertainty chart.
    7. Print and return a summary dict.
    """
    valid = [r for r in all_reps
             if r.get("scale", 1.0) == 1.0 and "_jobs" in r and "_staff_log" in r]
    if not valid:
        print("  [robust schedule] No valid reps with full job data; skipping.")
        return {}

    # --- Composite score (z-normalised; lower = better scenario) ---
    ms_arr   = np.array([r["makespan_h"]    for r in valid])
    sc2_arr  = np.array([r["sc2"]           for r in valid])
    util_arr = np.array([r["oven_util_pct"] for r in valid])

    def _zn(arr: np.ndarray) -> np.ndarray:
        s = arr.std(ddof=1)
        return (arr - arr.mean()) / s if s > 1e-9 else np.zeros_like(arr)

    composite = _zn(ms_arr) + _zn(sc2_arr) - _zn(util_arr)

    cutoff = float(np.percentile(composite, coverage_target * 100))
    idx    = int(np.argmin(np.abs(composite - cutoff)))
    rep    = valid[idx]

    sel_ms   = rep["makespan_h"]
    sel_sc2  = rep["sc2"]
    sel_util = rep["oven_util_pct"]

    ms_cov   = float(np.mean(ms_arr   <= sel_ms))
    sc2_cov  = float(np.mean(sc2_arr  <= sel_sc2))
    util_cov = float(np.mean(util_arr >= sel_util))

    ev_util = compute_expected_value(batch_results, "oven_util_pct")
    ev_sc2  = compute_expected_value(batch_results, "sc2")
    ev_ms   = compute_expected_value(batch_results, "makespan_h")

    # --- Filter jobs for display ---
    jobs_copy  = rep["_jobs"]
    staff_log  = rep["_staff_log"]
    primary_sel = [j for j in jobs_copy if not j.is_buffered][:n_primary]
    buffer_sel  = [j for j in jobs_copy
                   if j.is_buffered and not j.job_id.startswith("GEN-")][:n_buffer]
    shown_jobs  = primary_sel + buffer_sel
    shown_ids   = [j.job_id for j in shown_jobs]

    print(f"\n{'='*72}")
    print(f"  Robust Schedule  -  P{coverage_target*100:.0f} coverage target")
    print(f"  dist={dist.upper()}  |  {n_primary} primary + {n_buffer} buffer  "
          f"|  {len(valid)} valid SAA reps")
    print(f"  Selected: batch={rep['batch']}  replication={rep['replication']}  "
          f"composite_score={composite[idx]:.3f}")
    print(f"  {'-'*68}")
    print(f"  {'KPI':<16}  {'Selected':>10}  {'E[KPI]':>10}  Coverage")
    print(f"  {'Makespan':<16}  {sel_ms:>9.1f}h  {ev_ms['E_f']:>9.1f}h  "
          f"{ms_cov*100:5.1f}% of scenarios <= {sel_ms:.1f} h")
    print(f"  {'SC-2 blocking':<16}  {sel_sc2:>9.2f}h  {ev_sc2['E_f']:>9.2f}h  "
          f"{sc2_cov*100:5.1f}% of scenarios <= {sel_sc2:.2f} h")
    print(f"  {'Oven util':<16}  {sel_util:>9.1f}%  {ev_util['E_f']:>9.1f}%  "
          f"{util_cov*100:5.1f}% of scenarios >= {sel_util:.1f} %")
    print(f"{'='*72}")

    # --- Gantt ---
    if save_gantt_path is None:
        save_gantt_path = (
            f"robust_sa_robust_gantt_{dist}_P{int(coverage_target*100)}_{n_jobs}jobs.png"
        )
    plot_gantt_des(
        shown_jobs, staff_log,
        pool_B=pool_B, pool_C=pool_C,
        save_path=save_gantt_path,
    )
    print(f"  Robust Gantt saved -> {save_gantt_path}")

    # --- Timing uncertainty across all valid reps ---
    timing_data = _collect_job_timing(valid, shown_ids)
    if save_timing_path is None:
        save_timing_path = (
            f"robust_sa_schedule_timing_{dist}_"
            f"P{int(coverage_target*100)}_{n_jobs}jobs.png"
        )
    plot_robust_schedule_timing(
        timing_data, rep, shown_ids,
        dist=dist, n_jobs=n_jobs,
        coverage_pct=coverage_target,
        save_path=save_timing_path,
    )

    return {
        "dist":              dist,
        "coverage_target":   coverage_target,
        "batch":             rep["batch"],
        "replication":       rep["replication"],
        "makespan_h":        sel_ms,
        "sc2_h":             sel_sc2,
        "oven_util_pct":     sel_util,
        "ms_coverage_pct":   round(ms_cov   * 100, 1),
        "sc2_coverage_pct":  round(sc2_cov  * 100, 1),
        "util_coverage_pct": round(util_cov * 100, 1),
    }


# ---------------------------------------------------------------------------
# EV schedule — deterministic run using E[delay] as a fixed constant
# ---------------------------------------------------------------------------

def run_ev_schedule(
    jobs_template: list,
    chromosome:    list[int],
    pool_B:        int,
    pool_C:        int,
    dist:          str,
    n_jobs:        int,
    n_primary:     int = 12,
    save_path:     Optional[str] = None,
) -> None:
    """
    Run one deterministic simulation where every delay draw equals E[delay]
    (the empirical mean from data_real.csv).  After the run, filter the Gantt
    to the first `n_primary` primary jobs + all buffer jobs so the chart stays
    readable.

    This is the classical EV (Expected Value) solution from stochastic
    programming: replace every random variable with its expectation and solve
    the resulting deterministic problem.
    """
    ev_delay = 0.0              # deterministic baseline — no delay applied

    jobs_copy = _fresh_jobs(jobs_template)
    original  = _des_mod._sample_pert_delay
    _des_mod._sample_pert_delay = lambda _rng, _job=None: ev_delay
    try:
        staff_log, _, sc2, sc3 = simulate_des_bc(
            jobs_copy, chromosome, pool_B, pool_C,
            use_sa=True, use_lookahead=True,
            use_delays=True,
            delay_seed=0,
            gen_seed=0,
        )
    finally:
        _des_mod._sample_pert_delay = original

    done     = [j for j in jobs_copy if j.oven_end is not None]
    makespan = max((j.oven_end for j in done), default=0.0)
    fitness  = makespan + SC2_WEIGHT * sc2 + SC3_WEIGHT * sc3
    oven_util = _oven_util_pct(jobs_copy, makespan)

    print(f"\n{'='*60}")
    print(f"  EV Schedule  (delay = 0 h - deterministic baseline  |  dist={dist.upper()})")
    print(f"  Jobs shown : first {n_primary} primary + all buffer")
    print(f"  Makespan   : {makespan:.1f} h")
    print(f"  SC-2       : {sc2:.2f} h   SC-3 : {sc3:.2f} h")
    print(f"  Fitness    : {fitness:.1f}")
    print(f"  Oven util  : {oven_util:.1f} %")
    n_primary_done = sum(1 for j in done if not j.is_buffered)
    n_buffer_done  = sum(1 for j in done if j.is_buffered
                         and not j.job_id.startswith("GEN-"))
    n_gen_done     = sum(1 for j in done if j.job_id.startswith("GEN-"))
    print(f"  Completed  : {n_primary_done} primary  {n_buffer_done} buffer  "
          f"{n_gen_done} generated")
    print(f"{'='*60}")

    # Filter to first n_primary primary jobs + buffer jobs (no GEN- jobs)
    primary_jobs = [j for j in jobs_copy if not j.is_buffered][:n_primary]
    buffer_jobs  = [j for j in jobs_copy
                    if j.is_buffered and not j.job_id.startswith("GEN-")]
    shown_jobs   = primary_jobs + buffer_jobs

    if save_path is None:
        save_path = f"robust_sa_ev_gantt_{n_jobs}jobs.png"
    plot_gantt_des(
        shown_jobs, staff_log,
        pool_B=pool_B, pool_C=pool_C,
        save_path=save_path,
    )
    print(f"  EV Gantt saved -> {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAA + robustness analysis (oven util & track blocking)"
    )
    parser.add_argument("--workers_b",         type=int,   default=N_WORKERS_B)
    parser.add_argument("--workers_c",         type=int,   default=N_WORKERS_C)
    parser.add_argument("--jobs",              type=int,   default=None)
    parser.add_argument("--generations",       type=int,   default=GA_GENERATIONS)
    parser.add_argument("--popsize",           type=int,   default=GA_POP_SIZE)
    parser.add_argument("--batches",           type=int,   default=20)
    parser.add_argument("--batch_size",        type=int,   default=50)
    parser.add_argument("--dist",              type=str,   default="pert",
                        choices=DIST_NAMES)
    parser.add_argument("--all_dists",         action="store_true")
    parser.add_argument("--base_seed",         type=int,   default=0)
    parser.add_argument("--stress",            action="store_true")
    parser.add_argument("--stress_scales",     type=float, nargs="+",
                        default=[1.0, 1.5, 2.0, 2.5])
    parser.add_argument("--stress_batches",    type=int,   default=10)
    parser.add_argument("--stress_batch_size", type=int,   default=30)
    parser.add_argument("--max_gen_jobs",      type=int,   default=MAX_GENERATED_JOBS)
    parser.add_argument("--robust_schedule",   action="store_true",
                        help="Generate the robust schedule Gantt + timing uncertainty plot")
    parser.add_argument("--robust_coverage",   type=float, default=0.75,
                        help="Coverage target for the robust schedule (default 0.75 = P75)")
    parser.add_argument("--robust_n_primary",  type=int,   default=12,
                        help="Number of primary blades to show in robust Gantt")
    parser.add_argument("--robust_n_buffer",   type=int,   default=3,
                        help="Number of buffer blades to show in robust Gantt")
    args = parser.parse_args()

    _ga_mod.MAX_GENERATED_JOBS = args.max_gen_jobs
    _des_mod.MAX_GENERATED_JOBS = args.max_gen_jobs

    pool_B = args.workers_b
    pool_C = args.workers_c

    jobs_template = load_jobs()
    if args.jobs is not None:
        jobs_template = jobs_template[:args.jobs]
    n = len(jobs_template) + args.max_gen_jobs

    ests = [j.estimated_h for j in jobs_template]
    print(f"\nLoaded {len(jobs_template)} jobs  |  B workers: {pool_B}  |  C workers: {pool_C}")
    print(f"  Estimated duration: min={min(ests):.1f}h  "
          f"max={max(ests):.1f}h  avg={sum(ests)/len(jobs_template):.1f}h")

    # Build per-batch job templates: current_status.csv + current_status_1..9.csv
    # Each file covers 2 batches → 10 files × 2 = 20 batches total.
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    _csv_paths = [os.path.join(_base_dir, "current_status.csv")] + [
        os.path.join(_base_dir, f"current_status_{i}.csv") for i in range(1, 10)
    ]
    batch_templates: list = []
    for _path in _csv_paths:
        if os.path.exists(_path):
            _tmpl = load_jobs(_path)
            if args.jobs is not None:
                _tmpl = _tmpl[:args.jobs]
            batch_templates.append(_tmpl)
        else:
            print(f"  [warning] {os.path.basename(_path)} not found — falling back to current_status.csv")
            batch_templates.append(jobs_template)
    print(f"  Loaded {len(batch_templates)} batch templates "
          f"({len(batch_templates)} files × 2 batches each)")

    p = _get_dist_params()
    print(f"\nDelay params fitted from data_real.csv (n={p['n']} observations)")
    print(f"  PERT / Triangular : a={p['a']:.2f}  mode={p['mode']:.2f}  b={p['b']:.2f}")
    print(f"  Normal            : mu={p['mu']:.2f}  sigma={p['sigma']:.2f}")
    print(f"  Lognormal shift   : {p['ln_shift']:.4f}  ln_mu={p['ln_mu']:.4f}  "
          f"ln_sigma={p['ln_sigma']:.4f}")

    # Run GA once per CSV file — each chromosome covers its 2 batches.
    _csv_names = ["current_status.csv"] + [f"current_status_{i}.csv" for i in range(1, 10)]
    print(f"\nRunning GA for each of {len(batch_templates)} templates "
          f"(pop={args.popsize}, gen={args.generations}) ...")
    batch_chromosomes: list = []
    _all_histories:    list = []
    for _i, _tmpl in enumerate(batch_templates):
        print(f"  GA {_i + 1}/{len(batch_templates)}  [{_csv_names[_i]}] ...")
        _chrom, _fit, _hist = run_ga(
            _tmpl, pool_B=pool_B, pool_C=pool_C,
            pop_size=args.popsize, n_gen=args.generations,
        )
        print(f"    fitness: {_fit:.1f}")
        batch_chromosomes.append(_chrom)
        _all_histories.append(_hist)

    plot_multi_ga_convergence(
        _all_histories, _csv_names[:len(batch_templates)], n,
        save_path=f"robust_sa_ga_all_convergence_{n}jobs.png",
    )

    best_chrom  = batch_chromosomes[0]
    best_fit_ga = None

    dists_to_run   = DIST_NAMES if args.all_dists else [args.dist]
    all_batch_map: dict[str, list[dict]] = {}
    all_reps_map:  dict[str, list[dict]] = {}

    for dist in dists_to_run:
        batch_results, all_reps = run_saa(
            jobs_template, best_chrom, pool_B, pool_C,
            n_batches=args.batches, batch_size=args.batch_size,
            dist=dist, base_seed=args.base_seed,
            batch_templates=batch_templates,
            batch_chromosomes=batch_chromosomes,
        )
        print_robustness_summary(batch_results, all_reps, dist)
        export_csv_batches(batch_results, n, dist)
        export_csv_reps(all_reps, n, dist)
        plot_saa_convergence(batch_results, dist, n)
        plot_tail_risk(all_reps, dist, n)
        plot_robustness_dashboard(batch_results, all_reps, dist, n)
        all_batch_map[dist] = batch_results
        all_reps_map[dist]  = all_reps

        if args.stress:
            print(f"\n  --- Stress test [{dist.upper()}] ---")
            stress_summaries = run_stress_test(
                jobs_template, best_chrom, pool_B, pool_C,
                scales=args.stress_scales, dist=dist,
                n_batches=args.stress_batches,
                batch_size=args.stress_batch_size,
                base_seed=args.base_seed + 10_000,
            )
            plot_stress_test(stress_summaries, dist, n)

    if args.all_dists and len(all_batch_map) > 1:
        plot_ev_comparison(all_batch_map, n)

    print(f"\n--- EV Schedule (first 12 primary + buffer) ---")
    run_ev_schedule(
        jobs_template, best_chrom, pool_B, pool_C,
        dist=args.dist, n_jobs=n,
    )

    if args.robust_schedule:
        for dist in dists_to_run:
            print(f"\n--- Robust Schedule [{dist.upper()}]  "
                  f"(P{int(args.robust_coverage*100)} coverage, "
                  f"first {args.robust_n_primary} primary + "
                  f"{args.robust_n_buffer} buffer) ---")
            run_robust_schedule(
                pool_B=pool_B,
                pool_C=pool_C,
                all_reps=all_reps_map[dist],
                batch_results=all_batch_map[dist],
                dist=dist,
                n_jobs=n,
                n_primary=args.robust_n_primary,
                n_buffer=args.robust_n_buffer,
                coverage_target=args.robust_coverage,
            )
