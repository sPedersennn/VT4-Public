"""
des_mc_analysis.py
==================
Monte Carlo DES: 200 independent replications of the blade repair DES,
comparing five delay distributions applied to each job's phase-2 duration.

Distributions tested:
  pert        Modified PERT fitted to data_real.csv  (matches base simulation)
  normal      Normal(mu, sigma) fitted to data_real.csv
  uniform     Uniform[a, b] spanning data_real.csv delay range
  triangular  Triangular(a, mode, b) from data_real.csv
  lognormal   Shifted LogNormal fitted to data_real.csv

How it works
  The existing simulate_des_bc() in des_simulationBCtester.py calls
  _sample_pert_delay(rng) by name at every epoch.  This script monkey-patches
  that name before each simulation run, swapping in the chosen distribution's
  sampler.  The original file is never modified.

Outputs (all in the working directory)
  mc_results_<dist>_<n>jobs.csv     per-replication KPIs
  mc_summary_<dist>_<n>jobs.png     4-panel KPI distribution plots
  mc_comparison_<n>jobs.png         violin comparison across all dists
"""
from __future__ import annotations

import argparse
import csv
import contextlib
import os
import random
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as _scipy_beta

# Import the simulation engine and supporting utilities
import des_simulationBCtester as _des_mod
from des_simulationBCtester import (
    simulate_des_bc,
    plot_gantt_des,
    plot_weekly_throughput,
    _get_pert_params,
)
from ga_simulationBC import (
    _fresh_jobs,
    load_jobs,
    run_ga,
    plot_convergence,
    N_WORKERS_B,
    N_WORKERS_C,
    N_JOBS,
    SC1_WEIGHT,
    SC2_WEIGHT,
    SC3_WEIGHT,
    GA_POP_SIZE,
    GA_GENERATIONS,
)

# ---------------------------------------------------------------------------
# Distribution parameter fitting (data_real.csv, cached after first read)
# ---------------------------------------------------------------------------

_DIST_PARAMS_CACHE: Optional[dict] = None


def _load_dist_params() -> dict:
    """
    Read Actual-Estimated delays from data_real.csv and fit parameters for every
    supported distribution.
    """
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

    # Lognormal: shift to make all values strictly positive, then fit
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
# Delay samplers â€” all share signature  (rng: random.Random) -> float
# ---------------------------------------------------------------------------

def _sampler_pert(rng: random.Random) -> float:
    """Modified PERT via inverse-CDF on the underlying Beta distribution."""
    a, mode, b, gamma = _get_pert_params()          # reuse existing fitted params
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
    """Sample from a shifted lognormal to allow negative delays (early finish)."""
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
# Monkey-patch context manager
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _patch_sampler(dist: str):
    """
    Temporarily replace _sample_pert_delay in des_simulationBCtester with the
    chosen distribution's sampler.  _on_epoch looks up the name at call time,
    so this swap is transparent to the simulation engine.
    """
    original = _des_mod._sample_pert_delay
    _des_mod._sample_pert_delay = DISTRIBUTIONS[dist]
    try:
        yield
    finally:
        _des_mod._sample_pert_delay = original


# ---------------------------------------------------------------------------
# Per-replication runner
# ---------------------------------------------------------------------------

def run_one(
    jobs_template: list,
    chromosome:    list[int],
    pool_B:        int,
    pool_C:        int,
    delay_seed:    int,
    dist:          str,
) -> dict:
    """Run a single DES replication and return a KPI dict."""
    jobs_copy = _fresh_jobs(jobs_template)
    with _patch_sampler(dist):
        staff_log, sc1, sc2, sc3 = simulate_des_bc(
            jobs_copy, chromosome, pool_B, pool_C,
            use_sa=True, use_lookahead=True,
            use_delays=True,
            delay_seed=delay_seed,
        )
    done = [j for j in jobs_copy if j.oven_end is not None]
    makespan = max((j.oven_end for j in done), default=float("inf"))
    fitness  = makespan + SC1_WEIGHT * sc1 + SC2_WEIGHT * sc2 + SC3_WEIGHT * sc3
    mc_delays = [d for j in done if (d := getattr(j, "mc_delay", None)) is not None]

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
        # keep last replication's objects for optional downstream plotting
        "_jobs":      jobs_copy,
        "_staff_log": staff_log,
        "_pool_B":    pool_B,
        "_pool_C":    pool_C,
    }


# ---------------------------------------------------------------------------
# Monte Carlo loop
# ---------------------------------------------------------------------------

def run_monte_carlo(
    jobs_template:  list,
    chromosome:     list[int],
    pool_B:         int,
    pool_C:         int,
    n_replications: int  = 200,
    dist:           str  = "pert",
    base_seed:      int  = 0,
    verbose:        bool = True,
) -> list[dict]:
    """
    Run n_replications independent DES simulations with delay distribution dist.

    Each replication i uses delay_seed = base_seed + i so the delay RNG is
    unique per replication while the GA chromosome is fixed.
    """
    results: list[dict] = []

    if verbose:
        print(f"\n  Running {n_replications} replications  [dist={dist.upper()}] ...")

    for rep in range(n_replications):
        r = run_one(jobs_template, chromosome, pool_B, pool_C,
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
    fmt = "  {:<24} {:>9.2f}  Â±{:>7.2f}  [{:>7.1f}, {:>7.1f}]"
    print(f"\n{'='*60}")
    print(f"  Monte Carlo summary â€” {len(results)} reps  dist={dist.upper()}")
    print(f"  {'Metric':<24} {'Mean':>9}   {'Std':>7}   {'[Min':>7}, {'Max]':>7}")
    print(f"  {'-'*56}")
    for label, key in [
        ("Makespan (h)",       "makespan_h"),
        ("Fitness",            "fitness"),
        ("SC-1 (worker band)", "sc1"),
        ("SC-2 (oven wait)",   "sc2"),
        ("SC-3 (oven idle)",   "sc3"),
        ("Mean job delay (h)", "mean_delay_h"),
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
    path   = f"mc_results_{dist}_{n_jobs}jobs.csv"
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
    """4-panel figure: makespan hist, fitness hist, SC box plots, delay scatter."""
    ms  = np.array([r["makespan_h"]   for r in results])
    fit = np.array([r["fitness"]      for r in results])
    sc1 = np.array([r["sc1"]          for r in results])
    sc2 = np.array([r["sc2"]          for r in results])
    sc3 = np.array([r["sc3"]          for r in results])
    dly = np.array([r["mean_delay_h"] for r in results])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Monte Carlo DES â€” {len(results)} replications  |  "
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

    # ---- Fitness histogram ----
    ax = axes[0, 1]
    ax.hist(fit, bins=30, color="#F5A623", edgecolor="white", linewidth=0.5)
    ax.axvline(fit.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean {fit.mean():.1f}")
    ax.axvline(np.percentile(fit, 5),  color="orange", linestyle=":", linewidth=1.2,
               label=f"P5={np.percentile(fit,5):.1f}")
    ax.axvline(np.percentile(fit, 95), color="orange", linestyle=":", linewidth=1.2,
               label=f"P95={np.percentile(fit,95):.1f}")
    ax.set_xlabel("Fitness (weighted sum)")
    ax.set_ylabel("Frequency")
    ax.set_title("Total fitness distribution")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ---- SC penalty box plots ----
    ax = axes[1, 0]
    ax.boxplot(
        [sc1, sc2, sc3],
        tick_labels=["SC-1\nworker band", "SC-2\noven wait", "SC-3\noven idle"],
        patch_artist=True,
        boxprops=dict(facecolor="#D4E6F1", color="#2C7FB8"),
        medianprops=dict(color="red", linewidth=2),
        whiskerprops=dict(color="#2C7FB8"),
        capprops=dict(color="#2C7FB8"),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    ax.set_ylabel("Penalty value")
    ax.set_title("Soft-constraint penalties across replications")
    ax.grid(axis="y", alpha=0.3)

    # ---- Mean delay vs makespan scatter ----
    ax = axes[1, 1]
    sc = ax.scatter(dly, ms, c=fit, cmap="RdYlGn_r",
                    alpha=0.55, s=20, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Fitness")
    z = np.polyfit(dly, ms, 1)
    x_line = np.linspace(dly.min(), dly.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line),
            color="navy", linestyle="--", linewidth=1.2,
            label=f"Trend (slope {z[0]:.2f})")
    ax.set_xlabel("Mean per-job delay (h)")
    ax.set_ylabel("Makespan (h)")
    ax.set_title("Delay vs makespan  (colour = fitness)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        save_path = f"mc_summary_{dist}_{n_jobs}jobs.png"
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
    """Violin + mean-marker comparison of makespan and fitness across distributions."""
    dists     = list(all_results.keys())
    makespans = [np.array([r["makespan_h"] for r in all_results[d]]) for d in dists]
    fitnesses = [np.array([r["fitness"]    for r in all_results[d]]) for d in dists]
    n_reps    = len(next(iter(all_results.values())))

    colors = ["#5BA4CF", "#F5A623", "#7EC850", "#E05C2A", "#9B59B6"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Delay-distribution comparison â€” {n_jobs} jobs  ({n_reps} reps each)",
        fontsize=13, fontweight="bold",
    )

    for ax, datasets, ylabel, title in [
        (axes[0], makespans, "Makespan (h)",         "Makespan by distribution"),
        (axes[1], fitnesses, "Fitness (weighted sum)", "Fitness by distribution"),
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

        # Annotate mean value above each violin
        for i, m in enumerate(means):
            ax.text(i, m + 0.01 * (max(means) - min(means)),
                    f"{m:.0f}", ha="center", va="bottom", fontsize=8, color="red")

        ax.set_xticks(range(len(dists)))
        ax.set_xticklabels([d.upper() for d in dists], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        save_path = f"mc_comparison_{n_jobs}jobs.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Comparison plot saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monte Carlo DES â€” 200 replications with pluggable delay distributions"
    )
    parser.add_argument("--workers_b",    type=int, default=N_WORKERS_B)
    parser.add_argument("--workers_c",    type=int, default=N_WORKERS_C)
    parser.add_argument("--jobs",         type=int, default=N_JOBS)
    parser.add_argument("--generations",  type=int, default=GA_GENERATIONS)
    parser.add_argument("--popsize",      type=int, default=GA_POP_SIZE)
    parser.add_argument("--replications", type=int, default=200,
                        help="Number of MC replications per distribution (default: 200)")
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

    # Load jobs
    jobs_template = load_jobs()
    if args.jobs is not None:
        jobs_template = jobs_template[:args.jobs]
    n = len(jobs_template)

    ests = [j.estimated_h for j in jobs_template]
    print(f"\nLoaded {n} jobs  |  B workers: {pool_B}  |  C workers: {pool_C}")
    print(f"  Estimated duration: min={min(ests):.1f}h  "
          f"max={max(ests):.1f}h  avg={sum(ests)/n:.1f}h")

    # Print fitted delay params
    p = _get_dist_params()
    print(f"\nDelay params fitted from data_real.csv (n={p['n']} observations)")
    print(f"  PERT / Triangular : a={p['a']:.2f}  mode={p['mode']:.2f}  b={p['b']:.2f}")
    print(f"  Normal            : mu={p['mu']:.2f}  sigma={p['sigma']:.2f}")
    print(f"  Uniform           : [{p['a']:.2f}, {p['b']:.2f}]")
    print(f"  Lognormal shift   : {p['ln_shift']:.4f}  ln_mu={p['ln_mu']:.4f}  "
          f"ln_sigma={p['ln_sigma']:.4f}")

    # Run GA once â€” chromosome is shared across all MC replications
    print(f"\nRunning GA  (pop={args.popsize}, gen={args.generations}) ...")
    best_chrom, best_fit_ga, history = run_ga(
        jobs_template,
        pool_B=pool_B, pool_C=pool_C,
        pop_size=args.popsize, n_gen=args.generations,
    )
    print(f"Best GA fitness: {best_fit_ga:.1f}")
    plot_convergence(history, save_path=f"des_bc_convergence_{n}jobs.png")

    dists_to_run = DIST_NAMES if args.all_dists else [args.dist]
    all_results: dict[str, list[dict]] = {}

    for dist in dists_to_run:
        results = run_monte_carlo(
            jobs_template, best_chrom, pool_B, pool_C,
            n_replications=args.replications,
            dist=dist,
            base_seed=args.base_seed,
        )
        print_summary(results, dist)
        export_csv(results, n, dist)
        plot_summary(results, dist, n)
        all_results[dist] = results

    if args.all_dists and len(all_results) > 1:
        plot_comparison(all_results, n)

    # Optional Gantt from last replication of the primary dist
    if args.gantt:
        last = all_results[args.dist][-1]
        jobs_last     = last["_jobs"]
        staff_log     = last["_staff_log"]
        makespan_last = max(
            (j.oven_end for j in jobs_last if j.oven_end is not None), default=0.0
        )
        plot_gantt_des(jobs_last, staff_log,
                       pool_B=pool_B, pool_C=pool_C,
                       save_path=f"gantt_des_mc_{n}jobs.png")
        plot_weekly_throughput(jobs_last, makespan_last,
                               save_path=f"weekly_throughput_mc_{n}jobs.png")


