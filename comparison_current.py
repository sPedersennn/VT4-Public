"""
comparison_current.py
=============
Runs DES-SOT (des_simulationBC_sot_current.py) and DES-GA (des_simulationBCtester_current.py)
on the same job set and compares their results side by side.

Outputs:
  - Console summary table
  - comparison_overview.png    : makespan, SC penalties, worker stats
  - comparison_throughput.png  : weekly blade throughput, both methods
  - comparison_workers.png     : B / C worker utilisation over time
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

# ── SOT simulation ────────────────────────────────────────────────────────────
from sot_simulationBC_current import (
    load_jobs as load_sot_jobs,
    N_WORKERS_B,
    N_WORKERS_C,
    WEEK_HOURS,
    MAX_GENERATED_JOBS,
)
from des_simulationBC_sot_current import simulate_des_sot_bc, DELAY_SEED as SOT_DELAY_SEED

# ── GA-DES simulation ─────────────────────────────────────────────────────────
from ga_simulationBC_current import (
    load_jobs as load_ga_jobs,
    run_ga,
    GA_POP_SIZE,
    GA_GENERATIONS,
    SC2_WEIGHT,
    SC3_WEIGHT,
)
from des_simulationBCtester_current import simulate_des_bc, DELAY_SEED as GA_DELAY_SEED

N_JOBS = 200

# ── Shared constraints ────────────────────────────────────────────────────────
from constraintsBC import (
    CHANGEOVER_TIME,
    MAX_REPAIR_TRACKS,
    OVEN_PROCESS_TIME,
    WORKER_REASSIGN_INTERVAL,
    MIN_PROCESS_TIME_RATIO,
)


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_metrics(jobs: list, label: str) -> dict:
    """Return a KPI dict for a completed job list."""
    done = [j for j in jobs if j.oven_end is not None and j.repair_end is not None]
    if not done:
        return {"label": label, "n_done": 0}

    makespan    = max(j.oven_end for j in done)
    repair_times = [j.repair_end - j.repair_start for j in done if j.repair_start is not None]
    avg_repair  = sum(repair_times) / len(repair_times) if repair_times else 0.0
    avg_B       = sum(j.avg_workers_B for j in done) / len(done)
    avg_C       = sum(j.avg_workers_C for j in done) / len(done)

    ratios      = [(j.repair_end - j.repair_start) / j.estimated_h
                   for j in done if j.repair_start is not None]
    hc2_viol    = sum(1 for r in ratios if r < MIN_PROCESS_TIME_RATIO - 1e-6)

    delays      = [j.mc_delay for j in done
                   if getattr(j, "mc_delay", None) is not None]
    avg_delay   = sum(delays) / len(delays) if delays else 0.0

    oven_util   = (len(done) * OVEN_PROCESS_TIME) / makespan if makespan > 0 else 0.0

    track_idle  = 0.0
    for i in range(MAX_REPAIR_TRACKS):
        track_jobs = [j for j in done if j.repair_track == i]
        occupied   = sum((j.repair_end - j.repair_start) + CHANGEOVER_TIME
                         for j in track_jobs)
        track_idle += max(0.0, makespan - occupied)

    return {
        "label":      label,
        "n_done":     len(done),
        "makespan":   makespan,
        "avg_repair": avg_repair,
        "avg_B":      avg_B,
        "avg_C":      avg_C,
        "hc2_viol":   hc2_viol,
        "avg_delay":  avg_delay,
        "oven_util":  oven_util,
        "track_idle": track_idle,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Console table
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(
    sot_m: dict, ga_m: dict,
    sot_sc2: float, sot_sc3: float,
    ga_sc2:  float, ga_sc3:  float,
) -> None:
    sot_fit = sot_m["makespan"] + SC2_WEIGHT*sot_sc2 + SC3_WEIGHT*sot_sc3
    ga_fit  = ga_m["makespan"]  + SC2_WEIGHT*ga_sc2  + SC3_WEIGHT*ga_sc3

    rows = [
        ("Jobs completed",       f"{sot_m['n_done']}",            f"{ga_m['n_done']}"),
        ("Makespan (h)",         f"{sot_m['makespan']:.1f}",      f"{ga_m['makespan']:.1f}"),
        ("Avg repair time (h)",  f"{sot_m['avg_repair']:.1f}",    f"{ga_m['avg_repair']:.1f}"),
        ("Avg B workers / job",  f"{sot_m['avg_B']:.2f}",         f"{ga_m['avg_B']:.2f}"),
        ("Avg C workers / job",  f"{sot_m['avg_C']:.2f}",         f"{ga_m['avg_C']:.2f}"),
        ("Oven utilisation",     f"{sot_m['oven_util']:.1%}",     f"{ga_m['oven_util']:.1%}"),
        ("Track idle time (h)",  f"{sot_m['track_idle']:.1f}",    f"{ga_m['track_idle']:.1f}"),
        ("HC-2 violations",      f"{sot_m['hc2_viol']}",          f"{ga_m['hc2_viol']}"),
        ("Avg MC delay (h)",     f"{sot_m['avg_delay']:.2f}",     f"{ga_m['avg_delay']:.2f}"),
        ("SC-2 penalty (h)",     f"{sot_sc2:.1f}",                f"{ga_sc2:.1f}"),
        ("SC-3 penalty (h)",     f"{sot_sc3:.1f}",                f"{ga_sc3:.1f}"),
        ("Total fitness",        f"{sot_fit:.1f}",                f"{ga_fit:.1f}"),
    ]

    print(f"\n{'='*62}")
    print(f"{'COMPARISON : DES-SOT  vs  DES-GA':^62}")
    print(f"{'='*62}")
    print(f"  {'Metric':<28} {'DES-SOT':>14} {'DES-GA':>14}")
    print(f"  {'-'*58}")
    for name, sv, gv in rows:
        print(f"  {name:<28} {sv:>14} {gv:>14}")
    print(f"{'='*62}")

    delta = ga_m["makespan"] - sot_m["makespan"]
    winner = "DES-SOT" if delta > 0 else ("DES-GA" if delta < 0 else "Tie")
    print(f"\n  Makespan delta   : {delta:+.1f} h  ({winner} finishes first)")
    delta_fit = ga_fit - sot_fit
    winner_fit = "DES-SOT" if delta_fit > 0 else ("DES-GA" if delta_fit < 0 else "Tie")
    print(f"  Fitness delta    : {delta_fit:+.1f}  ({winner_fit} has lower total fitness)\n")


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

_SOT_COLOR = "#5BA4CF"
_GA_COLOR  = "#F5A623"


def plot_overview(
    sot_m: dict, ga_m: dict,
    sot_sc2: float, sot_sc3: float,
    ga_sc2:  float, ga_sc3:  float,
    save_path: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("DES-SOT vs DES-GA — Overview", fontsize=13, fontweight="bold")

    labels = ["DES-SOT", "DES-GA"]
    colors = [_SOT_COLOR, _GA_COLOR]

    # Panel 1 – makespan
    ax = axes[0]
    vals = [sot_m["makespan"], ga_m["makespan"]]
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f"{v:.1f} h", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Makespan (h)")
    ax.set_ylabel("Hours")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(vals) * 1.15)

    # Panel 2 – SC-2 and SC-3 penalties (grouped bars, weighted contributions)
    ax = axes[1]
    x   = np.arange(2)
    w   = 0.35
    sot_pen = [SC2_WEIGHT * sot_sc2, SC3_WEIGHT * sot_sc3]
    ga_pen  = [SC2_WEIGHT * ga_sc2,  SC3_WEIGHT * ga_sc3]
    ax.bar(x - w/2, sot_pen, width=w, color=_SOT_COLOR, label="DES-SOT", edgecolor="white")
    ax.bar(x + w/2, ga_pen,  width=w, color=_GA_COLOR,  label="DES-GA",  edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"SC-2\n(×{SC2_WEIGHT})", f"SC-3\n(×{SC3_WEIGHT})"])
    ax.set_title("Weighted SC Contributions to Fitness")
    ax.set_ylabel("Hours added to fitness")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    max_pen = max(max(sot_pen), max(ga_pen), 1)
    ax.set_ylim(0, max_pen * 1.2)

    # Panel 3 – worker allocation
    ax = axes[2]
    x  = np.arange(2)
    sot_w = [sot_m["avg_B"], sot_m["avg_C"]]
    ga_w  = [ga_m["avg_B"],  ga_m["avg_C"]]
    ax.bar(x - w/2, sot_w, width=w, color=_SOT_COLOR, label="DES-SOT", edgecolor="white")
    ax.bar(x + w/2, ga_w,  width=w, color=_GA_COLOR,  label="DES-GA",  edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(["Avg B workers\nper job", "Avg C workers\nper job"])
    ax.set_title("Average Worker Allocation per Job")
    ax.set_ylabel("Workers")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(sot_w), max(ga_w)) * 1.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Overview chart saved -> {save_path}")
    plt.show()
    plt.close()


def plot_throughput(
    sot_jobs: list,
    ga_jobs:  list,
    sot_makespan: float,
    ga_makespan:  float,
    save_path: str,
) -> None:
    n_weeks = int(np.ceil(max(sot_makespan, ga_makespan) / WEEK_HOURS))
    week_labels = [f"Wk {i + 1}" for i in range(n_weeks)]

    def _weekly(jobs, n_wk):
        done = [j for j in jobs if j.oven_end is not None]
        counts = np.zeros(n_wk, dtype=int)
        for j in done:
            wk = min(int(j.oven_end / WEEK_HOURS), n_wk - 1)
            counts[wk] += 1
        return counts

    sot_c = _weekly(sot_jobs, n_weeks)
    ga_c  = _weekly(ga_jobs,  n_weeks)
    x     = np.arange(n_weeks)
    w     = 0.35

    fig, ax = plt.subplots(figsize=(max(10, n_weeks * 0.9), 5))
    b1 = ax.bar(x - w/2, sot_c, width=w, color=_SOT_COLOR, label="DES-SOT", edgecolor="white")
    b2 = ax.bar(x + w/2, ga_c,  width=w, color=_GA_COLOR,  label="DES-GA",  edgecolor="white")

    for bars in (b1, b2):
        for bar in bars:
            v = int(bar.get_height())
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + 0.1, str(v),
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(week_labels)
    ax.set_xlabel("Simulation week  (1 week = 144 h)")
    ax.set_ylabel("Blades completed")
    ax.set_title(
        f"Weekly Throughput Comparison  "
        f"(SOT makespan {sot_makespan:.0f} h  |  GA makespan {ga_makespan:.0f} h)"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(sot_c.max(), ga_c.max()) * 1.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Throughput chart saved    -> {save_path}")
    plt.show()
    plt.close()


def plot_workers(
    sot_log:      list,
    ga_log:       list,
    pool_B:       int,
    pool_C:       int,
    sot_makespan: float,
    ga_makespan:  float,
    save_path:    str,
) -> None:
    def _steps(log, col_idx, makespan):
        if not log:
            return [0, makespan], [0, 0]
        ts = [e[0] for e in log] + [makespan]
        vs = [e[col_idx] for e in log] + [0]
        return ts, vs

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    fig.suptitle("Worker Utilisation Over Time: DES-SOT vs DES-GA",
                 fontsize=12, fontweight="bold")

    specs = [
        (axes[0], 1, pool_B, "B", "B worker pool"),
        (axes[1], 2, pool_C, "C", "C worker pool"),
    ]
    for ax, idx, pool, letter, pool_label in specs:
        sot_ts, sot_vs = _steps(sot_log, idx, sot_makespan)
        ga_ts,  ga_vs  = _steps(ga_log,  idx, ga_makespan)
        ax.step(sot_ts, sot_vs, where="post",
                color=_SOT_COLOR, linewidth=1.8, label="DES-SOT", alpha=0.9)
        ax.step(ga_ts,  ga_vs,  where="post",
                color=_GA_COLOR,  linewidth=1.8, label="DES-GA",  alpha=0.9)
        ax.axhline(pool, color="red", linestyle="--", linewidth=1.2,
                   label=f"{pool_label} ({pool})")
        ax.set_ylabel(f"{letter} workers in use")
        ax.set_title(f"{letter} Worker Utilisation")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, pool * 1.2)

    axes[1].set_xlabel("Hours")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Worker utilisation chart saved -> {save_path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare DES-SOT vs DES-GA on the same blade repair job set."
    )
    parser.add_argument("--workers_b",   type=int, default=N_WORKERS_B,
                        help=f"B worker pool size (default: {N_WORKERS_B})")
    parser.add_argument("--workers_c",   type=int, default=N_WORKERS_C,
                        help=f"C worker pool size (default: {N_WORKERS_C})")
    parser.add_argument("--jobs",        type=int, default=N_JOBS,
                        help=f"Number of jobs to simulate (default: {N_JOBS})")
    parser.add_argument("--no_delays",   action="store_true",
                        help="Disable Monte Carlo PERT delays (deterministic run)")
    parser.add_argument("--delay_seed",  type=int, default=SOT_DELAY_SEED,
                        help=f"RNG seed for MC delay sampling (default: {SOT_DELAY_SEED})")
    parser.add_argument("--generations", type=int, default=GA_GENERATIONS,
                        help=f"GA generations (default: {GA_GENERATIONS})")
    parser.add_argument("--popsize",     type=int, default=GA_POP_SIZE,
                        help=f"GA population size (default: {GA_POP_SIZE})")
    parser.add_argument("--no_ga",       action="store_true",
                        help="Skip GA optimisation — use a uniform chromosome instead "
                             "(faster, for quick tests)")
    args = parser.parse_args()

    pool_B     = args.workers_b
    pool_C     = args.workers_c
    n_jobs     = args.jobs
    use_delays = not args.no_delays

    # ── Load jobs (separate copies for each simulation) ──────────────────────
    sot_jobs = load_sot_jobs()[:n_jobs]
    ga_jobs  = load_ga_jobs()[:n_jobs]
    n        = len(sot_jobs)

    print(f"\nLoaded {n} jobs  |  B workers: {pool_B}  |  C workers: {pool_C}")
    ests = [j.estimated_h for j in sot_jobs]
    print(f"  min est={min(ests):.1f} h  max est={max(ests):.1f} h  avg est={sum(ests)/n:.1f} h")
    print(f"  MC delays : {'ON' if use_delays else 'OFF'}  seed={args.delay_seed}")

    # ── 1. DES-SOT ───────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("Running DES-SOT ...")
    sot_log, _, sot_sc2, sot_sc3 = simulate_des_sot_bc(
        sot_jobs, pool_B=pool_B, pool_C=pool_C,
        use_delays=use_delays, delay_seed=args.delay_seed,
    )
    sot_done    = [j for j in sot_jobs if j.oven_end is not None]
    sot_makespan = max(j.oven_end for j in sot_done) if sot_done else 0.0
    sot_n_gen   = sum(1 for j in sot_done if j.job_id.startswith("GEN-"))
    print(f"  DES-SOT makespan: {sot_makespan:.1f} h  ({len(sot_done)} jobs done: "
          f"{n} csv + {sot_n_gen} generated / max {MAX_GENERATED_JOBS})")

    # ── 2. DES-GA ────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    if args.no_ga:
        print("Skipping GA — using uniform chromosome (1 worker per job slot).")
        best_chrom = [1] * (2 * n)
    else:
        print(f"Running GA (pop={args.popsize}, gen={args.generations}) ...")
        ga_template = load_ga_jobs()[:n_jobs]
        best_chrom, best_fit, _ = run_ga(
            ga_template,
            pool_B=pool_B, pool_C=pool_C,
            pop_size=args.popsize, n_gen=args.generations,
        )
        print(f"  Best GA fitness: {best_fit:.1f}")

    # Fresh job copies for the final DES-GA run
    ga_jobs = load_ga_jobs()[:n_jobs]
    print("Running DES-GA (with SA worker refinement) ...")
    ga_log, _, ga_sc2, ga_sc3 = simulate_des_bc(
        ga_jobs, best_chrom, pool_B, pool_C,
        use_sa=True, use_lookahead=True,
        use_delays=use_delays, delay_seed=args.delay_seed,
    )
    ga_done     = [j for j in ga_jobs if j.oven_end is not None]
    ga_makespan = max(j.oven_end for j in ga_done) if ga_done else 0.0
    ga_n_gen    = sum(1 for j in ga_done if j.job_id.startswith("GEN-"))
    print(f"  DES-GA  makespan: {ga_makespan:.1f} h  ({len(ga_done)} jobs done: "
          f"{n} csv + {ga_n_gen} generated / max {MAX_GENERATED_JOBS})")

    # ── 3. Collect metrics ───────────────────────────────────────────────────
    sot_m   = _extract_metrics(sot_jobs, "DES-SOT")
    ga_m    = _extract_metrics(ga_jobs,  "DES-GA")
    n_total = max(len(sot_done), len(ga_done))

    # ── 4. Console table ─────────────────────────────────────────────────────
    print_comparison(sot_m, ga_m, sot_sc2, sot_sc3, ga_sc2, ga_sc3)

    # ── 5. Plots ─────────────────────────────────────────────────────────────
    print("Saving comparison plots ...")
    plot_overview(
        sot_m, ga_m,
        sot_sc2, sot_sc3,
        ga_sc2,  ga_sc3,
        save_path=f"comparison_overview_{n_total}jobs.png",
    )
    plot_throughput(
        sot_jobs, ga_jobs, sot_makespan, ga_makespan,
        save_path=f"comparison_throughput_{n_total}jobs.png",
    )
    plot_workers(
        sot_log, ga_log, pool_B, pool_C, sot_makespan, ga_makespan,
        save_path=f"comparison_workers_{n_total}jobs.png",
    )


if __name__ == "__main__":
    main()
