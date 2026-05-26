"""
des_ga_change_baseline.py
=========================
DES blade repair simulation using GA chromosome + SA worker refinement
on a fixed set of jobs loaded from data_real.csv (Estimated column),
matching the same 50-job baseline used by des_sot_baseline.py.

No buffer jobs, no PERT-generated jobs.  GA finds the optimal dispatch
order and initial worker targets; SA refines worker allocation at every
WORKER_REASSIGN_INTERVAL epoch; Monte Carlo PERT delays are applied to
phase 2 of each blade.

Relationship to other scripts
------------------------------
  des_sot_baseline.py         -- same fixed jobs, SOT heuristic (no GA/SA)
  des_simulation_SA_change.py -- GA+SA+DES but loads from current_status.csv
                                  and uses buffer / PERT-generated jobs
  des_ga_change_baseline.py   -- GA+SA+DES on the same fixed data_real.csv
                                  baseline (no buffer, no generated jobs)
"""
from __future__ import annotations

import argparse
import csv
import os

import ga_simulationBC_current as _ga_mod
from ga_simulationBC_current import (
    Job,
    _fresh_jobs,
    run_ga,
    plot_convergence,
    N_WORKERS_B,
    N_WORKERS_C,
    SC1_WEIGHT,
    SC2_WEIGHT,
    SC3_WEIGHT,
    GA_POP_SIZE,
    GA_GENERATIONS,
    DATA_REAL_CSV,
)
from des_simulation_SA_change import (
    DELAY_SEED,
    simulate_des_bc,
    evaluate_des,
    plot_gantt_des,
    plot_weekly_throughput,
    _export_csv,
    plot_sa_convergence_des,
    plot_worker_distribution,
)
from constraintsBC import MAX_REPAIR_TRACKS, MIN_PROCESS_TIME_RATIO

# Disable buffer replenishment in both the GA fitness evaluator and the DES
# engine -- this module always operates on a fixed primary-only job set.
_ga_mod.MAX_GENERATED_JOBS = 0

N_JOBS = 50   # rows to load from data_real.csv (same as des_sot_baseline)


# ---------------------------------------------------------------------------
# Job loader  (data_real.csv -> primary-only ga_simulationBC_current.Job list)
# ---------------------------------------------------------------------------

def load_jobs_baseline(
    csv_path: str = DATA_REAL_CSV,
    n_jobs:   int = N_JOBS,
) -> list[Job]:
    """Load the first n_jobs Estimated durations from data_real.csv.

    All jobs are created as primary (is_buffered=False).
    The first MAX_REPAIR_TRACKS jobs have skip_phase1=True (already on tracks).
    """
    jobs: list[Job] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for i, row in enumerate(reader):
            if i >= n_jobs:
                break
            estimated = float(row["Estimated"])
            jobs.append(Job(
                job_id      = f"JOB-{i + 1:03d}",
                estimated_h = estimated,
                job_index   = i,
                is_buffered = False,
                skip_phase1 = i < MAX_REPAIR_TRACKS,
            ))
    return jobs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DES blade repair simulation (GA + SA, fixed data_real.csv jobs)"
    )
    parser.add_argument("--workers_b",   type=int, default=N_WORKERS_B,
                        help=f"B worker pool (default: {N_WORKERS_B})")
    parser.add_argument("--workers_c",   type=int, default=N_WORKERS_C,
                        help=f"C worker pool (default: {N_WORKERS_C})")
    parser.add_argument("--jobs",        type=int, default=None,
                        help=f"Number of jobs to load from data_real.csv "
                             f"(default: {N_JOBS})")
    parser.add_argument("--generations", type=int, default=GA_GENERATIONS,
                        help=f"GA generations (default: {GA_GENERATIONS})")
    parser.add_argument("--popsize",     type=int, default=GA_POP_SIZE,
                        help=f"GA population size (default: {GA_POP_SIZE})")
    parser.add_argument("--no_delays",   action="store_true",
                        help="Disable Monte Carlo PERT delays (deterministic run)")
    parser.add_argument("--delay_seed",  type=int, default=DELAY_SEED,
                        help=f"RNG seed for delay sampling (default: {DELAY_SEED})")
    parser.add_argument("--compare",     action="store_true",
                        help="Also run original time-stepped GA sim for comparison")
    args = parser.parse_args()

    pool_B = args.workers_b
    pool_C = args.workers_c
    n_jobs = args.jobs if args.jobs is not None else N_JOBS

    # ------------------------------------------------------------------
    # 1. Load jobs
    # ------------------------------------------------------------------
    jobs_template = load_jobs_baseline(n_jobs=n_jobs)
    n = len(jobs_template)
    ests = [j.estimated_h for j in jobs_template]

    print(f"Loaded {n} jobs from data_real.csv  (all primary, no buffer)")
    print(f"  min/avg/max estimated : "
          f"{min(ests):.1f} / {sum(ests)/n:.1f} / {max(ests):.1f} h")
    print(f"  B workers: {pool_B}  |  C workers: {pool_C}")

    # ------------------------------------------------------------------
    # 2. GA optimisation
    # ------------------------------------------------------------------
    print(f"\nRunning GA  (pop={args.popsize}, gen={args.generations}) ...")
    best_chrom, best_fit_ga, history = run_ga(
        jobs_template,
        pool_B=pool_B, pool_C=pool_C,
        pop_size=args.popsize, n_gen=args.generations,
    )
    print(f"Best GA fitness : {best_fit_ga:.1f}")

    # ------------------------------------------------------------------
    # 3. Final DES run with SA worker refinement + PERT delays
    # ------------------------------------------------------------------
    use_delays = not args.no_delays
    print(f"\nRunning final DES with SA worker refinement every "
          f"{int(_ga_mod.WORKER_REASSIGN_INTERVAL)} h ...")
    print(f"  Monte Carlo delays : {'ON' if use_delays else 'OFF'}  "
          f"(seed={args.delay_seed})")

    des_jobs = load_jobs_baseline(n_jobs=n_jobs)
    sa_hist: list = []
    staff_log, sc1, sc2, sc3 = simulate_des_bc(
        des_jobs, best_chrom, pool_B, pool_C,
        use_sa=True, use_lookahead=True,
        use_delays=use_delays, delay_seed=args.delay_seed,
        sa_history_out=sa_hist,
    )

    # ------------------------------------------------------------------
    # 4. Results
    # ------------------------------------------------------------------
    done_des = [j for j in des_jobs if j.oven_end is not None]
    print(f"\n=== DES results ===")
    print(f"Completed  : {len(done_des)}/{n} jobs")

    if done_des:
        makespan_des = max(j.oven_end for j in done_des)
        avg_B        = sum(j.avg_workers_B for j in done_des) / len(done_des)
        avg_C        = sum(j.avg_workers_C for j in done_des) / len(done_des)
        fitness_des  = makespan_des + SC2_WEIGHT * sc2 + SC3_WEIGHT * sc3
        print(f"Makespan          : {makespan_des:.1f} h")
        print(f"Avg B workers/job : {avg_B:.2f}")
        print(f"Avg C workers/job : {avg_C:.2f}")
        print(f"SC-1 penalty      : {sc1:.1f}")
        print(f"SC-2 penalty      : {sc2:.1f}")
        print(f"SC-3 penalty      : {sc3:.1f}")
        print(f"Total fitness     : {fitness_des:.1f}")

        ratios     = [(j.repair_end - j.repair_start) / j.estimated_h
                      for j in done_des if j.repair_end is not None]
        violations = sum(1 for r in ratios if r < MIN_PROCESS_TIME_RATIO - 1e-6)
        print(f"HC-2 check        : min ratio = {min(ratios):.4f}  "
              f"violations = {violations}")

    # ------------------------------------------------------------------
    # 5. Optional comparison against time-stepped GA sim
    # ------------------------------------------------------------------
    if args.compare:
        from ga_simulationBC_current import simulate_ga_bc
        print("\nRunning original time-stepped GA sim for comparison ...")
        ga_jobs = load_jobs_baseline(n_jobs=n_jobs)
        staff_log_ga, sc1_ga, sc2_ga, sc3_ga = simulate_ga_bc(
            ga_jobs, best_chrom, pool_B, pool_C,
            use_sa=True, use_lookahead=True,
        )
        done_ga = [j for j in ga_jobs if j.oven_end is not None]
        if done_ga:
            makespan_ga = max(j.oven_end for j in done_ga)
            fitness_ga  = makespan_ga + SC2_WEIGHT * sc2_ga + SC3_WEIGHT * sc3_ga
            print(f"\n=== GA sim results (time-stepped) ===")
            print(f"Makespan  : {makespan_ga:.1f} h")
            print(f"Fitness   : {fitness_ga:.1f}")
            if done_des:
                print(f"\nDES improvement : "
                      f"{fitness_ga - fitness_des:+.1f}  "
                      f"({'better' if fitness_des < fitness_ga else 'worse or equal'})")

    # ------------------------------------------------------------------
    # 6. Plots and CSV export
    # ------------------------------------------------------------------
    if done_des:
        _export_csv(
            des_jobs, makespan_des, sc1, sc2, sc3,
            pool_B=pool_B, pool_C=pool_C,
            use_delays=use_delays,
            save_path=f"des_ga_baseline_results_{n}jobs.csv",
        )
        plot_weekly_throughput(
            des_jobs, makespan_des,
            save_path=f"des_ga_baseline_weekly_throughput_{n}jobs.png",
        )

    plot_gantt_des(
        des_jobs, staff_log, pool_B=pool_B, pool_C=pool_C,
        save_path=f"des_ga_baseline_gantt_{n}jobs.png",
    )
    plot_convergence(history,
                     save_path=f"des_ga_baseline_convergence_{n}jobs.png")
    plot_sa_convergence_des(
        sa_hist,
        save_path=f"des_ga_baseline_sa_convergence_{n}jobs.png",
    )
    plot_worker_distribution(
        des_jobs, pool_B=pool_B, pool_C=pool_C,
        save_path=f"des_ga_baseline_worker_dist_{n}jobs.png",
    )
