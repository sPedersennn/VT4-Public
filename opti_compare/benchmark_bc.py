"""
benchmark_bc.py
===============
Four-stage benchmark for the BC blade-repair scheduling model.

Compares five optimizers plus the SOT baseline on the B/C worker-split model
(Phase 1: C workers 9 h → Phase 2: B workers variable → Phase 3: C workers 27 h).

All optimizers minimise the same fitness function:
    fitness = makespan + SC2_WEIGHT * oven_wait_h
                       + SC3_WEIGHT * oven_idle_h

Chromosome encoding  (length 3 × N):
    [prio_0 … prio_{N-1}]   dispatch priority ∈ [0, 2N-1]
    [B_0 … B_{N-1}]         target B-workers  ∈ [0, MAX_WORKERS_PER_BLADE]
    [C_0 … C_{N-1}]         target C-workers  ∈ [0, MAX_WORKERS_PER_BLADE]

All optimisers are implemented as standalone code within this file.
The GA wraps the existing run_ga() from ga_simulationBC.py; SA, PSO, Greedy,
and Random Search are self-contained implementations below.

Stages
------
  Stage 1  — Correctness checks (each optimizer beats SOT baseline)
  Stage 2  — Hyperparameter tuning (grid search over GA, SA, PSO)
  Stage 21 — Budget sensitivity sweep (diminishing-returns analysis)
  Stage 3  — Head-to-head comparison (boxplots, convergence, Wilcoxon, Pareto)

Usage
-----
    python benchmark_bc.py                    # all stages
    python benchmark_bc.py --stage 1          # correctness only
    python benchmark_bc.py --stage 2          # tuning only
    python benchmark_bc.py --stage 21         # budget sweep only
    python benchmark_bc.py --stage 3          # comparison only
    python benchmark_bc.py --skip-tuning      # skip tuning, use defaults in stage 3
    python benchmark_bc.py --budget 3000      # normalise to 3000 evaluations/run
    python benchmark_bc.py --n-workers 1      # disable parallelism
    python benchmark_bc.py --n-jobs 30        # use first 30 jobs from data.csv
    --stage 4 #graphs
    --stage 5 #DES comparison
"""

from __future__ import annotations

import argparse
import copy
import itertools
import math
import os
import random as py_random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon as _wilcoxon, ttest_rel as _ttest_rel

# ── BC-model imports ──────────────────────────────────────────────────────────
_BC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BC_DIR not in sys.path:
    sys.path.insert(0, _BC_DIR)

from ga_simulationBC import (
    Job as _BCJob,
    load_jobs      as _load_jobs_bc,
    _fresh_jobs    as _bc_fresh_jobs,
    random_chromosome as _bc_random_chromosome,
    two_point_crossover as _bc_two_point_cx,
    mutate              as _bc_mutate,
    tournament_select   as _bc_tournament,
    simulate_ga_bc,
    N_WORKERS_B, N_WORKERS_C,
    GA_POP_SIZE, GA_GENERATIONS,
    GA_CROSSOVER_RATE, GA_MUTATION_RATE,
    GA_ELITISM_N, GA_TOURNAMENT_K, GA_SEED,
    SC2_WEIGHT, SC3_WEIGHT,
    PHASE1_H, PHASE3_H,
)
try:
    from des_simulationBC import simulate_des_bc as _simulate_des_bc
    _DES_AVAILABLE = True
except ImportError:
    _DES_AVAILABLE = False

from constraintsBC import (
    MAX_WORKERS_PER_BLADE,
    MAX_WEEKLY_HOURS,
    OVEN_PROCESS_TIME,
    CHANGEOVER_TIME,
)

WEEK_HOURS: float = MAX_WEEKLY_HOURS


# =============================================================================
# USER CONFIG  — edit these values for "just click run" behaviour.
# All settings can also be overridden via CLI flags (see --help).
# =============================================================================

# -- Data ---------------------------------------------------------------------
CFG_N_JOBS        = None   # None = load all jobs from data.csv; int = first N only

# -- Worker pools -------------------------------------------------------------
CFG_POOL_B        = N_WORKERS_B   # B workers (repair phase)
CFG_POOL_C        = N_WORKERS_C   # C workers (inspection phases)

# -- Dataset sampling ---------------------------------------------------------
CFG_N_DATASETS        = 10   # Stage 3: random subsets to average over (1 = full dataset)
CFG_SUBSET_SIZE       = 50   # jobs per random subset
CFG_TUNING_N_DATASETS = 3    # Stage 2: subsets averaged during hyperparameter tuning
CFG_DATASET_SEED      = 42    # base RNG seed for subset sampling (subset i → seed+i)

# -- Optimizer seeds ----------------------------------------------------------
CFG_EVAL_SEEDS    = list(range(1,  26))   # Stage 3 evaluation seeds  (1–25)
CFG_TUNING_SEEDS  = list(range(26, 31))   # Stage 2 tuning seeds      (26–30)

# -- Parallelism --------------------------------------------------------------
CFG_N_WORKERS     = max(1, (os.cpu_count() or 1) - 1)   # parallel processes


# =============================================================================
# CONFIG DATACLASSES
# =============================================================================

@dataclass
class BCGAConfig:
    population_size: int   = 45
    n_generations:   int   = GA_GENERATIONS
    crossover_rate:  float = 0.5
    mutation_rate:   float = 0.5
    elitism_n:       int   = GA_ELITISM_N
    tournament_k:    int   = GA_TOURNAMENT_K
    seed:            int   = GA_SEED


@dataclass
class BCSAConfig:
    n_iterations:    int   = 2000
    temp_start:      float = 1000.0
    temp_end:        float = 0.1
    cooling_rate:    float = 0.99
    mutations_per_iter: int = 3
    seed:            int   = 42


@dataclass
class BCPSOConfig:
    n_particles:  int   = 60
    n_iterations: int   = 100
    w_inertia:    float = 0.3
    c_personal:   float = 1.0
    c_global:     float = 1.5
    seed:         int   = 42


@dataclass
class BCRSConfig:
    n_samples: int = 2000
    seed:      int = 42


# =============================================================================
# HYPERPARAMETER GRIDS  (Stage 2 tuning)
# =============================================================================

def _build_grid(param_lists: dict) -> list[dict]:
    keys   = list(param_lists.keys())
    combos = itertools.product(*[param_lists[k] for k in keys])
    return [dict(zip(keys, combo)) for combo in combos]


GA_GRID: list[dict] = _build_grid({
    "population_size": [30, 45, 60],
    "mutation_rate":   [0.05, 0.1, 0.20],
    "crossover_rate":  [0.25, 0.50, 0.75],
})

SA_GRID: list[dict] = _build_grid({
    "temp_start":         [500, 1000, 2500],
    "temp_end":           [0.1, 0.5, 1.0],
    "mutations_per_iter": [1, 2, 3],
})

PSO_GRID: list[dict] = _build_grid({
    "n_particles": [20, 40, 60],
    "w_inertia":   [0.3, 0.6, 0.9],
    "c_personal":  [1.0, 1.5, 2.0],
    "c_global":    [1.5, 2.0, 2.5],
})

# Reduced budgets during Stage 2 (fast tuning)
_TUNING_BUDGET: dict[str, dict] = {
    "ga":     {"n_generations": 40},
    "sa":     {"n_iterations":  500},
    "pso":    {"n_iterations":  30},
    "random": {"n_samples":     500},
}

# Budget sweep values for Stage 21
BUDGET_GRID: dict[str, list[int]] = {
    "ga_n_generations":  [20, 50, 100, 150, 200],
    "sa_n_iterations":   [500, 1000, 2000, 3500, 5000],
    "pso_n_iterations":  [20, 50, 100, 150, 200],
    "random_n_samples":  [500, 1000, 2000, 5000, 10000],
}

_ALPHA = 0.05   # significance level before Bonferroni correction


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class BCResult:
    best_chromosome: list[int]
    best_fitness:    float
    history:         list[float]
    elapsed_s:       float = 0.0


# =============================================================================
# HELPERS
# =============================================================================

def _sample_jobs_bc(all_jobs: list[_BCJob], n: int, seed: int = 0) -> list[_BCJob]:
    """Draw n jobs at random; reassign job_index 0..n-1 so evaluate() works."""
    rng = np.random.default_rng(seed)
    n   = min(n, len(all_jobs))
    idx = sorted(rng.choice(len(all_jobs), size=n, replace=False).tolist())
    return [
        _BCJob(job_id=all_jobs[i].job_id, estimated_h=all_jobs[i].estimated_h,
               job_index=k)
        for k, i in enumerate(idx)
    ]


def _sot_chromosome(jobs: list[_BCJob], pool_B: int, pool_C: int) -> list[int]:
    """
    Warm-start chromosome equivalent to SOT dispatch with equal worker split.
    Priorities are assigned so the shortest job gets the lowest priority number
    (dispatched first).  Workers default to an equal split across 12 tracks.
    """
    n = len(jobs)
    sorted_by_h = sorted(range(n), key=lambda i: jobs[i].estimated_h)
    prios = [0] * n
    for rank, job_idx in enumerate(sorted_by_h):
        prios[job_idx] = rank

    n_tracks    = 12  # N_REPAIR_TRACKS
    b_per       = min(MAX_WORKERS_PER_BLADE, max(1, pool_B // n_tracks))
    c_per       = min(MAX_WORKERS_PER_BLADE, max(1, pool_C // n_tracks))
    b_workers   = [b_per] * n
    c_workers   = [c_per] * n
    return prios + b_workers + c_workers


def _bc_eval(chromosome: list[int], jobs: list[_BCJob],
             pool_B: int, pool_C: int) -> float:
    fresh = _bc_fresh_jobs(jobs)
    _, _sc1, sc2, sc3 = simulate_ga_bc(fresh, chromosome, pool_B, pool_C)
    done     = [j for j in fresh if j.oven_end is not None]
    makespan = max((j.oven_end for j in done), default=float("inf"))
    return makespan + SC2_WEIGHT * sc2 + SC3_WEIGHT * sc3


def _baseline_cost(jobs: list[_BCJob], pool_B: int, pool_C: int) -> float:
    """Evaluate the SOT warm-start chromosome as the baseline cost."""
    chrom = _sot_chromosome(jobs, pool_B, pool_C)
    return _bc_eval(chrom, jobs, pool_B, pool_C)


def _schedule_metrics(chromosome: list[int], jobs: list[_BCJob],
                      pool_B: int, pool_C: int) -> dict:
    """Run the GA simulation and collect schedule metrics."""
    fresh = _bc_fresh_jobs(jobs)
    _, _sc1, sc2, sc3 = simulate_ga_bc(fresh, chromosome, pool_B, pool_C)
    done = [j for j in fresh if j.oven_end is not None]
    if not done:
        return {}
    makespan = max(j.oven_end for j in done)

    # Oven idle time (SC-3 source)
    oven_jobs  = sorted(done, key=lambda j: j.oven_start or 0.0)
    oven_idle  = 0.0
    prev_end   = 0.0
    for j in oven_jobs:
        if j.oven_start is not None and j.oven_start > prev_end:
            oven_idle += j.oven_start - prev_end
        if j.oven_end is not None:
            prev_end = max(prev_end, j.oven_end)

    avg_B = sum(j.avg_workers_B for j in done) / len(done)
    avg_C = sum(j.avg_workers_C for j in done) / len(done)

    return {
        "n_jobs":      len(done),
        "makespan_h":  round(makespan, 1),
        "oven_idle_h": round(oven_idle, 2),
        "oven_wait_h": round(sc2, 2),
        "avg_B":       round(avg_B, 2),
        "avg_C":       round(avg_C, 2),
    }


def _apply_params(config, params: dict):
    cfg = copy.copy(config)
    for k, v in params.items():
        setattr(cfg, k, v)
    return cfg


def _apply_budget(name: str, config, budget: int | None):
    """Scale config so the optimizer uses ~budget objective evaluations."""
    if budget is None or name == "greedy":
        return config
    cfg = copy.copy(config)
    if name == "ga":
        pop = getattr(cfg, "population_size", 60)
        cfg.n_generations = max(10, budget // pop - 1)
    elif name == "sa":
        cfg.n_iterations = max(100, budget - 1)
    elif name == "pso":
        n_p = getattr(cfg, "n_particles", 30)
        cfg.n_iterations = max(10, budget // n_p - 1)
    elif name == "random":
        cfg.n_samples = max(100, budget - 1)
    return cfg


def _eval_counts(name: str, history_len: int, config) -> np.ndarray:
    """Cumulative objective evaluations at each history checkpoint."""
    if name == "ga":
        pop = getattr(config, "population_size", 60)
        return np.array([pop * (i + 1) for i in range(history_len)])
    if name == "pso":
        n_p = getattr(config, "n_particles", 30)
        return np.array([n_p * (i + 1) for i in range(history_len)])
    return np.arange(1, history_len + 1)


# =============================================================================
# STANDALONE OPTIMIZERS
# =============================================================================

# ── Genetic Algorithm ─────────────────────────────────────────────────────────

def run_ga_bc(jobs: list[_BCJob], config: BCGAConfig,
              pool_B: int = N_WORKERS_B,
              pool_C: int = N_WORKERS_C,
              verbose: bool = False) -> BCResult:
    """GA for BC model — wraps ga_simulationBC.run_ga with a warm-start SOT seed."""
    rng = py_random.Random(config.seed)
    n   = len(jobs)
    t0  = time.perf_counter()

    pop  = [_bc_random_chromosome(n, rng) for _ in range(config.population_size)]
    pop[0] = _sot_chromosome(jobs, pool_B, pool_C)
    fits = [_bc_eval(c, jobs, pool_B, pool_C) for c in pop]

    best_idx  = min(range(len(fits)), key=lambda i: fits[i])
    best, b_f = pop[best_idx][:], fits[best_idx]
    history   = [b_f]

    print_every = max(1, config.n_generations // 5) if verbose else 0

    for gen in range(config.n_generations):
        ranked   = sorted(range(config.population_size), key=lambda i: fits[i])
        next_pop = [pop[i][:] for i in ranked[:config.elitism_n]]

        while len(next_pop) < config.population_size:
            p1 = _bc_tournament(pop, fits, config.tournament_k, rng)
            p2 = _bc_tournament(pop, fits, config.tournament_k, rng)
            if rng.random() < config.crossover_rate:
                c1, c2 = _bc_two_point_cx(p1, p2, rng)
            else:
                c1, c2 = p1[:], p2[:]
            next_pop.append(_bc_mutate(c1, config.mutation_rate, rng))
            if len(next_pop) < config.population_size:
                next_pop.append(_bc_mutate(c2, config.mutation_rate, rng))

        pop  = next_pop
        fits = [_bc_eval(c, jobs, pool_B, pool_C) for c in pop]
        gen_best = min(fits)
        if gen_best < b_f:
            b_f  = gen_best
            best = pop[min(range(len(fits)), key=lambda i: fits[i])][:]
        history.append(b_f)

        if print_every and (gen + 1) % print_every == 0:
            pct = (gen + 1) * 100 // config.n_generations
            print(f"      GA  gen {gen+1:>4}/{config.n_generations}  ({pct:>3}%)  "
                  f"best={b_f:.2f}  elapsed={time.perf_counter()-t0:.1f}s")

    return BCResult(best, b_f, history, time.perf_counter() - t0)


# ── Simulated Annealing ────────────────────────────────────────────────────────

def _sa_mutate_bc(chrom: list[int], n: int, rng: py_random.Random) -> list[int]:
    """
    Neighbour operator for the BC chromosome.
    Alternates between mutating priorities and worker counts.
    """
    out = chrom[:]
    segment = rng.choice(["prio", "b", "c"])
    if segment == "prio":
        i, j = rng.sample(range(n), 2)
        out[i], out[j] = out[j], out[i]
    elif segment == "b":
        i = rng.randrange(n)
        delta = rng.choice([-1, 1])
        out[n + i] = max(0, min(MAX_WORKERS_PER_BLADE, out[n + i] + delta))
    else:
        i = rng.randrange(n)
        delta = rng.choice([-1, 1])
        out[2 * n + i] = max(0, min(MAX_WORKERS_PER_BLADE, out[2 * n + i] + delta))
    return out


def run_sa_bc(jobs: list[_BCJob], config: BCSAConfig,
              pool_B: int = N_WORKERS_B,
              pool_C: int = N_WORKERS_C,
              verbose: bool = False) -> BCResult:
    """Simulated Annealing for the BC chromosome.  Warm-starts from SOT."""
    rng = py_random.Random(config.seed)
    n   = len(jobs)
    t0  = time.perf_counter()

    current      = _sot_chromosome(jobs, pool_B, pool_C)
    current_f    = _bc_eval(current, jobs, pool_B, pool_C)
    best, best_f = current[:], current_f
    history      = [best_f]

    print_every = max(1, config.n_iterations // 5) if verbose else 0

    for it in range(config.n_iterations):
        progress  = it / max(config.n_iterations - 1, 1)
        temp      = config.temp_start * (1.0 - progress) + config.temp_end * progress
        candidate = current[:]
        for _ in range(config.mutations_per_iter):
            candidate = _sa_mutate_bc(candidate, n, rng)
        cand_f    = _bc_eval(candidate, jobs, pool_B, pool_C)
        delta     = cand_f - current_f
        if delta < 0 or (temp > 0 and rng.random() < math.exp(-delta / temp)):
            current, current_f = candidate, cand_f
        if current_f < best_f:
            best_f, best = current_f, current[:]
        history.append(best_f)

        if print_every and (it + 1) % print_every == 0:
            pct = (it + 1) * 100 // config.n_iterations
            print(f"      SA  iter {it+1:>5}/{config.n_iterations}  ({pct:>3}%)  "
                  f"T={temp:.2f}  best={best_f:.2f}  elapsed={time.perf_counter()-t0:.1f}s")

    return BCResult(best, best_f, history, time.perf_counter() - t0)


# ── Particle Swarm Optimization ───────────────────────────────────────────────

def run_pso_bc(jobs: list[_BCJob], config: BCPSOConfig,
               pool_B: int = N_WORKERS_B,
               pool_C: int = N_WORKERS_C,
               verbose: bool = False) -> BCResult:
    """PSO for the BC chromosome.  Particles are continuous; rounded for eval."""
    rng_py = py_random.Random(config.seed)
    rng_np = np.random.default_rng(config.seed)
    n      = len(jobs)
    d      = 3 * n   # chromosome dimension
    t0     = time.perf_counter()

    # Bounds: priorities [0, 2n-1], workers [0, MAX]
    lo = np.array([0.0] * n + [0.0] * (2 * n))
    hi = np.array([float(2 * n - 1)] * n + [float(MAX_WORKERS_PER_BLADE)] * (2 * n))

    # Initialise particles randomly
    pos = rng_np.uniform(lo, hi, size=(config.n_particles, d))
    vel = rng_np.uniform(-(hi - lo) / 4, (hi - lo) / 4,
                         size=(config.n_particles, d))

    # First particle = SOT warm start
    sot = _sot_chromosome(jobs, pool_B, pool_C)
    pos[0] = np.array([float(g) for g in sot], dtype=float)

    def _eval_pos(p: np.ndarray) -> float:
        chrom = [int(round(float(np.clip(p[i], lo[i], hi[i])))) for i in range(d)]
        return _bc_eval(chrom, jobs, pool_B, pool_C)

    pbest_pos  = pos.copy()
    pbest_fit  = np.array([_eval_pos(p) for p in pos])
    gbest_idx  = int(np.argmin(pbest_fit))
    gbest_pos  = pbest_pos[gbest_idx].copy()
    gbest_fit  = float(pbest_fit[gbest_idx])
    history    = [gbest_fit]

    print_every = max(1, config.n_iterations // 5) if verbose else 0

    for it in range(config.n_iterations):
        r1 = rng_np.random((config.n_particles, d))
        r2 = rng_np.random((config.n_particles, d))
        vel = (config.w_inertia * vel
               + config.c_personal * r1 * (pbest_pos - pos)
               + config.c_global   * r2 * (gbest_pos  - pos))
        v_max = (hi - lo) / 2
        vel   = np.clip(vel, -v_max, v_max)
        pos   = np.clip(pos + vel, lo, hi)

        fits = np.array([_eval_pos(p) for p in pos])
        improved = fits < pbest_fit
        pbest_pos[improved] = pos[improved].copy()
        pbest_fit[improved] = fits[improved]

        gi = int(np.argmin(pbest_fit))
        if pbest_fit[gi] < gbest_fit:
            gbest_fit = float(pbest_fit[gi])
            gbest_pos = pbest_pos[gi].copy()
        history.append(gbest_fit)

        if print_every and (it + 1) % print_every == 0:
            pct = (it + 1) * 100 // config.n_iterations
            print(f"      PSO iter {it+1:>4}/{config.n_iterations}  ({pct:>3}%)  "
                  f"best={gbest_fit:.2f}  elapsed={time.perf_counter()-t0:.1f}s")

    best_chrom = [int(round(float(np.clip(gbest_pos[i], lo[i], hi[i]))))
                  for i in range(d)]
    return BCResult(best_chrom, gbest_fit, history, time.perf_counter() - t0)


# ── Greedy (coordinate descent) ───────────────────────────────────────────────

def run_greedy_bc(jobs: list[_BCJob],
                  pool_B: int = N_WORKERS_B,
                  pool_C: int = N_WORKERS_C) -> BCResult:
    """
    Coordinate-descent greedy for the BC chromosome.

    Sweep order: B workers → C workers → priority swaps.
    Two sweeps total; deterministic (no random state).
    """
    n    = len(jobs)
    t0   = time.perf_counter()
    best = _sot_chromosome(jobs, pool_B, pool_C)
    b_f  = _bc_eval(best, jobs, pool_B, pool_C)
    history = [b_f]

    for _ in range(2):
        # B worker values for each job
        for i in range(n):
            for w in range(MAX_WORKERS_PER_BLADE + 1):
                if w == best[n + i]:
                    continue
                trial = best[:]
                trial[n + i] = w
                f = _bc_eval(trial, jobs, pool_B, pool_C)
                history.append(min(b_f, f))
                if f < b_f:
                    b_f, best = f, trial[:]

        # C worker values for each job
        for i in range(n):
            for w in range(MAX_WORKERS_PER_BLADE + 1):
                if w == best[2 * n + i]:
                    continue
                trial = best[:]
                trial[2 * n + i] = w
                f = _bc_eval(trial, jobs, pool_B, pool_C)
                history.append(min(b_f, f))
                if f < b_f:
                    b_f, best = f, trial[:]

        # Priority: try swapping every pair of jobs
        for i in range(n):
            for j in range(i + 1, n):
                trial = best[:]
                trial[i], trial[j] = trial[j], trial[i]
                f = _bc_eval(trial, jobs, pool_B, pool_C)
                history.append(min(b_f, f))
                if f < b_f:
                    b_f, best = f, trial[:]

    return BCResult(best, b_f, history, time.perf_counter() - t0)


# ── Random Search ─────────────────────────────────────────────────────────────

def run_random_bc(jobs: list[_BCJob], config: BCRSConfig,
                  pool_B: int = N_WORKERS_B,
                  pool_C: int = N_WORKERS_C) -> BCResult:
    """Pure Monte Carlo random search.  Warm-starts with SOT chromosome."""
    rng = py_random.Random(config.seed)
    n   = len(jobs)
    t0  = time.perf_counter()

    best  = _sot_chromosome(jobs, pool_B, pool_C)
    b_f   = _bc_eval(best, jobs, pool_B, pool_C)
    history = [b_f]

    for _ in range(config.n_samples):
        candidate = _bc_random_chromosome(n, rng)
        f = _bc_eval(candidate, jobs, pool_B, pool_C)
        if f < b_f:
            b_f, best = f, candidate[:]
        history.append(b_f)

    return BCResult(best, b_f, history, time.perf_counter() - t0)


# =============================================================================
# DISPATCHER
# =============================================================================

def _run_one(name: str, jobs: list[_BCJob], config, seed: int,
             pool_B: int = N_WORKERS_B, pool_C: int = N_WORKERS_C,
             verbose: bool = False) -> BCResult:
    """Dispatch to the correct optimizer.  Greedy ignores seed (deterministic)."""
    if name == "greedy":
        return run_greedy_bc(jobs, pool_B, pool_C)
    cfg = _apply_params(config, {"seed": seed})
    if name == "ga":     return run_ga_bc(jobs, cfg, pool_B, pool_C, verbose=verbose)
    if name == "sa":     return run_sa_bc(jobs, cfg, pool_B, pool_C, verbose=verbose)
    if name == "pso":    return run_pso_bc(jobs, cfg, pool_B, pool_C, verbose=verbose)
    if name == "random": return run_random_bc(jobs, cfg, pool_B, pool_C)
    raise ValueError(f"Unknown optimizer: {name!r}")


# =============================================================================
# PARALLEL WORKERS  (module-level so ProcessPoolExecutor can pickle them)
# =============================================================================

def _tune_worker(args: tuple) -> float:
    name, jobs, merged_params, seed, base_config, pool_B, pool_C = args
    try:
        cfg = _apply_params(base_config, {**merged_params, "seed": seed})
        r   = _run_one(name, jobs, cfg, seed, pool_B, pool_C)
        return r.best_fitness
    except Exception:
        return float("inf")


def _tune_budget_worker(args: tuple) -> tuple[float, float]:
    """Like _tune_worker but also returns wall-clock seconds for the run."""
    name, jobs, merged_params, seed, base_config, pool_B, pool_C = args
    try:
        t0  = time.perf_counter()
        cfg = _apply_params(base_config, {**merged_params, "seed": seed})
        r   = _run_one(name, jobs, cfg, seed, pool_B, pool_C)
        return r.best_fitness, time.perf_counter() - t0
    except Exception:
        return float("inf"), 0.0


def _comparison_worker(args: tuple):
    name, jobs, cfg, seed, pool_B, pool_C, verbose = args
    t0 = time.perf_counter()
    r  = _run_one(name, jobs, cfg, seed, pool_B, pool_C, verbose=verbose)
    return seed, r, time.perf_counter() - t0


# =============================================================================
# STAGE 1 — CORRECTNESS CHECKS
# =============================================================================

@dataclass
class SanityResult:
    name:           str
    cost:           float
    beats_baseline: bool
    n_finished:     int
    elapsed_s:      float
    error:          str = ""


_STATIC_NOTES = [
    ("GA",     "OK",
     "Two-point crossover, tournament selection, elitism.  "
     "Chromosome: [prio×N, B_workers×N, C_workers×N].  "
     "Warm-starts with SOT dispatch."),
    ("SA",     "OK",
     "Linear cooling T_start→T_end.  Alternating neighbour: prio swap, "
     "B±1, or C±1.  Warm-starts with SOT chromosome."),
    ("PSO",    "OK",
     "Standard inertia-weight PSO on continuous [0,1]-normalised genes, "
     "rounded to int for evaluation.  First particle = SOT warm start."),
    ("Greedy", "OK",
     "Coordinate descent: B workers → C workers → pairwise priority swaps. "
     "Two sweeps; deterministic (no random state)."),
    ("Random", "OK",
     "Pure Monte Carlo.  Statistical floor — any real optimizer must beat "
     "it convincingly."),
]


def run_sanity_checks(jobs: list[_BCJob], baseline_cost: float,
                      pool_B: int, pool_C: int) -> list[SanityResult]:
    print("\n" + "=" * 60)
    print("  Stage 1: Correctness checks")
    print("=" * 60)
    print("\n  Static code-review notes:")
    for opt, status, note in _STATIC_NOTES:
        sym = "✓" if status == "OK" else "⚠"
        print(f"  {sym}  {opt:<10}  {note}")

    configs = {
        "ga": BCGAConfig(), "sa": BCSAConfig(), "pso": BCPSOConfig(),
        "random": BCRSConfig(),
    }
    results: list[SanityResult] = []
    print(f"\n  Running each optimizer once (seed=42) ...")
    for name, cfg in configs.items():
        t0 = time.perf_counter()
        cost, n_fin, error = float("inf"), 0, ""
        try:
            r     = _run_one(name, jobs, cfg, seed=42, pool_B=pool_B, pool_C=pool_C)
            cost  = r.best_fitness
            n_fin = _schedule_metrics(r.best_chromosome, jobs, pool_B, pool_C).get("n_jobs", 0)
        except Exception as e:
            error = str(e)
        elapsed = time.perf_counter() - t0
        beats   = cost < baseline_cost and not error
        status  = "PASS" if beats else "FAIL"
        print(f"  [{status}] {name.upper():<8}  cost={cost:>10.2f}  "
              f"baseline={baseline_cost:.2f}  finished={n_fin}  ({elapsed:.1f}s)"
              + (f"  ERROR: {error}" if error else ""))
        results.append(SanityResult(name, cost, beats, n_fin, elapsed, error))

    n_pass = sum(1 for r in results if r.beats_baseline)
    print(f"\n  Result: {n_pass}/{len(results)} optimizers beat the baseline.")
    return results


# =============================================================================
# STAGE 2 — PARAMETER TUNING
# =============================================================================

@dataclass
class TuningResult:
    name:        str
    best_params: dict[str, Any]
    best_mean:   float
    best_std:    float
    all_records: list[dict] = field(default_factory=list)
    elapsed_s:   float = 0.0


def _tune_one(name: str, grid: list[dict], base_config,
              job_sets: list, seeds: list[int],
              pool_B: int, pool_C: int,
              n_workers: int = 1) -> TuningResult:
    budget       = _TUNING_BUDGET.get(name, {})
    n_per_config = len(job_sets) * len(seeds)
    total_tasks  = len(grid) * n_per_config
    print(f"\n  {name.upper()}: {len(grid)} configs × {len(seeds)} seeds × "
          f"{len(job_sets)} dataset(s) = {total_tasks} tasks  "
          f"(budget: {budget},  workers: {n_workers})")

    tasks = [
        (name, job_set, {**budget, **params}, seed, base_config, pool_B, pool_C)
        for params in grid
        for job_set in job_sets
        for seed in seeds
    ]

    t0          = time.perf_counter()
    report_every = max(1, len(tasks) // 10)

    if n_workers > 1:
        all_costs = [float("inf")] * len(tasks)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            future_map = {pool.submit(_tune_worker, t): i for i, t in enumerate(tasks)}
            done_count = 0
            for fut in as_completed(future_map):
                all_costs[future_map[fut]] = fut.result()
                done_count += 1
                if done_count % report_every == 0 or done_count == len(tasks):
                    pct = done_count * 100 // len(tasks)
                    print(f"    [{name.upper()} tuning] {done_count}/{len(tasks)} "
                          f"({pct}%)  elapsed={time.perf_counter()-t0:.0f}s")
    else:
        all_costs = []
        for i, t in enumerate(tasks):
            all_costs.append(_tune_worker(t))
            if (i + 1) % report_every == 0 or i + 1 == len(tasks):
                pct = (i + 1) * 100 // len(tasks)
                print(f"    [{name.upper()} tuning] {i+1}/{len(tasks)} "
                      f"({pct}%)  elapsed={time.perf_counter()-t0:.0f}s")

    records: list[dict] = []
    for i, params in enumerate(grid):
        costs  = all_costs[i * n_per_config : (i + 1) * n_per_config]
        mean_c = float(np.mean(costs))
        std_c  = float(np.std(costs))
        records.append({"params": params, "mean": mean_c, "std": std_c})

    records.sort(key=lambda x: x["mean"])
    best    = records[0]
    elapsed = time.perf_counter() - t0
    print(f"  → best params : {best['params']}")
    print(f"  → best mean   : {best['mean']:.2f} ± {best['std']:.2f}  ({elapsed:.0f}s)")
    return TuningResult(name, best["params"], best["mean"], best["std"],
                        records, elapsed)


def run_tuning(job_sets: list, seeds: list[int],
               pool_B: int, pool_C: int,
               n_workers: int = 1,
               tune_only: list[str] | None = None) -> dict[str, TuningResult]:
    _all_opts = [("ga", GA_GRID, BCGAConfig()), ("sa", SA_GRID, BCSAConfig()),
                 ("pso", PSO_GRID, BCPSOConfig())]
    opts = [(n, g, c) for n, g, c in _all_opts
            if tune_only is None or n in tune_only]
    print("\n" + "=" * 60)
    print("  Stage 2: Parameter tuning")
    print(f"  Tuning seeds   : {seeds}")
    for n, g, _ in opts:
        print(f"  {n.upper():<4} grid size  : {len(g)}")
    print(f"  Parallel workers: {n_workers}")
    print("=" * 60)
    results = {
        n: _tune_one(n, g, c, job_sets, seeds, pool_B, pool_C, n_workers)
        for n, g, c in opts
    }
    print("\n  Generating tuning plots ...")
    for _name, _tr in results.items():
        plot_tuning_heatmap(_name, _tr.all_records)
        plot_tuning_3d(_name, _tr.all_records)
    return results


# =============================================================================
# STAGE 21 — BUDGET SENSITIVITY SWEEP
# =============================================================================

@dataclass
class BudgetTuningResult:
    name:       str
    param:      str
    values:     list[int]
    means:      list[float]
    stds:       list[float]
    best_value: int
    elapsed_s:  float      = 0.0
    times:      list[float] = field(default_factory=list)  # mean wall-clock s per run per budget value


def _tune_budget_one(name: str, param: str, values: list[int],
                     best_params: dict, base_config,
                     job_sets: list, seeds: list[int],
                     pool_B: int, pool_C: int,
                     n_workers: int = 1) -> BudgetTuningResult:
    n_per_value = len(job_sets) * len(seeds)
    total_tasks = len(values) * n_per_value
    print(f"\n  {name.upper()} [{param}]: {len(values)} values × "
          f"{len(seeds)} seeds × {len(job_sets)} dataset(s) = {total_tasks} tasks")

    tasks = [
        (name, job_set, {**best_params, param: value}, seed, base_config, pool_B, pool_C)
        for value in values
        for job_set in job_sets
        for seed in seeds
    ]

    t0           = time.perf_counter()
    report_every = max(1, len(tasks) // 10)

    if n_workers > 1:
        all_costs = [float("inf")] * len(tasks)
        all_times = [0.0] * len(tasks)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            future_map = {pool.submit(_tune_budget_worker, t): i for i, t in enumerate(tasks)}
            done_count = 0
            for fut in as_completed(future_map):
                cost, run_t = fut.result()
                all_costs[future_map[fut]] = cost
                all_times[future_map[fut]] = run_t
                done_count += 1
                if done_count % report_every == 0 or done_count == len(tasks):
                    pct = done_count * 100 // len(tasks)
                    print(f"    [{name.upper()} budget] {done_count}/{len(tasks)} "
                          f"({pct}%)  elapsed={time.perf_counter()-t0:.0f}s")
    else:
        all_costs = []
        all_times = []
        for i, t in enumerate(tasks):
            cost, run_t = _tune_budget_worker(t)
            all_costs.append(cost)
            all_times.append(run_t)
            if (i + 1) % report_every == 0 or i + 1 == len(tasks):
                pct = (i + 1) * 100 // len(tasks)
                print(f"    [{name.upper()} budget] {i+1}/{len(tasks)} "
                      f"({pct}%)  elapsed={time.perf_counter()-t0:.0f}s")

    means, stds, times = [], [], []
    for i, value in enumerate(values):
        costs  = all_costs[i * n_per_value: (i + 1) * n_per_value]
        t_runs = all_times[i * n_per_value: (i + 1) * n_per_value]
        means.append(float(np.mean(costs)))
        stds.append(float(np.std(costs)))
        times.append(float(np.mean(t_runs)))
        print(f"    {param}={value:>7}:  mean={means[-1]:.2f} ± {stds[-1]:.2f}  "
              f"({times[-1]:.1f}s/run)")

    best_idx = int(np.argmin(means))
    elapsed  = time.perf_counter() - t0
    print(f"  → best {param}: {values[best_idx]}  "
          f"(mean={means[best_idx]:.2f} ± {stds[best_idx]:.2f},  {elapsed:.0f}s)")
    return BudgetTuningResult(name, param, values, means, stds,
                              values[best_idx], elapsed, times)


def run_budget_tuning(job_sets: list, seeds: list[int],
                      tuning: dict[str, TuningResult],
                      pool_B: int, pool_C: int,
                      n_workers: int = 1) -> dict[str, BudgetTuningResult]:
    print("\n" + "=" * 60)
    print("  Stage 2.1: Budget sensitivity sweep")
    print("=" * 60)

    def _shape(name: str, exclude: str) -> dict:
        if name not in tuning:
            return {}
        return {k: v for k, v in tuning[name].best_params.items() if k != exclude}

    results: dict[str, BudgetTuningResult] = {}
    results["ga"] = _tune_budget_one(
        "ga", "n_generations", BUDGET_GRID["ga_n_generations"],
        _shape("ga", "n_generations"), BCGAConfig(),
        job_sets, seeds, pool_B, pool_C, n_workers)
    results["sa"] = _tune_budget_one(
        "sa", "n_iterations", BUDGET_GRID["sa_n_iterations"],
        _shape("sa", "n_iterations"), BCSAConfig(),
        job_sets, seeds, pool_B, pool_C, n_workers)
    results["pso"] = _tune_budget_one(
        "pso", "n_iterations", BUDGET_GRID["pso_n_iterations"],
        _shape("pso", "n_iterations"), BCPSOConfig(),
        job_sets, seeds, pool_B, pool_C, n_workers)
    results["random"] = _tune_budget_one(
        "random", "n_samples", BUDGET_GRID["random_n_samples"],
        {}, BCRSConfig(), job_sets, seeds, pool_B, pool_C, n_workers)
    return results


# =============================================================================
# STAGE 3 — HEAD-TO-HEAD COMPARISON
# =============================================================================

@dataclass
class BenchmarkRun:
    name:             str
    config_desc:      str
    config_obj:       Any
    costs:            list[float]       = field(default_factory=list)
    times_s:          list[float]       = field(default_factory=list)
    histories:        list[list[float]] = field(default_factory=list)
    metrics:          list[dict]        = field(default_factory=list)
    best_chromosome:  list[int]         = field(default_factory=list)

    @property
    def mean_cost(self):  return float(np.mean(self.costs))  if self.costs else float("inf")
    @property
    def std_cost(self):   return float(np.std(self.costs))   if self.costs else 0.0
    @property
    def best_cost(self):  return float(np.min(self.costs))   if self.costs else float("inf")
    @property
    def mean_time(self):  return float(np.mean(self.times_s)) if self.times_s else 0.0


def _best_config(name: str, base_config, tuning: dict[str, TuningResult],
                 budget: int | None):
    if name in tuning:
        tr  = tuning[name]
        cfg = _apply_params(base_config, tr.best_params)
        desc = str(tr.best_params)
    else:
        cfg, desc = base_config, "default"
    cfg = _apply_budget(name, cfg, budget)
    return cfg, desc


def run_comparison(jobs: list[_BCJob], tuning: dict[str, TuningResult],
                   eval_seeds: list[int], bl_cost: float,
                   pool_B: int, pool_C: int,
                   budget: int | None = None,
                   n_workers: int = 1) -> list[BenchmarkRun]:
    print("\n" + "=" * 60)
    print("  Stage 3: Head-to-head comparison")
    print(f"  Evaluation seeds : {eval_seeds}")
    if budget:
        print(f"  Evaluation budget: {budget} objective evaluations per run")
    print("=" * 60)

    optimizers = [
        ("ga",     BCGAConfig()),
        ("sa",     BCSAConfig()),
        ("pso",    BCPSOConfig()),
        ("random", BCRSConfig()),
    ]

    runs: list[BenchmarkRun] = []
    for name, base_cfg in optimizers:
        cfg, desc = _best_config(name, base_cfg, tuning, budget)
        run = BenchmarkRun(name=name, config_desc=desc, config_obj=cfg)
        print(f"\n  [{name.upper()}]  config: {desc}")

        seeds_to_use = [eval_seeds[0]] if name == "greedy" else eval_seeds
        # Verbose per-iteration prints only make sense in single-worker mode
        verbose_run  = (n_workers == 1)
        tasks = [(name, jobs, cfg, seed, pool_B, pool_C, verbose_run)
                 for seed in seeds_to_use]

        if n_workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                raw = list(pool.map(_comparison_worker, tasks, chunksize=2))
        else:
            raw = [_comparison_worker(t) for t in tasks]

        for seed, r, elapsed in sorted(raw, key=lambda x: x[0]):
            m = _schedule_metrics(r.best_chromosome, jobs, pool_B, pool_C)
            run.costs.append(r.best_fitness)
            run.times_s.append(elapsed)
            run.histories.append(r.history)
            run.metrics.append(m)
            if not run.best_chromosome or r.best_fitness < min(run.costs[:-1], default=float("inf")):
                run.best_chromosome = r.best_chromosome[:]
            print(f"    seed={seed:3d}  cost={r.best_fitness:.2f}  time={elapsed:.1f}s"
                  + (f"  makespan={m['makespan_h']}h" if m else ""))

        if run.costs:
            if name == "greedy":
                print(f"  → cost={run.mean_cost:.2f} (deterministic)")
            else:
                pct  = (run.mean_cost - bl_cost) / bl_cost * 100
                sign = "+" if pct >= 0 else ""
                print(f"  → mean={run.mean_cost:.2f} ± {run.std_cost:.2f}  "
                      f"best={run.best_cost:.2f}  vs_baseline={sign}{pct:.1f}%")
        runs.append(run)
    return runs


# =============================================================================
# STATISTICAL SIGNIFICANCE
# =============================================================================

def _vargha_delaney_a12(x: np.ndarray, y: np.ndarray) -> float:
    m, n = len(x), len(y)
    less  = sum(xi < yj for xi in x for yj in y)
    equal = sum(xi == yj for xi in x for yj in y)
    return (less + 0.5 * equal) / (m * n)


def compute_significance_table(
    runs: list[BenchmarkRun],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.DataFrame, pd.DataFrame, float]:
    """
    Returns (wilcoxon_p_df, wilcoxon_sig_df, a12_df,
             ttest_p_df, ttest_sig_df, alpha_corr).

    Both Wilcoxon signed-rank and paired t-test are run on the same
    paired cost vectors.  Significance is Bonferroni-corrected.
    sig_df values: +1 = row is better (lower cost), -1 = row is worse, 0 = n.s.
    """
    stochastic = [br for br in runs if br.name != "greedy" and len(br.costs) > 1]
    n          = len(stochastic)
    n_pairs    = max(1, n * (n - 1) // 2)
    alpha_corr = _ALPHA / n_pairs

    names      = [br.name for br in stochastic]
    p_mat      = np.full((n, n), np.nan)   # Wilcoxon
    sig_mat    = np.zeros((n, n), dtype=int)
    a12_mat    = np.full((n, n), np.nan)
    tp_mat     = np.full((n, n), np.nan)   # paired t-test
    tsig_mat   = np.zeros((n, n), dtype=int)

    for i, br_i in enumerate(stochastic):
        for j, br_j in enumerate(stochastic):
            if i == j:
                continue
            ci = np.array(br_i.costs)
            cj = np.array(br_j.costs)
            if len(ci) != len(cj):
                continue
            a12_mat[i, j] = _vargha_delaney_a12(ci, cj)
            diffs = ci - cj
            direction = -1 if np.mean(ci) > np.mean(cj) else +1

            if np.all(diffs == 0):
                p_mat[i, j]  = 1.0
                tp_mat[i, j] = 1.0
                continue

            # Wilcoxon signed-rank
            try:
                _, p = _wilcoxon(ci, cj, zero_method="wilcox", alternative="two-sided")
                p_mat[i, j] = p
                if p < alpha_corr:
                    sig_mat[i, j] = direction
            except ValueError:
                pass

            # Paired t-test
            try:
                _, tp = _ttest_rel(ci, cj)
                tp_mat[i, j] = tp
                if tp < alpha_corr:
                    tsig_mat[i, j] = direction
            except Exception:
                pass

    p_df    = pd.DataFrame(p_mat,    index=names, columns=names)
    sig_df  = pd.DataFrame(sig_mat,  index=names, columns=names)
    a12_df  = pd.DataFrame(a12_mat,  index=names, columns=names)
    tp_df   = pd.DataFrame(tp_mat,   index=names, columns=names)
    tsig_df = pd.DataFrame(tsig_mat, index=names, columns=names)
    return p_df, sig_df, a12_df, tp_df, tsig_df, alpha_corr


# =============================================================================
# PLOTS
# =============================================================================

_COLORS = {
    "ga": "#1f77b4", "sa": "#ff7f0e", "pso": "#2ca02c",
    "greedy": "#9467bd", "random": "#d62728", "sot": "#000000",
}


def _draw_convergence(ax, entries: list[tuple], zoom: bool = False) -> None:
    """
    Shared convergence renderer.
    entries: list of (name, xs, mean_h, std_h, mean_cost, std_cost)
    mean line drawn as steps; ±1σ band shaded.
    """
    all_means = []
    for name, xs_ref, mean_h, std_h, mu, std in entries:
        c     = _COLORS.get(name, "gray")
        label = (f"{name.upper()}  μ={mu:.1f} ±{std:.1f}"
                 if std > 0 else f"{name.upper()}  {mu:.1f}  (det.)")
        ax.fill_between(xs_ref, mean_h - std_h, mean_h + std_h,
                        color=c, alpha=0.15, step="post", zorder=2)
        ax.plot(xs_ref, mean_h, color=c, linewidth=2.0, label=label,
                drawstyle="steps-post", zorder=3)
        all_means.append(mean_h)

    if zoom and all_means:
        all_vals = np.concatenate(all_means)
        margin   = (all_vals.max() - all_vals.min()) * 0.15 + 1e-6
        ax.set_ylim(all_vals.min() - margin, all_vals.max() + margin)

    ax.set_xlabel("Cumulative objective-function evaluations")
    ax.set_ylabel("Fitness  (lower is better)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)


def plot_convergence(runs: list[BenchmarkRun], bl_cost: float,
                     path: str = "bc_convergence.png",
                     show_baseline: bool = False,
                     zoom: bool = False):
    fig, ax = plt.subplots(figsize=(11, 5))
    if show_baseline:
        ax.axhline(bl_cost, color="black", linestyle=":", linewidth=1.5,
                   label=f"SOT baseline ({bl_cost:.1f})", zorder=4)

    entries = []
    for br in runs:
        if not br.histories:
            continue
        all_x   = [_eval_counts(br.name, len(h), br.config_obj) for h in br.histories]
        min_len = min(len(h) for h in br.histories)
        raw     = np.array([h[:min_len] for h in br.histories])
        xs_ref  = all_x[0][:min_len]
        mean_h  = raw.mean(axis=0)
        std_h   = raw.std(axis=0)
        n_runs  = len(br.histories)
        entries.append((br.name, xs_ref, mean_h, std_h, br.mean_cost,
                        br.std_cost if n_runs > 1 else 0.0))

    title_suffix = "zoomed on final values" if zoom else "full convergence"
    _draw_convergence(ax, entries, zoom=zoom)
    ax.set_title(f"BC-model optimizer convergence — {title_suffix}")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Convergence saved -> {path}")
    plt.close()


def plot_boxplots(runs: list[BenchmarkRun], bl_cost: float,
                  path: str = "bc_boxplots.png",
                  show_baseline: bool = False):
    stochastic = [br for br in runs if len(br.costs) > 1]
    det        = [br for br in runs if len(br.costs) == 1]

    fig, ax = plt.subplots(figsize=(9, 5))
    data   = [br.costs for br in stochastic]
    labels = [br.name.upper() for br in stochastic]
    colors = [_COLORS.get(br.name, "gray") for br in stochastic]

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    rng = np.random.default_rng(0)
    for k, (br, color) in enumerate(zip(stochastic, colors), start=1):
        jitter = rng.uniform(-0.15, 0.15, len(br.costs))
        ax.scatter(k + jitter, br.costs, color=color, alpha=0.55, s=18, zorder=3)

    for br in det:
        ax.axhline(br.costs[0], color=_COLORS.get(br.name, "gray"),
                   linestyle="--", linewidth=1.4,
                   label=f"{br.name.upper()} = {br.costs[0]:.1f} (det.)")

    if show_baseline:
        ax.axhline(bl_cost, color="black", linestyle=":", linewidth=1.8,
                   label=f"SOT baseline ({bl_cost:.1f})", zorder=4)

    ax.set_xticks(range(1, len(stochastic) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fitness cost  (lower is better)")
    ax.set_title(f"BC-model cost distribution across {len(stochastic[0].costs) if stochastic else 0} "
                 f"evaluation seeds")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Boxplots saved -> {path}")
    plt.close()


def plot_significance_matrix(p_df: pd.DataFrame, sig_df: pd.DataFrame,
                              a12_df: pd.DataFrame, alpha_corr: float,
                              ttest_p_df: pd.DataFrame | None = None,
                              ttest_sig_df: pd.DataFrame | None = None,
                              path: str = "bc_significance.png"):
    """
    Significance matrix showing Wilcoxon p, optional paired t-test p, and A₁₂.
    Cell colour is driven by Wilcoxon significance (primary test).
    If ttest_p_df is supplied a second row is added per cell with the t-test p.
    """
    _A12_LABELS = [(0.71, "large"), (0.64, "medium"), (0.56, "small"), (0.0, "neg.")]

    def _a12_label(a: float) -> str:
        delta = abs(a - 0.5)
        for threshold, label in _A12_LABELS:
            if delta >= threshold - 0.5:
                return label
        return "neg."

    names = list(p_df.index)
    n     = len(names)
    cmap  = np.full((n, n, 4), [0.92, 0.92, 0.92, 1.0])
    for i in range(n):
        for j in range(n):
            if i == j:
                cmap[i, j] = [0.6, 0.6, 0.6, 1.0]
            elif sig_df.iloc[i, j] == +1:
                cmap[i, j] = [0.55, 0.86, 0.55, 1.0]
            elif sig_df.iloc[i, j] == -1:
                cmap[i, j] = [0.95, 0.55, 0.55, 1.0]

    fig, ax = plt.subplots(figsize=(max(6, n * 2.2), max(5, n * 1.8)))
    ax.imshow(cmap, aspect="equal", interpolation="nearest")

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="white")
                continue
            p   = p_df.iloc[i, j]
            a12 = a12_df.iloc[i, j]
            if np.isnan(p):
                txt = "N/A"
            else:
                def _fmt_p(pv, sig_val):
                    mark  = " ✓" if sig_val != 0 else ""
                    pstr  = f"{pv:.2e}" if pv < 0.001 else f"{pv:.3f}"
                    return f"p={pstr}{mark}"

                lines = [f"W: {_fmt_p(p, sig_df.iloc[i, j])}"]
                if ttest_p_df is not None:
                    tp = ttest_p_df.iloc[i, j]
                    ts = ttest_sig_df.iloc[i, j] if ttest_sig_df is not None else 0
                    if not np.isnan(tp):
                        lines.append(f"t: {_fmt_p(tp, ts)}")
                if not np.isnan(a12):
                    lines.append(f"A₁₂={a12:.2f} ({_a12_label(a12)})")
                txt = "\n".join(lines)
            ax.text(j, i, txt, ha="center", va="center", fontsize=7)

    ax.set_xticks(range(n)); ax.set_xticklabels(names, fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Column optimizer")
    ax.set_ylabel("Row optimizer")
    has_ttest = ttest_p_df is not None
    ax.set_title(
        f"Pairwise Wilcoxon (W) + {'Paired t-test (t) + ' if has_ttest else ''}"
        f"Vargha-Delaney A₁₂  (Bonferroni α={alpha_corr:.4f})\n"
        f"Green = row better  |  Red = row worse  |  ✓ = significant")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Significance matrix saved -> {path}")
    plt.close()


def plot_cost_bars(runs: list[BenchmarkRun], bl_cost: float,
                   path: str = "bc_cost_bars.png",
                   show_baseline: bool = False):
    fig, ax = plt.subplots(figsize=(9, 4))
    names  = [br.name.upper() for br in runs]
    means  = [br.mean_cost for br in runs]
    stds   = [br.std_cost  for br in runs]
    colors = [_COLORS.get(br.name, "gray") for br in runs]
    bars = ax.bar(names, means, yerr=stds, color=colors,
                  capsize=5, edgecolor="white", linewidth=0.5)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.5,
                f"{m:.1f}", ha="center", va="bottom", fontsize=9)
    if show_baseline:
        ax.axhline(bl_cost, color="black", linestyle=":", linewidth=1.8,
                   label=f"SOT baseline ({bl_cost:.1f})", zorder=4)
    ax.set_ylabel("Fitness cost  (lower is better)")
    ax.set_title("BC-model: mean fitness cost ± 1σ across evaluation seeds")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Cost bars saved -> {path}")
    plt.close()


def plot_time_quality(runs: list[BenchmarkRun], path: str = "bc_time_quality.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    for br in runs:
        if not br.costs:
            continue
        ax.scatter(br.mean_time, br.mean_cost,
                   color=_COLORS.get(br.name, "gray"), s=140, zorder=3)
        ax.annotate(br.name.upper(), (br.mean_time, br.mean_cost),
                    textcoords="offset points", xytext=(7, 4), fontsize=9)
    ax.set_xlabel("Mean runtime (s)")
    ax.set_ylabel("Mean fitness cost")
    ax.set_title("BC-model: runtime vs quality  (Pareto trade-off view)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Time-quality saved -> {path}")
    plt.close()


def plot_budget_sensitivity(budget_results: dict[str, BudgetTuningResult],
                             path: str = "bc_budget.png"):
    """Produce two figures: GA+SA and PSO+Random, each with dual y-axes (cost + time)."""
    stem   = path.replace(".png", "")
    groups = [
        (["ga", "sa"],      f"{stem}_ga_sa.png"),
        (["pso", "random"], f"{stem}_pso_random.png"),
    ]
    title_base = ("Stage 2.1: Budget sensitivity — mean cost vs. budget parameter\n"
                  "(shape-hyperparams fixed at Stage 2 best;  ±1σ shaded)")

    for group_names, out_path in groups:
        group = {k: budget_results[k] for k in group_names if k in budget_results}
        if not group:
            continue

        fig, axes = plt.subplots(1, len(group), figsize=(6 * len(group), 5))
        if len(group) == 1:
            axes = [axes]
        fig.suptitle(title_base, fontsize=11, fontweight="bold")

        for ax, (name, br) in zip(axes, group.items()):
            c    = _COLORS.get(name, "gray")
            xs   = np.array(br.values)
            ys   = np.array(br.means)
            errs = np.array(br.stds)

            l1, = ax.plot(xs, ys, color=c, marker="o", linewidth=1.8,
                          markersize=5, label="Mean cost")
            ax.fill_between(xs, ys - errs, ys + errs, alpha=0.15, color=c)
            vl = ax.axvline(br.best_value, color=c, linestyle="--", linewidth=1.2,
                            label=f"Best = {br.best_value}")
            ax.set_xlabel(br.param.replace("_", " "))
            ax.set_ylabel("Mean fitness cost", color=c)
            ax.tick_params(axis="y", labelcolor=c)
            ax.set_title(name.upper())
            ax.grid(axis="y", alpha=0.3)

            handles, labels = [l1, vl], ["Mean cost", f"Best = {br.best_value}"]

            if br.times:
                ax2  = ax.twinx()
                ts   = np.array(br.times)
                l2,  = ax2.plot(xs, ts, color="dimgray", marker="s", linestyle=":",
                                linewidth=1.5, markersize=4, alpha=0.8,
                                label="Mean time (s)")
                ax2.set_ylabel("Mean time per run (s)", color="dimgray")
                ax2.tick_params(axis="y", labelcolor="dimgray")
                handles.append(l2)
                labels.append("Mean time (s)")

            ax.legend(handles, labels, fontsize=8)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Budget sensitivity saved -> {out_path}")
        plt.close()


def plot_budget_combined(budget_results: dict[str, BudgetTuningResult],
                          path: str = "bc_budget_combined.png") -> None:
    """
    Overlay all optimizers on one plot for easy comparison.
    Primary mode (when timing data is available): X = mean wall-clock time per run (s).
    Fallback (no timing): X = budget parameter value normalised to [0, 1] per optimizer.
    Timing data is collected during Stage 21; re-run it to enable the primary mode.
    """
    has_timing = any(br.times for br in budget_results.values() if br.means)

    fig, ax = plt.subplots(figsize=(9, 6))
    all_points: list[tuple[float, float, str, int]] = []  # (x, cost, name, value)

    for name, br in budget_results.items():
        if not br.means:
            continue
        c    = _COLORS.get(name, "gray")
        ys   = np.array(br.means)
        errs = np.array(br.stds) if br.stds else np.zeros_like(ys)

        if has_timing and br.times:
            xs = np.array(br.times)
        elif has_timing:
            continue  # others have timing; skip this one rather than mix scales
        else:
            # Normalise each optimizer's own budget values to [0, 1]
            raw = np.array(br.values, dtype=float)
            span = raw.max() - raw.min()
            xs = (raw - raw.min()) / span if span > 0 else np.linspace(0, 1, len(raw))

        ax.plot(xs, ys, color=c, marker="o", linewidth=1.8, markersize=6, label=name.upper())
        ax.fill_between(xs, ys - errs, ys + errs, alpha=0.10, color=c)

        for x, y, v in zip(xs, ys, br.values):
            ax.annotate(str(v), (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color=c, alpha=0.85)
            all_points.append((float(x), float(y), name, v))

        best_idx = int(np.argmin(ys))
        ax.scatter([xs[best_idx]], [ys[best_idx]], color=c, s=160, zorder=5,
                   marker="*", edgecolors="black", linewidths=0.5)

    if not all_points:
        plt.close()
        print("  [skip] Combined budget plot: no data to plot")
        return

    # Pareto frontier (minimise both x and cost)
    pts_sorted = sorted(all_points, key=lambda p: p[0])
    pareto: list[tuple[float, float]] = []
    best_cost = float("inf")
    for t, c_val, _, __ in pts_sorted:
        if c_val < best_cost:
            best_cost = c_val
            pareto.append((t, c_val))
    if len(pareto) >= 2:
        px, py = zip(*pareto)
        ax.step(list(px) + [px[-1]], list(py) + [py[-1]], where="post",
                color="black", linestyle="--", linewidth=1.3, alpha=0.55,
                label="Pareto frontier", zorder=2)

    if has_timing:
        xlabel     = "Mean time per run (s)"
        title_note = "dashed = Pareto frontier"
    else:
        xlabel     = "Relative budget (normalised per optimizer — re-run Stage 21 for time axis)"
        title_note = "timing unavailable; x-axis shows relative budget per optimizer"

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean fitness cost")
    ax.set_title(f"Budget sensitivity: cost comparison (all optimizers)\n"
                 f"★ = lowest cost per optimizer  |  numbers = budget value  |  {title_note}",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Combined budget plot saved -> {path}")
    plt.close()

    # Efficiency table (only meaningful when timing available)
    if has_timing:
        all_t  = [p[0] for p in all_points]
        all_c  = [p[1] for p in all_points]
        t_min, t_rng = min(all_t), max(all_t) - min(all_t) or 1.0
        c_min, c_rng = min(all_c), max(all_c) - min(all_c) or 1.0
        scored = [
            (p[2], p[3], p[0], p[1],
             ((p[1] - c_min) / c_rng) * ((p[0] - t_min) / t_rng))
            for p in all_points
        ]
        scored.sort(key=lambda x: x[4])
        print("\n  Time–cost efficiency ranking  (lower score = closer to bottom-left):")
        print(f"  {'Rank':<5} {'Optimizer':<10} {'Budget':<10} {'Time(s)':<10} "
              f"{'Cost':<10} {'Score':<8}")
        print("  " + "-" * 55)
        for rank, (name, val, t, c_val, score) in enumerate(scored[:10], 1):
            print(f"  {rank:<5} {name.upper():<10} {val:<10} {t:<10.1f} "
                  f"{c_val:<10.1f} {score:<8.4f}")


def plot_tuning_heatmap(name: str, records: list[dict],
                        path: str | None = None) -> None:
    """
    Single heatmap: first param on Y-axis, second param on X-axis.
    Values averaged over any remaining parameters.
    Matches the layout of benchmark_optimizers.py.
    """
    if not records:
        return
    param_keys = list(records[0]["params"].keys())
    if len(param_keys) < 2:
        return

    p1, p2 = param_keys[0], param_keys[1]
    v1s = sorted(set(r["params"][p1] for r in records))
    v2s = sorted(set(r["params"][p2] for r in records))

    grid_vals: dict = {}
    for r in records:
        k = (r["params"][p1], r["params"][p2])
        grid_vals.setdefault(k, []).append(r["mean"])

    matrix = np.array([
        [np.mean(grid_vals.get((v1, v2), [np.nan])) for v2 in v2s]
        for v1 in v1s
    ])

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(v2s)))
    ax.set_xticklabels([str(v) for v in v2s], rotation=45, ha="right")
    ax.set_yticks(range(len(v1s)))
    ax.set_yticklabels([str(v) for v in v1s])
    ax.set_xlabel(p2); ax.set_ylabel(p1)
    ax.set_title(f"{name.upper()} tuning heatmap  "
                 f"({p1} vs {p2}, averaged over remaining params)")
    plt.colorbar(im, ax=ax, label="Mean fitness cost")
    for i in range(len(v1s)):
        for j in range(len(v2s)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.0f}",
                        ha="center", va="center", fontsize=8)
    plt.tight_layout()
    out = path or f"bc_tuning_{name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Tuning heatmap saved -> {out}")
    plt.close()


def plot_tuning_3d(name: str, records: list[dict],
                   path: str | None = None) -> None:
    """
    3D surface plot for optimizers with 3+ parameters.

    X axis  = first param
    Y axis  = second param
    Z axis  = mean fitness cost  (labeled and shown as a colorbar)
    Surface colour = cost value via RdYlGn_r (red = expensive, green = cheap).
    Edge colour = one distinct colour per value of the third param so the
                  separate layers are easy to tell apart.
    Numeric cost labels are printed at every grid point on every surface.
    If a 4th param exists, its effect is averaged within each surface.
    """
    from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Patch

    if not records:
        return
    param_keys = list(records[0]["params"].keys())
    if len(param_keys) < 3:
        return

    p1, p2, p3 = param_keys[0], param_keys[1], param_keys[2]
    extra = param_keys[3:]

    v1s = sorted(set(r["params"][p1] for r in records))
    v2s = sorted(set(r["params"][p2] for r in records))
    v3s = sorted(set(r["params"][p3] for r in records))

    # Global cost range for a shared colormap
    all_finite = [r["mean"] for r in records
                  if not np.isnan(r["mean"]) and not np.isinf(r["mean"])]
    vmin, vmax = (min(all_finite), max(all_finite)) if all_finite else (0, 1)
    norm   = Normalize(vmin=vmin, vmax=vmax)
    cmap   = plt.cm.RdYlGn_r
    palette = plt.cm.tab10(np.linspace(0, 0.6, len(v3s)))

    X_g, Y_g = np.meshgrid(range(len(v1s)), range(len(v2s)))

    fig = plt.figure(figsize=(13, 8))
    ax  = fig.add_subplot(111, projection="3d")

    for k, v3 in enumerate(v3s):
        grid_vals: dict = {}
        for r in records:
            if r["params"][p3] != v3:
                continue
            key = (r["params"][p1], r["params"][p2])
            grid_vals.setdefault(key, []).append(r["mean"])

        # Z[row=j, col=i] = cost(v1s[i], v2s[j])
        Z = np.array([
            [np.mean(grid_vals.get((v1, v2), [np.nan])) for v1 in v1s]
            for v2 in v2s
        ])

        # Surface face colour = cost (shared RdYlGn_r scale)
        face_rgba = cmap(norm(np.nan_to_num(Z, nan=vmax)))
        face_rgba[..., 3] = 0.60   # alpha

        ax.plot_surface(X_g, Y_g, Z,
                        facecolors=face_rgba,
                        shade=False)
        # Wireframe overlay distinguishes p3 layers by colour
        ax.plot_wireframe(X_g, Y_g, Z,
                          color=palette[k], linewidth=0.7, alpha=0.9)

        # Numeric cost label at each grid point
        for j in range(len(v2s)):
            for i in range(len(v1s)):
                z_val = Z[j, i]
                if not np.isnan(z_val):
                    ax.text(i, j, z_val, f"{z_val:.0f}",
                            fontsize=6, ha="center", va="bottom",
                            color="black", zorder=5)

    ax.set_xticks(range(len(v1s)))
    ax.set_xticklabels([str(v) for v in v1s], fontsize=7)
    ax.set_yticks(range(len(v2s)))
    ax.set_yticklabels([str(v) for v in v2s], fontsize=7)
    ax.set_xlabel(p1, fontsize=8, labelpad=8)
    ax.set_ylabel(p2, fontsize=8, labelpad=8)
    ax.set_zlabel("Mean fitness cost", fontsize=8)

    avg_note = f"  (avg over {extra})" if extra else ""
    ax.set_title(f"{name.upper()} tuning — {p1} / {p2} / {p3}{avg_note}",
                 fontsize=9)

    # Colorbar for the cost scale
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Mean fitness cost",
                 shrink=0.45, pad=0.12)

    # Legend: edge colour → p3 value
    handles = [Patch(edgecolor=palette[k], facecolor="none", linewidth=2,
                     label=f"{p3}={v}")
               for k, v in enumerate(v3s)]
    ax.legend(handles=handles, fontsize=8, loc="upper left")

    plt.tight_layout()
    out = path or f"bc_tuning_{name}_3d.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  3D tuning plot saved -> {out}")
    plt.close()


# =============================================================================
# EXPORTS
# =============================================================================

def export_summary(sanity: list[SanityResult], runs: list[BenchmarkRun],
                   p_df: pd.DataFrame | None, sig_df: pd.DataFrame | None,
                   bl_cost: float, path: str = "bc_summary.csv") -> pd.DataFrame:
    sanity_map = {s.name: s for s in sanity}
    rows = []
    for br in runs:
        sr  = sanity_map.get(br.name)
        avg_m: dict[str, list] = {}
        for m in br.metrics:
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    avg_m.setdefault(k, []).append(v)
        avg_flat = {k: round(float(np.mean(vs)), 2) for k, vs in avg_m.items()}
        pct = (br.mean_cost - bl_cost) / bl_cost * 100 if bl_cost else 0.0
        row = {
            "optimizer":       br.name,
            "config":          br.config_desc,
            "sanity_pass":     sr.beats_baseline if sr else None,
            "n_runs":          len(br.costs),
            "mean_cost":       round(br.mean_cost, 2),
            "std_cost":        round(br.std_cost, 2),
            "best_cost":       round(br.best_cost, 2),
            "vs_baseline_pct": round(pct, 1),
            "mean_time_s":     round(br.mean_time, 2),
            **{f"avg_{k}": v for k, v in avg_flat.items()},
        }
        if sig_df is not None and br.name in sig_df.index:
            for col in sig_df.columns:
                if col != br.name:
                    v = sig_df.loc[br.name, col]
                    row[f"sig_vs_{col}"] = (
                        "better" if v == 1 else "worse" if v == -1 else "ns"
                    )
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, sep=";", decimal=",")
    print(f"  Summary saved -> {path}")
    return df


def export_significance(p_df: pd.DataFrame, sig_df: pd.DataFrame,
                        a12_df: pd.DataFrame, alpha_corr: float,
                        ttest_p_df: pd.DataFrame | None = None,
                        ttest_sig_df: pd.DataFrame | None = None,
                        path: str = "bc_significance.csv"):
    rows = []
    for row_name in p_df.index:
        for col_name in p_df.columns:
            if row_name == col_name:
                continue
            p   = p_df.loc[row_name, col_name]
            sig = sig_df.loc[row_name, col_name]
            a12 = a12_df.loc[row_name, col_name]
            row = {
                "optimizer_row":        row_name,
                "optimizer_col":        col_name,
                "wilcoxon_p":           float(p) if not np.isnan(p) else None,
                "wilcoxon_significant": bool(sig != 0),
                "direction":            ("better" if sig == 1 else "worse" if sig == -1 else "ns"),
                "a12_effect_size":      round(float(a12), 4) if not np.isnan(a12) else None,
                "alpha_bonferroni":     round(alpha_corr, 6),
            }
            if ttest_p_df is not None:
                tp  = ttest_p_df.loc[row_name, col_name]
                ts  = ttest_sig_df.loc[row_name, col_name] if ttest_sig_df is not None else 0
                row["ttest_p"]           = float(tp) if not np.isnan(tp) else None
                row["ttest_significant"] = bool(ts != 0)
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, sep=";", decimal=",")
    print(f"  Significance table saved -> {path}")


def export_tuning_table(tuning: dict[str, TuningResult],
                        path: str = "bc_tuning.csv"):
    rows = []
    for name, tr in tuning.items():
        for rec in tr.all_records:
            rows.append({"optimizer": name,
                         "mean_cost": round(rec["mean"], 2),
                         "std_cost":  round(rec["std"], 2),
                         **rec["params"]})
    pd.DataFrame(rows).to_csv(path, index=False, sep=";", decimal=",")
    print(f"  Tuning table saved -> {path}")


def export_boxplot_data(runs: list[BenchmarkRun], path: str = "bc_boxplot.csv"):
    rows = []
    for br in runs:
        for i, (cost, t) in enumerate(zip(br.costs, br.times_s)):
            row = {"optimizer": br.name, "run_index": i,
                   "cost": round(cost, 4), "time_s": round(t, 2)}
            if i < len(br.metrics):
                row.update({f"m_{k}": v for k, v in br.metrics[i].items()})
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, sep=";", decimal=",")
    print(f"  Boxplot data saved -> {path}")


def export_budget_data(budget_results: dict[str, BudgetTuningResult],
                       path: str = "bc_budget.csv"):
    rows = []
    for name, br in budget_results.items():
        times_iter = iter(br.times) if br.times else None
        for value, mean, std in zip(br.values, br.means, br.stds):
            t = round(next(times_iter), 4) if times_iter is not None else None
            rows.append({"optimizer": name, "param": br.param, "value": value,
                         "mean_cost": round(mean, 4), "std_cost": round(std, 4),
                         "mean_time_s": t, "is_best": value == br.best_value})
    pd.DataFrame(rows).to_csv(path, index=False, sep=";", decimal=",")
    print(f"  Budget data saved -> {path}")


def export_convergence_data(runs: list[BenchmarkRun], bl_cost: float,
                            n_points: int = 300,
                            path: str = "bc_convergence.csv"):
    """
    Export raw per-seed convergence histories to CSV (long format).
    Each row is one (optimizer, seed_index, eval_count, fitness) record.
    Stage 4 reads this to plot individual seed lines + mean without smoothing.
    """
    valid = [br for br in runs if br.histories]
    if not valid:
        return
    rows = []
    for br in valid:
        all_x = [_eval_counts(br.name, len(h), br.config_obj) for h in br.histories]
        for si, (xs, h) in enumerate(zip(all_x, br.histories)):
            for ev, fit in zip(xs, h):
                rows.append({
                    "optimizer": br.name,
                    "seed_idx":  si,
                    "eval_count": int(ev),
                    "fitness":   round(float(fit), 4),
                    "baseline":  round(bl_cost, 4),
                })
    pd.DataFrame(rows).to_csv(path, index=False, sep=";", decimal=",")
    print(f"  Convergence data saved -> {path}")


# =============================================================================
# STAGE 4 — REGENERATE PLOTS FROM SAVED CSVs
# =============================================================================
# STAGE 5 — DETERMINISTIC vs DES COMPARISON
# =============================================================================

@dataclass
class DESRunResult:
    name:      str
    det_cost:  float
    des_costs: list[float]
    n_reps:    int

    @property
    def des_mean(self): return float(np.mean(self.des_costs))
    @property
    def des_std(self):  return float(np.std(self.des_costs))
    @property
    def des_p10(self):  return float(np.percentile(self.des_costs, 10))
    @property
    def des_p50(self):  return float(np.percentile(self.des_costs, 50))
    @property
    def des_p90(self):  return float(np.percentile(self.des_costs, 90))


def _bc_eval_des(chrom: list[int], jobs: list[_BCJob],
                 pool_B: int, pool_C: int, delay_seed: int) -> float:
    """One DES replication of a 3N benchmark chromosome with PERT delays."""
    n     = len(jobs)
    fresh = _bc_fresh_jobs(jobs)
    # Dispatch order: sort by priority (first N chromosome elements)
    fresh.sort(key=lambda j: chrom[j.job_index])
    # DES chromosome is 2N: [B_workers … C_workers …]
    des_chrom = list(chrom[n:2 * n]) + list(chrom[2 * n:3 * n])
    _, _sc1, sc2, sc3 = _simulate_des_bc(
        fresh, des_chrom, pool_B, pool_C,
        use_delays=True, delay_seed=delay_seed,
    )
    done     = [j for j in fresh if j.oven_end is not None]
    makespan = max((j.oven_end for j in done), default=float("inf"))
    return makespan + SC2_WEIGHT * sc2 + SC3_WEIGHT * sc3


def export_chromosomes(runs: list[BenchmarkRun],
                       path: str = "bc_chromosomes.csv") -> None:
    rows = [{"optimizer": br.name,
             "det_cost": round(br.mean_cost, 4),
             "chromosome_len": len(br.best_chromosome),
             "chromosome": " ".join(map(str, br.best_chromosome))}
            for br in runs if br.best_chromosome]
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False, sep=";")
        print(f"  Chromosomes saved -> {path}")


def run_des_comparison(jobs: list[_BCJob], det_runs: list[BenchmarkRun],
                       pool_B: int, pool_C: int,
                       n_reps: int = 200, seed: int = 42) -> dict[str, DESRunResult]:
    if not _DES_AVAILABLE:
        print("  [skip] des_simulationBC not importable — Stage 5 requires it")
        return {}

    print("\n" + "=" * 60)
    print("  Stage 5: Deterministic vs DES comparison")
    print(f"  Replications per optimizer : {n_reps}")
    print(f"  Base delay seed            : {seed}")
    print("=" * 60)

    results: dict[str, DESRunResult] = {}
    for br in det_runs:
        if not br.best_chromosome:
            print(f"  [{br.name.upper()}] no chromosome stored — skipping")
            continue
        print(f"\n  [{br.name.upper()}]  det_cost={br.mean_cost:.2f}")
        t0        = time.perf_counter()
        des_costs = [_bc_eval_des(br.best_chromosome, jobs, pool_B, pool_C,
                                  delay_seed=seed + rep)
                     for rep in range(n_reps)]
        elapsed   = time.perf_counter() - t0
        r         = DESRunResult(br.name, br.mean_cost, des_costs, n_reps)
        results[br.name] = r
        print(f"    DES  mean={r.des_mean:.2f} ±{r.des_std:.2f}  "
              f"P10={r.des_p10:.2f}  P50={r.des_p50:.2f}  P90={r.des_p90:.2f}  "
              f"({elapsed:.1f}s)")

    if results:
        print("\n  Sensitivity ranking  (CV = std/mean, lower = more robust under delay):")
        print(f"  {'Rank':<5} {'Optimizer':<10} {'Det cost':<12} {'DES mean':<12} "
              f"{'Drift':<12} {'CV':<8}")
        print("  " + "-" * 60)
        ranked = sorted(results.values(), key=lambda r: r.des_std / (r.des_mean or 1))
        for i, r in enumerate(ranked, 1):
            drift = r.des_mean - r.det_cost
            cv    = r.des_std / (r.des_mean or 1)
            sign  = "+" if drift >= 0 else ""
            print(f"  {i:<5} {r.name.upper():<10} {r.det_cost:<12.1f} {r.des_mean:<12.1f} "
                  f"{sign}{drift:<11.1f} {cv:<8.4f}")

    return results


def plot_det_vs_des(det_runs: list[BenchmarkRun],
                    des_results: dict[str, DESRunResult],
                    path: str = "bc_det_vs_des.png") -> None:
    names = [br.name for br in det_runs if br.name in des_results]
    if not names:
        return
    x  = np.arange(len(names))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    det_vals  = [next(br.mean_cost for br in det_runs if br.name == n) for n in names]
    des_means = [des_results[n].des_mean for n in names]
    des_stds  = [des_results[n].des_std  for n in names]
    colors    = [_COLORS.get(n, "gray") for n in names]

    bars1 = ax.bar(x - w / 2, det_vals, w, color=colors, alpha=0.95,
                   label="Deterministic", edgecolor="white")
    bars2 = ax.bar(x + w / 2, des_means, w, color=colors, alpha=0.50,
                   yerr=des_stds, capsize=5, label="DES mean ±1σ", edgecolor="white")

    for bar, v in zip(bars1, det_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    for bar, v, s in zip(bars2, des_means, des_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 5,
                f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in names])
    ax.set_ylabel("Fitness cost  (lower is better)")
    ax.set_title("Stage 5: Deterministic vs DES  "
                 "(solid = deterministic, faded = DES mean ±1σ)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Det vs DES saved -> {path}")
    plt.close()


def plot_des_distribution(det_runs: list[BenchmarkRun],
                          des_results: dict[str, DESRunResult],
                          path: str = "bc_des_dist.png") -> None:
    names = [br.name for br in det_runs if br.name in des_results]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    data   = [des_results[n].des_costs for n in names]
    colors = [_COLORS.get(n, "gray") for n in names]

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    rng = np.random.default_rng(0)
    for k, (n, color) in enumerate(zip(names, colors), start=1):
        jitter = rng.uniform(-0.15, 0.15, len(des_results[n].des_costs))
        ax.scatter(k + jitter, des_results[n].des_costs,
                   color=color, alpha=0.35, s=14, zorder=3)
        det = next(br.mean_cost for br in det_runs if br.name == n)
        ax.scatter(k, det, marker="D", color=color, s=90, zorder=5,
                   edgecolors="black", linewidths=0.8)

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels([n.upper() for n in names])
    ax.set_ylabel("Fitness cost  (lower is better)")
    ax.set_title(f"Stage 5: DES cost distribution  ({des_results[names[0]].n_reps} replications)\n"
                 "◆ = deterministic cost  |  box = DES distribution")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  DES distribution saved -> {path}")
    plt.close()


def plot_des_percentiles(des_results: dict[str, DESRunResult],
                         det_runs: list[BenchmarkRun],
                         path: str = "bc_des_percentiles.png") -> None:
    if not des_results:
        return
    names = list(des_results.keys())
    fig, ax = plt.subplots(figsize=(9, 5))

    for xi, name in enumerate(names):
        r   = des_results[name]
        c   = _COLORS.get(name, "gray")
        p10, p50, p90 = r.des_p10, r.des_p50, r.des_p90
        ax.plot([xi, xi], [p10, p90], color=c, linewidth=3,
                solid_capstyle="round", zorder=2)
        ax.scatter(xi, p50, color=c, s=120, zorder=4, marker="o")
        for pv, label in [(p10, f"P10 {p10:.0f}"), (p90, f"P90 {p90:.0f}")]:
            ax.scatter(xi, pv, color=c, s=60, zorder=4,
                       marker="_", linewidths=2.5)
        ax.text(xi + 0.08, p10, f"P10={p10:.0f}", fontsize=7,
                va="center", color=c)
        ax.text(xi + 0.08, p50, f"P50={p50:.0f}", fontsize=7,
                va="center", color=c)
        ax.text(xi + 0.08, p90, f"P90={p90:.0f}", fontsize=7,
                va="center", color=c)
        det = next((br.mean_cost for br in det_runs if br.name == name), None)
        if det is not None:
            ax.scatter(xi, det, marker="D", color=c, s=90, zorder=5,
                       edgecolors="black", linewidths=0.8)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.upper() for n in names])
    ax.set_ylabel("Fitness cost  (lower is better)")
    ax.set_title("Stage 5: DES cost percentiles  (P10 / P50 / P90)\n"
                 "Bar = P10–P90 range  |  Circle = median  |  ◆ = deterministic cost")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  DES percentiles saved -> {path}")
    plt.close()


def export_des_results(det_runs: list[BenchmarkRun],
                       des_results: dict[str, DESRunResult],
                       path: str = "bc_des_results.csv") -> None:
    rows = []
    for name, r in des_results.items():
        rows.append({
            "optimizer": name,
            "det_cost":  round(r.det_cost, 4),
            "des_mean":  round(r.des_mean, 4),
            "des_std":   round(r.des_std,  4),
            "des_p10":   round(r.des_p10,  4),
            "des_p50":   round(r.des_p50,  4),
            "des_p90":   round(r.des_p90,  4),
            "n_reps":    r.n_reps,
            "drift":     round(r.des_mean - r.det_cost, 4),
            "cv":        round(r.des_std / (r.des_mean or 1), 6),
        })
    pd.DataFrame(rows).to_csv(path, index=False, sep=";", decimal=",")
    print(f"  DES results saved -> {path}")


# =============================================================================

def run_plots_from_csv(bl_cost: float) -> None:
    """
    Regenerate every available plot from previously saved CSV files.
    Reads whichever of the bc_*.csv files exist; silently skips missing ones.
    """
    print("\n" + "=" * 60)
    print("  Stage 4: Regenerate plots from saved CSV files")
    print("=" * 60)

    _CSV_SEP = ";"
    _CSV_DEC = ","
    _ORDER   = ["ga", "sa", "pso", "random", "sot"]
    generated: list[str] = []

    # ── Tuning heatmaps ──────────────────────────────────────────────────────
    if os.path.exists("bc_tuning.csv"):
        print("\n  Reading bc_tuning.csv ...")
        df = pd.read_csv("bc_tuning.csv", sep=_CSV_SEP, decimal=_CSV_DEC)
        for name in df["optimizer"].unique():
            sub        = df[df["optimizer"] == name]
            # Drop columns that belong to other optimizers (all-NaN for this one)
            param_cols = [c for c in sub.columns
                          if c not in ("optimizer", "mean_cost", "std_cost")
                          and sub[c].notna().any()]
            records = [
                {"params": {c: row[c] for c in param_cols},
                 "mean":   float(row["mean_cost"]),
                 "std":    float(row["std_cost"])}
                for _, row in sub.iterrows()
            ]
            plot_tuning_heatmap(name, records, path=f"bc_tuning_{name}.png")
            generated.append(f"bc_tuning_{name}.png")
            plot_tuning_3d(name, records, path=f"bc_tuning_{name}_3d.png")
            if len(param_cols) >= 3:
                generated.append(f"bc_tuning_{name}_3d.png")
    else:
        print("  [skip] bc_tuning.csv not found")

    # ── Budget sensitivity ───────────────────────────────────────────────────
    if os.path.exists("bc_budget.csv"):
        print("\n  Reading bc_budget.csv ...")
        df = pd.read_csv("bc_budget.csv", sep=_CSV_SEP, decimal=_CSV_DEC)
        budget_results: dict[str, BudgetTuningResult] = {}
        for name in df["optimizer"].unique():
            sub      = df[df["optimizer"] == name].sort_values("value")
            param    = sub["param"].iloc[0]
            values   = [int(v) for v in sub["value"].tolist()]
            means    = sub["mean_cost"].tolist()
            stds     = sub["std_cost"].tolist()
            times    = (sub["mean_time_s"].tolist()
                        if "mean_time_s" in sub.columns and sub["mean_time_s"].notna().any()
                        else [])
            best_row = sub[sub["is_best"] == True]  # noqa: E712
            best_val = int(best_row["value"].iloc[0]) if len(best_row) else values[int(np.argmin(means))]
            budget_results[name] = BudgetTuningResult(name, param, values, means, stds,
                                                      best_val, times=times)
        plot_budget_sensitivity(budget_results)
        plot_budget_combined(budget_results)
        generated.extend(["bc_budget_ga_sa.png", "bc_budget_pso_random.png",
                           "bc_budget_combined.png"])
    else:
        print("  [skip] bc_budget.csv not found")

    # ── Stage 3 boxplots / cost bars / time-quality ──────────────────────────
    if os.path.exists("bc_boxplot.csv"):
        print("\n  Reading bc_boxplot.csv ...")
        df   = pd.read_csv("bc_boxplot.csv", sep=_CSV_SEP, decimal=_CSV_DEC)
        names = sorted(df["optimizer"].unique(),
                       key=lambda x: _ORDER.index(x) if x in _ORDER else 99)
        runs: list[BenchmarkRun] = []
        for name in names:
            sub = df[df["optimizer"] == name]
            br  = BenchmarkRun(name=name, config_desc="(from CSV)", config_obj=None)
            br.costs   = sub["cost"].tolist()
            br.times_s = sub["time_s"].tolist()
            runs.append(br)
        plot_boxplots(runs, bl_cost, path="bc_boxplots.png",               show_baseline=False)
        plot_boxplots(runs, bl_cost, path="bc_boxplots_with_baseline.png", show_baseline=True)
        plot_cost_bars(runs, bl_cost, path="bc_cost_bars.png",               show_baseline=False)
        plot_cost_bars(runs, bl_cost, path="bc_cost_bars_with_baseline.png", show_baseline=True)
        plot_time_quality(runs)
        generated += ["bc_boxplots.png", "bc_cost_bars.png", "bc_time_quality.png"]
    else:
        print("  [skip] bc_boxplot.csv not found")

    # ── Significance matrix ──────────────────────────────────────────────────
    if os.path.exists("bc_significance.csv"):
        print("\n  Reading bc_significance.csv ...")
        df    = pd.read_csv("bc_significance.csv", sep=_CSV_SEP, decimal=_CSV_DEC)
        names = list(dict.fromkeys(df["optimizer_row"].tolist()))
        n     = len(names)
        idx   = {name: i for i, name in enumerate(names)}
        p_mat    = np.full((n, n), np.nan)
        sig_mat  = np.zeros((n, n), dtype=int)
        a12_mat  = np.full((n, n), np.nan)
        tp_mat   = np.full((n, n), np.nan)
        tsig_mat = np.zeros((n, n), dtype=int)
        alpha_v  = float(_ALPHA)
        for _, row in df.iterrows():
            ri = idx.get(row["optimizer_row"])
            ci = idx.get(row["optimizer_col"])
            if ri is None or ci is None:
                continue
            p_col = "wilcoxon_p" if "wilcoxon_p" in df.columns else "p_value"
            p_mat[ri, ci]   = row[p_col] if pd.notna(row[p_col]) else np.nan
            a12_mat[ri, ci] = row["a12_effect_size"] if pd.notna(row["a12_effect_size"]) else np.nan
            sig_mat[ri, ci] = (1  if row["direction"] == "better" else
                               -1 if row["direction"] == "worse"  else 0)
            if "ttest_p" in df.columns and pd.notna(row["ttest_p"]):
                tp_mat[ri, ci]   = row["ttest_p"]
                tsig_mat[ri, ci] = (1  if row.get("ttest_significant") and row["direction"] == "better" else
                                    -1 if row.get("ttest_significant") and row["direction"] == "worse"  else 0)
            alpha_v = float(row.get("alpha_bonferroni", _ALPHA))

        _p_df    = pd.DataFrame(p_mat,    index=names, columns=names)
        _sig_df  = pd.DataFrame(sig_mat,  index=names, columns=names)
        _a12_df  = pd.DataFrame(a12_mat,  index=names, columns=names)
        has_ttest = not np.all(np.isnan(tp_mat))
        _tp_df   = pd.DataFrame(tp_mat,   index=names, columns=names) if has_ttest else None
        _tsig_df = pd.DataFrame(tsig_mat, index=names, columns=names) if has_ttest else None
        plot_significance_matrix(_p_df, _sig_df, _a12_df, alpha_v,
                                 ttest_p_df=_tp_df, ttest_sig_df=_tsig_df)
        generated.append("bc_significance.png")
    else:
        print("  [skip] bc_significance.csv not found")

    # ── Convergence curves ───────────────────────────────────────────────────
    if os.path.exists("bc_convergence.csv"):
        print("\n  Reading bc_convergence.csv ...")
        df   = pd.read_csv("bc_convergence.csv", sep=_CSV_SEP, decimal=_CSV_DEC)
        opts = [n for n in _ORDER if n in df["optimizer"].unique()]
        opts += [n for n in df["optimizer"].unique() if n not in opts]

        # Build entries in the same format _draw_convergence expects
        entries = []
        for name in opts:
            grp   = df[df["optimizer"] == name].sort_values(["seed_idx", "eval_count"])
            seeds = sorted(grp["seed_idx"].unique())
            # Stack seeds to shortest common length (matches Stage 3 behaviour)
            histories = [grp[grp["seed_idx"] == si]["fitness"].values for si in seeds]
            xs_list   = [grp[grp["seed_idx"] == si]["eval_count"].values for si in seeds]
            min_len   = min(len(h) for h in histories)
            raw       = np.array([h[:min_len] for h in histories])
            xs_ref    = xs_list[0][:min_len]
            mean_h    = raw.mean(axis=0)
            finals    = [float(h[-1]) for h in histories]
            mu        = float(np.mean(finals))
            std       = float(np.std(finals))
            std_h     = raw.std(axis=0)
            entries.append((name, xs_ref, mean_h, std_h, mu, std))

        for suffix, zoom, show_bl in [
                ("bc_convergence.png",                      False, False),
                ("bc_convergence_with_baseline.png",        False, True),
                ("bc_convergence_zoomed.png",               True,  False),
                ("bc_convergence_zoomed_with_baseline.png", True,  True),
        ]:
            fig, ax = plt.subplots(figsize=(11, 5))
            title_suffix = "zoomed on final values" if zoom else "full convergence"
            if show_bl:
                ax.axhline(bl_cost, color="black", linestyle=":", linewidth=1.5,
                           label=f"SOT baseline ({bl_cost:.1f})", zorder=4)
            _draw_convergence(ax, entries, zoom=zoom)
            ax.set_title(f"BC-model optimizer convergence — {title_suffix}")
            plt.tight_layout()
            plt.savefig(suffix, dpi=150, bbox_inches="tight")
            print(f"  Convergence saved -> {suffix}")
            plt.close()
        generated += ["bc_convergence.png", "bc_convergence_with_baseline.png",
                      "bc_convergence_zoomed.png", "bc_convergence_zoomed_with_baseline.png"]
    else:
        print("  [skip] bc_convergence.csv not found  "
              "(run Stage 3 first to generate it)")

    # ── Stage 5 DES results ──────────────────────────────────────────────────
    if os.path.exists("bc_des_results.csv"):
        print("\n  Reading bc_des_results.csv ...")
        df   = pd.read_csv("bc_des_results.csv", sep=_CSV_SEP, decimal=_CSV_DEC)
        fake_det = [BenchmarkRun(name=row["optimizer"], config_desc="", config_obj=None,
                                 costs=[row["det_cost"]])
                    for _, row in df.iterrows()]
        des_results_csv: dict[str, DESRunResult] = {}
        for _, row in df.iterrows():
            des_results_csv[row["optimizer"]] = DESRunResult(
                name=row["optimizer"], det_cost=row["det_cost"],
                des_costs=[], n_reps=int(row["n_reps"]))
            # Reconstruct synthetic cost list from percentiles for plotting
            p10, p50, p90 = row["des_p10"], row["des_p50"], row["des_p90"]
            mu, sd        = row["des_mean"], row["des_std"]
            rng_s         = np.random.default_rng(42)
            synthetic     = list(np.clip(rng_s.normal(mu, sd, int(row["n_reps"])),
                                         p10 * 0.95, p90 * 1.05))
            des_results_csv[row["optimizer"]].des_costs = synthetic
        plot_det_vs_des(fake_det, des_results_csv)
        plot_des_distribution(fake_det, des_results_csv)
        plot_des_percentiles(des_results_csv, fake_det)
        generated += ["bc_det_vs_des.png", "bc_des_dist.png", "bc_des_percentiles.png"]
    else:
        print("  [skip] bc_des_results.csv not found  (run Stage 5 first)")

    print(f"\n  Stage 4 done — {len(generated)} output(s) written.")


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BC-model optimizer benchmark (thesis-ready)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--n-workers", type=int, default=CFG_N_WORKERS, metavar="N",
                   help=f"Parallel worker processes  (default: {CFG_N_WORKERS}).")
    p.add_argument("--skip-tuning", action="store_true",
                   help="Skip Stage 2; use default parameters in Stage 3.")
    p.add_argument("--tune-only", type=str, nargs="+", metavar="OPT",
                   choices=["ga", "sa", "pso"],
                   help="Tune only the listed optimizers in Stage 2 (e.g. --tune-only sa).")
    p.add_argument("--stage", type=int, choices=[1, 2, 21, 3, 4, 5], default=None,
                   help="Run only one stage.\n"
                        "  4 = regenerate all plots from existing bc_*.csv files\n"
                        "      (no optimizers run; SOT baseline recomputed for axis scaling).\n"
                        "  5 = deterministic vs DES comparison (requires Stage 3 outputs).")
    p.add_argument("--des-reps", type=int, default=200, metavar="N",
                   help="Monte Carlo replications per optimizer in Stage 5  (default: 100).")
    p.add_argument("--budget", type=int, default=None, metavar="N",
                   help="Total objective evaluations per run (normalises budget across optimizers).")
    p.add_argument("--tuning-seeds", type=int, nargs="+",
                   default=CFG_TUNING_SEEDS, metavar="S",
                   help=f"Seeds for Stage 2 tuning  (default: {CFG_TUNING_SEEDS[0]}–{CFG_TUNING_SEEDS[-1]}).")
    p.add_argument("--eval-seeds", type=int, nargs="+",
                   default=CFG_EVAL_SEEDS, metavar="S",
                   help=f"Seeds for Stage 3 evaluation  (default: {CFG_EVAL_SEEDS[0]}–{CFG_EVAL_SEEDS[-1]}).")
    p.add_argument("--n-jobs", type=int, default=CFG_N_JOBS, metavar="N",
                   help=f"Use only the first N jobs from data.csv  (default: {CFG_N_JOBS or 'all'}).")
    p.add_argument("--subset-size", type=int, default=CFG_SUBSET_SIZE, metavar="K",
                   help=f"Jobs per subset when --n-datasets > 1  (default: {CFG_SUBSET_SIZE}).")
    p.add_argument("--n-datasets", type=int, default=CFG_N_DATASETS, metavar="N",
                   help=f"Random job subsets for Stage 3 generalisation  (default: {CFG_N_DATASETS}).")
    p.add_argument("--tuning-n-datasets", type=int, default=CFG_TUNING_N_DATASETS, metavar="N",
                   help=f"Random subsets averaged during Stage 2 tuning  (default: {CFG_TUNING_N_DATASETS}).")
    p.add_argument("--dataset-seed", type=int, default=CFG_DATASET_SEED, metavar="S",
                   help=f"Base RNG seed for dataset sampling  (default: {CFG_DATASET_SEED}).")
    p.add_argument("--workers-b", type=int, default=CFG_POOL_B, metavar="N",
                   help=f"B worker pool size  (default: {CFG_POOL_B}).")
    p.add_argument("--workers-c", type=int, default=CFG_POOL_C, metavar="N",
                   help=f"C worker pool size  (default: {CFG_POOL_C}).")
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    args = _parse_args()

    print("=" * 60)
    print("  BC-Model Optimizer Benchmark  (thesis-ready edition)")
    print("=" * 60)

    all_jobs = _load_jobs_bc()
    if args.n_jobs:
        all_jobs = all_jobs[:args.n_jobs]
    # Reassign job_index after any slicing
    all_jobs = [_BCJob(job_id=j.job_id, estimated_h=j.estimated_h, job_index=i)
                for i, j in enumerate(all_jobs)]

    pool_B, pool_C = args.workers_b, args.workers_c
    n = len(all_jobs)
    ests = [j.estimated_h for j in all_jobs]
    print(f"\nLoaded {n} jobs  |  B pool: {pool_B}  C pool: {pool_C}")
    print(f"  min={min(ests):.1f} h  max={max(ests):.1f} h  "
          f"mean={sum(ests)/len(ests):.1f} h")

    bl_cost = _baseline_cost(all_jobs, pool_B, pool_C)
    print(f"SOT baseline fitness: {bl_cost:.2f}")
    print(f"Evaluation seeds    : {len(args.eval_seeds)} seeds  "
          f"({args.eval_seeds[0]}–{args.eval_seeds[-1]})")
    print(f"Tuning seeds        : {len(args.tuning_seeds)} seeds  "
          f"({args.tuning_seeds[0]}–{args.tuning_seeds[-1]})")
    print(f"Parallel workers    : {args.n_workers}")

    only   = args.stage
    sanity: list[SanityResult]      = []
    tuning: dict[str, TuningResult] = {}
    runs:   list[BenchmarkRun]      = []
    p_df = sig_df = a12_df = ttest_p_df = ttest_sig_df = None
    alpha_corr = _ALPHA

    # ── Stage 1 ────────────────────────────────────────────────
    if only in (None, 1):
        if args.n_datasets > 1:
            _s1_seed = args.dataset_seed + args.n_datasets + args.tuning_n_datasets + 2
            s1_jobs  = _sample_jobs_bc(all_jobs, args.subset_size, seed=_s1_seed)
            s1_bl    = _baseline_cost(s1_jobs, pool_B, pool_C)
            print(f"\n  Stage 1: {len(s1_jobs)}-job representative subset")
        else:
            s1_jobs, s1_bl = all_jobs, bl_cost
        sanity = run_sanity_checks(s1_jobs, s1_bl, pool_B, pool_C)

    # ── Stage 2 ────────────────────────────────────────────────
    if only in (None, 2) and not args.skip_tuning:
        if args.n_datasets > 1:
            _tune_seeds = [args.dataset_seed + args.n_datasets + 1 + i
                           for i in range(args.tuning_n_datasets)]
            tuning_job_sets = [_sample_jobs_bc(all_jobs, args.subset_size, seed=s)
                               for s in _tune_seeds]
            print(f"\n  Tuning across {args.tuning_n_datasets} random "
                  f"{args.subset_size}-job subsets")
        else:
            tuning_job_sets = [all_jobs]
        new_tuning = run_tuning(tuning_job_sets, args.tuning_seeds,
                               pool_B, pool_C, args.n_workers,
                               tune_only=args.tune_only)
        # Merge: keep existing tuning for any optimizer not re-tuned
        tuning = {**tuning, **new_tuning}
        export_tuning_table(tuning)

    # ── Stage 21 ───────────────────────────────────────────────
    if only in (None, 21) and not args.skip_tuning:
        if args.n_datasets > 1:
            _bseed = [args.dataset_seed + args.n_datasets + 1 + i
                      for i in range(args.tuning_n_datasets)]
            budget_job_sets = [_sample_jobs_bc(all_jobs, args.subset_size, seed=s)
                               for s in _bseed]
        else:
            budget_job_sets = [all_jobs]
        budget_results = run_budget_tuning(
            budget_job_sets, args.tuning_seeds, tuning,
            pool_B, pool_C, args.n_workers)
        plot_budget_sensitivity(budget_results)
        plot_budget_combined(budget_results)
        export_budget_data(budget_results)

        print("\n  Budget sweep summary:")
        for name, br in budget_results.items():
            print(f"    {name.upper():<8}  best {br.param}={br.best_value}"
                  f"  (cost={br.means[br.values.index(br.best_value)]:.2f})")

    # ── Stage 3 ────────────────────────────────────────────────
    if only in (None, 3):
        if args.n_datasets > 1:
            from collections import defaultdict as _dd
            print(f"\n  Dataset generalisation: {args.n_datasets} random "
                  f"{args.subset_size}-job subsets")
            _all: dict[str, list[BenchmarkRun]] = _dd(list)
            _ds_baselines: list[float] = []
            _stage3_t0 = time.perf_counter()
            for _ds in range(args.n_datasets):
                _ds_seed  = args.dataset_seed + _ds
                _ds_jobs  = _sample_jobs_bc(all_jobs, args.subset_size, seed=_ds_seed)
                _ds_bl    = _baseline_cost(_ds_jobs, pool_B, pool_C)
                _ds_baselines.append(_ds_bl)
                print(f"\n  --- Dataset {_ds+1}/{args.n_datasets}  "
                      f"n_jobs={len(_ds_jobs)}  baseline={_ds_bl:.2f} ---")
                _ds_runs = run_comparison(
                    _ds_jobs, tuning, args.eval_seeds, _ds_bl,
                    pool_B, pool_C, args.budget, args.n_workers)
                for _br in _ds_runs:
                    _all[_br.name].append(_br)
                _elapsed  = time.perf_counter() - _stage3_t0
                _avg_per  = _elapsed / (_ds + 1)
                _remaining = _avg_per * (args.n_datasets - _ds - 1)
                print(f"  [Dataset {_ds+1}/{args.n_datasets} done]  "
                      f"elapsed={_elapsed:.0f}s  "
                      f"ETA remaining≈{_remaining:.0f}s")
            # Use mean subset baseline so plots are on the same scale as optimizer results
            bl_cost = float(np.mean(_ds_baselines))
            print(f"\n  Mean subset baseline: {bl_cost:.2f}  "
                  f"(range {min(_ds_baselines):.2f}–{max(_ds_baselines):.2f})")
            runs = []
            for _name, _rlist in _all.items():
                _merged = BenchmarkRun(name=_name,
                                       config_desc=_rlist[0].config_desc,
                                       config_obj=_rlist[0].config_obj)
                _merged_best_cost = float("inf")
                for _r in _rlist:
                    _merged.costs.extend(_r.costs)
                    _merged.times_s.extend(_r.times_s)
                    _merged.histories.extend(_r.histories)
                    _merged.metrics.extend(_r.metrics)
                    _r_best = min(_r.costs, default=float("inf"))
                    if _r.best_chromosome and _r_best < _merged_best_cost:
                        _merged_best_cost = _r_best
                        _merged.best_chromosome = _r.best_chromosome[:]
                runs.append(_merged)
        else:
            runs = run_comparison(all_jobs, tuning, args.eval_seeds, bl_cost,
                                  pool_B, pool_C, args.budget, args.n_workers)

        # Add SOT baseline as a named run so Stage 5 can DES-evaluate it
        _sot_jobs = (_sample_jobs_bc(all_jobs, args.subset_size, seed=args.dataset_seed)
                     if args.n_datasets > 1 else all_jobs)
        _sot_br = BenchmarkRun(name="sot", config_desc="SOT baseline", config_obj=None,
                               costs=[bl_cost],
                               best_chromosome=_sot_chromosome(_sot_jobs, pool_B, pool_C))

        # Export CSVs first so Stage 4 always reads data from this run
        export_convergence_data(runs, bl_cost)
        export_boxplot_data(runs)
        export_chromosomes(runs + [_sot_br])

        plot_convergence(runs, bl_cost, path="bc_convergence.png",                       zoom=False, show_baseline=False)
        plot_convergence(runs, bl_cost, path="bc_convergence_with_baseline.png",          zoom=False, show_baseline=True)
        plot_convergence(runs, bl_cost, path="bc_convergence_zoomed.png",                 zoom=True,  show_baseline=False)
        plot_convergence(runs, bl_cost, path="bc_convergence_zoomed_with_baseline.png",   zoom=True,  show_baseline=True)
        plot_boxplots(runs, bl_cost, path="bc_boxplots.png",               show_baseline=False)
        plot_boxplots(runs, bl_cost, path="bc_boxplots_with_baseline.png", show_baseline=True)
        plot_cost_bars(runs, bl_cost, path="bc_cost_bars.png",               show_baseline=False)
        plot_cost_bars(runs, bl_cost, path="bc_cost_bars_with_baseline.png", show_baseline=True)
        plot_time_quality(runs)

        p_df, sig_df, a12_df, ttest_p_df, ttest_sig_df, alpha_corr = compute_significance_table(runs)
        plot_significance_matrix(p_df, sig_df, a12_df, alpha_corr,
                                 ttest_p_df=ttest_p_df, ttest_sig_df=ttest_sig_df)

        df = export_summary(sanity, runs, p_df, sig_df, bl_cost)
        export_significance(p_df, sig_df, a12_df, alpha_corr,
                            ttest_p_df=ttest_p_df, ttest_sig_df=ttest_sig_df)

        if only in (None, 5) or only == 3:
            # DES jobs must have job_index 0..N-1 matching the stored chromosome length.
            # When using subsets, sample a representative set of the same size.
            if args.n_datasets > 1:
                _des_jobs = _sample_jobs_bc(all_jobs, args.subset_size,
                                            seed=args.dataset_seed)
            else:
                _des_jobs = all_jobs
            des_res = run_des_comparison(
                _des_jobs, runs, pool_B, pool_C,
                n_reps=args.des_reps, seed=42)
            if des_res:
                plot_det_vs_des(runs, des_res)
                plot_des_distribution(runs, des_res)
                plot_des_percentiles(des_res, runs)
                export_des_results(runs, des_res)

        print("\n" + "=" * 60)
        print("  FINAL SUMMARY")
        print("=" * 60)
        cols = ["optimizer", "n_runs", "mean_cost", "std_cost",
                "best_cost", "vs_baseline_pct", "mean_time_s"]
        present = [c for c in cols if c in df.columns]
        print(df[present].to_string(index=False))

        print(f"\n  Wilcoxon p-values  (Bonferroni α = {alpha_corr:.4f}):")
        print(p_df.round(4).to_string())
        print(f"\n  Paired t-test p-values  (Bonferroni α = {alpha_corr:.4f}):")
        print(ttest_p_df.round(4).to_string())

    # ── Stage 4 ────────────────────────────────────────────────
    if only == 4:
        run_plots_from_csv(bl_cost)

    # ── Stage 5 (standalone) ────────────────────────────────────
    if only == 5:
        chrom_path = "bc_chromosomes.csv"
        if not os.path.exists(chrom_path):
            print("  [error] bc_chromosomes.csv not found — run Stage 3 first.")
        else:
            df_ch = pd.read_csv(chrom_path, sep=";")
            # Pre-load summary once for cost lookup
            _sumdf = (pd.read_csv("bc_summary.csv", sep=";", decimal=",")
                      if os.path.exists("bc_summary.csv") else None)
            fake_runs = []
            for _, row in df_ch.iterrows():
                chrom = list(map(int, str(row["chromosome"]).split()))
                # Priority: det_cost column in chromosomes CSV (covers sot + all optimizers)
                if "det_cost" in row and pd.notna(row["det_cost"]):
                    det_cost = float(row["det_cost"])
                elif _sumdf is not None:
                    match = _sumdf[_sumdf["optimizer"] == row["optimizer"]]
                    det_cost = float(match["mean_cost"].iloc[0]) if len(match) else 0.0
                else:
                    det_cost = 0.0
                br = BenchmarkRun(name=row["optimizer"], config_desc="loaded",
                                  config_obj=None, costs=[det_cost],
                                  best_chromosome=chrom)
                fake_runs.append(br)
            # Infer N from chromosome length; sample matching subset so job_index
            # values stay in 0..N-1 regardless of full dataset size.
            _chrom_n = len(fake_runs[0].best_chromosome) // 3 if fake_runs else len(all_jobs)
            if _chrom_n < len(all_jobs):
                _des_jobs5 = _sample_jobs_bc(all_jobs, _chrom_n, seed=args.dataset_seed)
            else:
                _des_jobs5 = all_jobs
            # Fix any zero det_cost for sot by computing it fresh
            for _br in fake_runs:
                if _br.name == "sot" and _br.mean_cost == 0.0:
                    _br.costs = [_baseline_cost(_des_jobs5, pool_B, pool_C)]
            des_res = run_des_comparison(
                _des_jobs5, fake_runs, pool_B, pool_C,
                n_reps=args.des_reps, seed=42)
            if des_res:
                plot_det_vs_des(fake_runs, des_res)
                plot_des_distribution(fake_runs, des_res)
                plot_des_percentiles(des_res, fake_runs)
                export_des_results(fake_runs, des_res)

    print("\nDone.")
