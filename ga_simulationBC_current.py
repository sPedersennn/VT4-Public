"""
ga_simulationBC_current.py
==========================
Variant of ga_simulationBC.py that loads from current_status.csv instead of
data_real.csv.

current_status.csv columns
---------------------------
  total_remaining  â€” remaining repair hours for the blade
  buffer           â€” 0 = primary job (must be processed)
                     1 = buffer job (available; GA decides when to pull it in)

All 15 jobs go through the normal repair-track â†’ oven pipeline.  The
chromosome covers all jobs.  To warm-start the GA the chromosome initialiser
seeds primary jobs (buffer=0) with low dispatch-priority values so they are
scheduled first by default, while buffer jobs (buffer=1) receive high priority
values so they naturally slot in after the primary workload â€” the GA is free
to override this ordering if it reduces the makespan.

Everything else (phases, SA worker refinement, constraint enforcement, Gantt
chart, convergence plot) is identical to ga_simulationBC.py.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from constraintsBC import (
    BLADE_QUEUE_LOOKAHEAD,
    CHANGEOVER_TIME,
    MAX_REPAIR_TRACKS,
    MAX_WEEKLY_HOURS,
    MAX_WORKERS_PER_BLADE,
    MAX_WORKERS_TOTAL_B,
    MAX_WORKERS_TOTAL_C,
    MIN_PROCESS_TIME_RATIO,
    OVEN_PROCESS_TIME,
    WORKER_REASSIGN_INTERVAL,
    efficiency_factor,
    penalty_worker_band,
    penalty_oven_idle,
    enforce_full_worker_utilization_B,
    enforce_full_worker_utilization_C,
    enforce_empty_track_fill,
)

# ---------------------------------------------------------------------------
# Top-level constants
# ---------------------------------------------------------------------------

WEEK_HOURS:  float = MAX_WEEKLY_HOURS
N_WORKERS_B: int   = 35
N_WORKERS_C: int   = 25

PHASE1_H: float = 9.0
PHASE3_H: float = 27.0

STATUS_CSV:    str = os.path.join(os.path.dirname(__file__), "current_status.csv")
DATA_REAL_CSV: str = os.path.join(os.path.dirname(__file__), "data_real.csv")

PERT_GAMMA:         float = 60.0
MAX_GENERATED_JOBS: int   = 35   # max new jobs spawned by buffer replenishment per simulation

HC2_EFF_CAP: float = 1.0 / MIN_PROCESS_TIME_RATIO

SC1_WEIGHT: float = 0.1
SC2_WEIGHT: float = 10.0
SC3_WEIGHT: float = 10.0

GA_POP_SIZE:       int   = 45
GA_GENERATIONS:    int   = 200
GA_CROSSOVER_RATE: float = 0.5
GA_MUTATION_RATE:  float = 0.05
GA_ELITISM_N:      int   = 3
GA_TOURNAMENT_K:   int   = 4
GA_SEED:           int   = 42

SA_INTERVAL:   float = 4.0
SA_ITERATIONS: int   = 50
SA_TEMP_INIT:  float = 50.0
SA_COOLING:    float = 0.97


# ---------------------------------------------------------------------------
# PERT job-duration distribution  (fitted from data_real.csv)
# ---------------------------------------------------------------------------

def _load_pert_params() -> tuple[float, float, float]:
    """Return (a, mode, b) from the Estimated column of data_real.csv."""
    ests = []
    with open(DATA_REAL_CSV, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            ests.append(float(row["Estimated"]))
    arr = np.array(ests)
    a   = float(arr.min())
    b   = float(arr.max())
    counts, edges = np.histogram(arr, bins="auto")
    best = int(np.argmax(counts))
    mode = float((edges[best] + edges[best + 1]) / 2)
    return a, mode, b


def pert_sample(rng: random.Random,
                a: float, m: float, b: float,
                gamma: float = PERT_GAMMA) -> float:
    """Draw one sample from a PERT(a, m, b, gamma) distribution."""
    alpha1 = 1.0 + gamma * (m - a) / (b - a)
    alpha2 = 1.0 + gamma * (b - m) / (b - a)
    z = rng.betavariate(alpha1, alpha2)
    return a + z * (b - a)


# Computed once at import time so every simulation run uses the same parameters.
_PERT_A, _PERT_MODE, _PERT_B = _load_pert_params()


# ---------------------------------------------------------------------------
# Phase helper
# ---------------------------------------------------------------------------

def _compute_phases(estimated_h: float) -> tuple[float, float, float]:
    p1 = min(PHASE1_H, estimated_h)
    remainder = estimated_h - p1
    p3 = min(PHASE3_H, remainder)
    p2 = max(0.0, remainder - p3)
    return p1, p2, p3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Job:
    job_id:      str
    estimated_h: float
    job_index:   int  = 0
    is_buffered: bool = False   # True = buffer job (goes later by default)
    skip_phase1: bool = False   # True = first 12 primary jobs (no phase 1)

    phase1_h: float = 0.0
    phase2_h: float = 0.0
    phase3_h: float = 0.0

    repair_track: Optional[int]   = None
    repair_start: Optional[float] = None
    repair_end:   Optional[float] = None
    oven_start:   Optional[float] = None
    oven_end:     Optional[float] = None

    phase1_end: Optional[float] = None
    phase2_end: Optional[float] = None

    worker_log: list = field(default_factory=list)

    def __post_init__(self):
        if self.skip_phase1:
            self.phase1_h = 0.0
            remainder     = self.estimated_h
            self.phase3_h = min(PHASE3_H, remainder)
            self.phase2_h = max(0.0, remainder - self.phase3_h)
        else:
            self.phase1_h, self.phase2_h, self.phase3_h = _compute_phases(self.estimated_h)

    @property
    def avg_workers_B(self) -> float:
        entries = [(s, e, w) for s, e, w, tp in self.worker_log if tp == 'B' and w > 0]
        if not entries:
            return 0.0
        wh = sum((e - s) * w for s, e, w in entries)
        h  = sum((e - s)     for s, e, _ in entries)
        return wh / h if h > 0 else 0.0

    @property
    def avg_workers_C(self) -> float:
        entries = [(s, e, w) for s, e, w, tp in self.worker_log if tp == 'C' and w > 0]
        if not entries:
            return 0.0
        wh = sum((e - s) * w for s, e, w in entries)
        h  = sum((e - s)     for s, e, _ in entries)
        return wh / h if h > 0 else 0.0


@dataclass
class _ActiveSlot:
    job:       Job
    track_idx: int
    p1_rem:    float
    p2_rem:    float
    p3_rem:    float
    workers_B: int = 0
    workers_C: int = 0

    @property
    def phase(self) -> int:
        if self.p1_rem > 1e-9: return 1
        if self.p2_rem > 1e-9: return 2
        return 3

    @property
    def remaining_h(self) -> float:
        return self.p1_rem + self.p2_rem + self.p3_rem


@dataclass
class _TrialSlot:
    p1_rem:       float
    p2_rem:       float
    p3_rem:       float
    repair_start: float
    estimated_h:  float
    repair_end:   Optional[float] = None
    workers_B:    int = 0
    workers_C:    int = 0

    @property
    def remaining_h(self) -> float:
        return self.p1_rem + self.p2_rem + self.p3_rem

    @property
    def phase(self) -> int:
        if self.p1_rem > 1e-9: return 1
        if self.p2_rem > 1e-9: return 2
        return 3


# ---------------------------------------------------------------------------
# Lookahead / forward simulation helpers
# ---------------------------------------------------------------------------

def _advance_trial_slot(s: _TrialSlot, t: float, dt: float) -> None:
    phase = s.phase
    w = s.workers_C if phase in (1, 3) else s.workers_B
    eff = min(efficiency_factor(w), HC2_EFF_CAP)
    if eff <= 0:
        return
    work = dt * eff

    if phase == 1:
        if work >= s.p1_rem:
            phase_end = t + s.p1_rem / eff
            s.p1_rem = 0.0
            if s.p2_rem <= 1e-9 and s.p3_rem <= 1e-9:
                s.repair_end = max(phase_end,
                                   s.repair_start + MIN_PROCESS_TIME_RATIO * s.estimated_h)
        else:
            s.p1_rem -= work
    elif phase == 2:
        if work >= s.p2_rem:
            phase_end = t + s.p2_rem / eff
            s.p2_rem = 0.0
            if s.p3_rem <= 1e-9:
                s.repair_end = max(phase_end,
                                   s.repair_start + MIN_PROCESS_TIME_RATIO * s.estimated_h)
        else:
            s.p2_rem -= work
    else:
        if work >= s.p3_rem:
            raw_end = t + s.p3_rem / eff
            s.repair_end = max(raw_end,
                               s.repair_start + MIN_PROCESS_TIME_RATIO * s.estimated_h)
            s.p3_rem = 0.0
        else:
            s.p3_rem -= work


def _forward_makespan_bc(
    trial_slots:  list,
    slot_free_at: list[float],
    oven_free_at: float,
    queue_ests:   list[float],
    t:            float,
    pool_B:       int,
    pool_C:       int,
    oven_queue:   Optional[list[float]] = None,
) -> float:
    dt    = WORKER_REASSIGN_INTERVAL * 2
    max_t = t + 15_000.0
    last_oven_end = oven_free_at
    pending: list[float] = sorted(oven_queue) if oven_queue else []

    while t < max_t:
        i = 0
        while i < len(pending):
            rep_end  = pending[i]
            ov_start = max(oven_free_at, rep_end)
            if ov_start > t:
                break
            ov_end        = ov_start + OVEN_PROCESS_TIME
            last_oven_end = max(last_oven_end, ov_end)
            oven_free_at  = ov_end + CHANGEOVER_TIME
            pending.pop(i)

        for i in range(MAX_REPAIR_TRACKS):
            if trial_slots[i] is None and slot_free_at[i] <= t and queue_ests:
                picks = queue_ests[:BLADE_QUEUE_LOOKAHEAD]
                si    = min(range(len(picks)), key=lambda bi: picks[bi])
                est   = queue_ests.pop(si)
                p1, p2, p3 = _compute_phases(est)
                trial_slots[i] = _TrialSlot(p1_rem=p1, p2_rem=p2, p3_rem=p3,
                                             repair_start=t, estimated_h=est)

        c_active = sorted(
            [s for s in trial_slots
             if s is not None and s.remaining_h > 0 and s.phase in (1, 3)],
            key=lambda x: x.remaining_h,
        )
        b_active = sorted(
            [s for s in trial_slots
             if s is not None and s.remaining_h > 0 and s.phase == 2],
            key=lambda x: x.remaining_h,
        )
        for s in c_active: s.workers_C = 0
        for s in b_active: s.workers_B = 0

        pool_left_C = pool_C
        for s in c_active:
            if pool_left_C <= 0: break
            s.workers_C = 1; pool_left_C -= 1
        for s in c_active:
            if pool_left_C <= 0: break
            give = min(MAX_WORKERS_PER_BLADE - s.workers_C, pool_left_C)
            s.workers_C += give; pool_left_C -= give

        pool_left_B = pool_B
        for s in b_active:
            if pool_left_B <= 0: break
            s.workers_B = 1; pool_left_B -= 1
        for s in b_active:
            if pool_left_B <= 0: break
            give = min(MAX_WORKERS_PER_BLADE - s.workers_B, pool_left_B)
            s.workers_B += give; pool_left_B -= give

        newly_done: list[float] = []
        for idx, s in enumerate(trial_slots):
            if s is not None and s.remaining_h > 0:
                _advance_trial_slot(s, t, dt)
                if s.remaining_h <= 1e-9 and s.repair_end is not None:
                    newly_done.append(s.repair_end)
                    slot_free_at[idx] = s.repair_end + CHANGEOVER_TIME
                    trial_slots[idx]  = None

        for rep_end in newly_done:
            pending.append(rep_end)
        pending.sort()

        if not queue_ests and not pending and all(s is None for s in trial_slots):
            break
        t += dt

    for rep_end in pending:
        ov_start      = max(oven_free_at, rep_end)
        ov_end        = ov_start + OVEN_PROCESS_TIME
        last_oven_end = max(last_oven_end, ov_end)
        oven_free_at  = ov_end + CHANGEOVER_TIME

    return last_oven_end


def _pick_best_candidate_bc(
    candidates:   list[Job],
    track_i:      int,
    slots:        list,
    slot_free_at: list[float],
    oven_free_at: float,
    queue:        list[Job],
    t:            float,
    pool_B:       int,
    pool_C:       int,
    oven_queue:   Optional[list[float]] = None,
) -> int:
    best_idx      = 0
    best_makespan = float("inf")

    for ci, cand in enumerate(candidates):
        trial_slots = []
        for s in slots:
            if s is None:
                trial_slots.append(None)
            else:
                trial_slots.append(_TrialSlot(
                    p1_rem=s.p1_rem, p2_rem=s.p2_rem, p3_rem=s.p3_rem,
                    repair_start=s.job.repair_start,
                    estimated_h=s.job.estimated_h,
                ))

        trial_slots[track_i] = _TrialSlot(p1_rem=cand.phase1_h, p2_rem=cand.phase2_h, p3_rem=cand.phase3_h,
                                           repair_start=t, estimated_h=cand.estimated_h)
        trial_queue = [j.estimated_h for j in queue if j is not cand]
        ms = _forward_makespan_bc(trial_slots, list(slot_free_at), oven_free_at,
                                   trial_queue, t, pool_B, pool_C, oven_queue)
        if ms < best_makespan:
            best_makespan = ms
            best_idx      = ci

    return best_idx


# ---------------------------------------------------------------------------
# Chromosome operators
# ---------------------------------------------------------------------------

def random_chromosome(jobs: list[Job], rng: random.Random) -> list[int]:
    """Length 3*n: priorities then B-worker targets then C-worker targets.

    The simulation hard-enforces that all primary jobs are dispatched before
    any buffer job, so priority genes only control ordering within each group.
    """
    n     = len(jobs)
    prios = [rng.randint(0, 2 * n - 1) for _ in range(n)]
    workers = [rng.randint(0, MAX_WORKERS_PER_BLADE) for _ in range(2 * n)]
    return prios + workers


def two_point_crossover(p1: list[int], p2: list[int],
                        rng: random.Random) -> tuple[list[int], list[int]]:
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))
    return p1[:a] + p2[a:b] + p1[b:], p2[:a] + p1[a:b] + p2[b:]


def mutate(chrom: list[int], rate: float, n: int,
           rng: random.Random) -> list[int]:
    return [
        (rng.randint(0, 2 * n - 1) if i < n else rng.randint(0, MAX_WORKERS_PER_BLADE))
        if rng.random() < rate else g
        for i, g in enumerate(chrom)
    ]


def tournament_select(pop: list[list[int]], fits: list[float],
                      k: int, rng: random.Random) -> list[int]:
    best = min(rng.sample(range(len(pop)), k), key=lambda i: fits[i])
    return pop[best][:]


# ---------------------------------------------------------------------------
# Worker allocation helpers
# ---------------------------------------------------------------------------

def _cap_allocation(alloc: dict[int, int], pool: int) -> dict[int, int]:
    alloc = {k: max(0, min(MAX_WORKERS_PER_BLADE, v)) for k, v in alloc.items()}
    total = sum(alloc.values())
    if total > pool:
        scale = pool / total
        alloc = {k: max(0, int(v * scale)) for k, v in alloc.items()}
    return alloc


# ---------------------------------------------------------------------------
# SA helpers
# ---------------------------------------------------------------------------

def _sa_fitness_bc(
    alloc_b:      dict[int, int],
    alloc_c:      dict[int, int],
    slots:        list,
    slot_free_at: list[float],
    oven_free_at: float,
    queue:        list,
    t:            float,
    pool_B:       int,
    pool_C:       int,
    oven_queue:   Optional[list[float]] = None,
) -> float:
    dt = WORKER_REASSIGN_INTERVAL
    trial_slots  = []
    new_pending: list[float] = []
    trial_sfa    = list(slot_free_at)

    for idx, s in enumerate(slots):
        if s is None:
            trial_slots.append(None)
        else:
            phase = s.phase
            w = alloc_c.get(s.job.job_index, 0) if phase in (1, 3) \
                else alloc_b.get(s.job.job_index, 0)
            eff = min(efficiency_factor(w), HC2_EFF_CAP)
            p1, p2, p3 = s.p1_rem, s.p2_rem, s.p3_rem
            repair_end = None
            if eff > 0:
                work = dt * eff
                if phase == 1:
                    if work >= p1:
                        phase_end = t + p1 / eff
                        p1 = 0.0
                        if p2 <= 1e-9 and p3 <= 1e-9:
                            repair_end = max(phase_end,
                                             s.job.repair_start + MIN_PROCESS_TIME_RATIO * s.job.estimated_h)
                    else:
                        p1 -= work
                elif phase == 2:
                    if work >= p2:
                        phase_end = t + p2 / eff
                        p2 = 0.0
                        if p3 <= 1e-9:
                            repair_end = max(phase_end,
                                             s.job.repair_start + MIN_PROCESS_TIME_RATIO * s.job.estimated_h)
                    else:
                        p2 -= work
                else:
                    if work >= p3:
                        raw_end = t + p3 / eff
                        repair_end = max(raw_end,
                                         s.job.repair_start + MIN_PROCESS_TIME_RATIO * s.job.estimated_h)
                        p3 = 0.0
                    else:
                        p3 -= work

            if repair_end is not None and p1 <= 1e-9 and p2 <= 1e-9 and p3 <= 1e-9:
                new_pending.append(repair_end)
                trial_sfa[idx] = repair_end + CHANGEOVER_TIME
                trial_slots.append(None)
            else:
                trial_slots.append(_TrialSlot(p1_rem=p1, p2_rem=p2, p3_rem=p3,
                                               repair_start=s.job.repair_start,
                                               estimated_h=s.job.estimated_h,
                                               repair_end=repair_end))

    combined_oq = sorted((oven_queue or []) + new_pending)
    queue_ests  = [j.estimated_h for j in queue]
    return _forward_makespan_bc(trial_slots, trial_sfa, oven_free_at,
                                 queue_ests, t + dt, pool_B, pool_C, combined_oq)


def _sa_neighbor_pool(alloc: dict[int, int], keys: list[int],
                      rng: random.Random) -> dict[int, int]:
    new       = dict(alloc)
    donors    = [k for k in keys if new.get(k, 0) > 0]
    receivers = [k for k in keys if new.get(k, 0) < MAX_WORKERS_PER_BLADE]
    if not donors or not receivers:
        return new
    donor   = rng.choice(donors)
    options = [r for r in receivers if r != donor]
    if not options:
        return new
    new[donor]              -= 1
    new[rng.choice(options)] += 1
    return new


def _sa_allocate_bc(
    active_b:     list,
    active_c:     list,
    slots:        list,
    slot_free_at: list[float],
    oven_free_at: float,
    queue:        list,
    t:            float,
    pool_B:       int,
    pool_C:       int,
    seed_alloc_b: dict[int, int],
    seed_alloc_c: dict[int, int],
    rng:          random.Random,
    oven_queue:   Optional[list[float]] = None,
) -> tuple[dict[int, int], dict[int, int]]:
    keys_b = [s.job.job_index for s in active_b]
    keys_c = [s.job.job_index for s in active_c]

    current_b = {k: seed_alloc_b.get(k, 0) for k in keys_b}
    current_c = {k: seed_alloc_c.get(k, 0) for k in keys_c}
    current_b = _cap_allocation(current_b, pool_B)
    current_c = _cap_allocation(current_c, pool_C)
    if keys_b:
        current_b = enforce_full_worker_utilization_B(current_b, pool_B)
    if keys_c:
        current_c = enforce_full_worker_utilization_C(current_c, pool_C)

    current_fit = _sa_fitness_bc(current_b, current_c, slots, slot_free_at,
                                  oven_free_at, queue, t, pool_B, pool_C, oven_queue)
    best_b, best_c = dict(current_b), dict(current_c)
    best_fit = current_fit
    temp = SA_TEMP_INIT

    for step in range(SA_ITERATIONS):
        if step % 2 == 0 and keys_b:
            nb = _sa_neighbor_pool(current_b, keys_b, rng)
            nc = current_c
        elif keys_c:
            nb = current_b
            nc = _sa_neighbor_pool(current_c, keys_c, rng)
        else:
            break

        nfit = _sa_fitness_bc(nb, nc, slots, slot_free_at,
                               oven_free_at, queue, t, pool_B, pool_C, oven_queue)
        delta = nfit - current_fit
        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
            current_b, current_c, current_fit = nb, nc, nfit
            if current_fit < best_fit:
                best_b, best_c, best_fit = dict(current_b), dict(current_c), current_fit
        temp *= SA_COOLING

    return best_b, best_c


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_jobs(csv_path: str = STATUS_CSV) -> list[Job]:
    """Load current_status.csv.

    Returns all jobs in file order.  Jobs with buffer=1 have is_buffered=True
    and are scheduled after primary jobs by default (see random_chromosome).
    The first 12 primary (non-buffer) jobs have skip_phase1=True.
    """
    jobs = []
    primary_count = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for i, row in enumerate(reader):
            remaining = float(row["total_remaining"])
            buffered  = int(row["buffer"])
            prefix    = "BUF" if buffered else "JOB"
            is_buf    = bool(buffered)
            # If the CSV has an explicit skip_phase1 column use it directly;
            # otherwise fall back to the old rule (first 12 primary blades).
            sp1_raw = row.get("skip_phase1", "").strip()
            if sp1_raw != "":
                skip_p1 = bool(int(sp1_raw))
            else:
                skip_p1 = False
                if not is_buf:
                    primary_count += 1
                    skip_p1 = primary_count <= 12
            jobs.append(Job(
                job_id=f"{prefix}-{i+1:03d}",
                estimated_h=remaining,
                job_index=i,
                is_buffered=is_buf,
                skip_phase1=skip_p1,
            ))
    return jobs


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def simulate_ga_bc(
    jobs:          list[Job],
    chromosome:    list[int],
    pool_B:        int = 0,
    pool_C:        int = 0,
    use_sa:        bool = False,
    use_lookahead: bool = False,
    gen_seed:      int  = GA_SEED + 1,
) -> tuple[list[tuple[float, int, int]], float, float, float]:
    """Run one simulation for all jobs driven by chromosome worker targets.

    Primary (buffer=0) jobs are always dispatched before buffer (buffer=1)
    jobs.  Each time a buffer job is placed on a repair track a new job is
    drawn from the PERT distribution and appended to the end of the queue
    (up to MAX_GENERATED_JOBS total new arrivals).  Generated jobs are
    appended to the `jobs` list in-place so callers can inspect them.

    Returns (staff_log, sc1_total, sc2_total, sc3_total).
    """
    n = len(jobs)
    if pool_B <= 0: pool_B = N_WORKERS_B
    if pool_C <= 0: pool_C = N_WORKERS_C

    sa_rng    = random.Random(GA_SEED) if use_sa else None
    gen_rng   = random.Random(gen_seed)
    next_sa_t = 0.0
    dt = WORKER_REASSIGN_INTERVAL

    # Default worker counts used for jobs generated at runtime (no chromosome gene).
    buf_indices = [j.job_index for j in jobs if j.is_buffered]
    if buf_indices:
        default_b = max(1, round(sum(chromosome[n + i] for i in buf_indices) / len(buf_indices)))
        default_c = max(1, round(sum(chromosome[2 * n + i] for i in buf_indices) / len(buf_indices)))
    else:
        default_b = MAX_WORKERS_PER_BLADE // 2
        default_c = MAX_WORKERS_PER_BLADE // 2

    generated_count = 0   # how many new jobs have been spawned so far

    slots:        list[Optional[_ActiveSlot]] = [None] * MAX_REPAIR_TRACKS
    slot_free_at: list[float]                 = [0.0]  * MAX_REPAIR_TRACKS
    oven_free_at: float = 0.0

    # Primary jobs always come before buffer jobs; within each group the
    # chromosome priority gene controls the order.
    primary_q = sorted([j for j in jobs if not j.is_buffered], key=lambda j: chromosome[j.job_index])
    buffer_q  = sorted([j for j in jobs if j.is_buffered],     key=lambda j: chromosome[j.job_index])
    queue:      list[Job] = primary_q + buffer_q
    oven_queue: list[Job]                    = []
    staff_log:  list[tuple[float, int, int]] = []
    sc1_total:  float = 0.0
    oven_starts: list[float] = []
    oven_ends:   list[float] = []

    t     = 0.0
    max_t = 15_000.0

    while t < max_t:

        # Step 1 â€” transfer staged blades to oven (FIFO by repair_end)
        oven_queue.sort(key=lambda j: j.repair_end)
        i = 0
        while i < len(oven_queue):
            job      = oven_queue[i]
            ov_start = max(oven_free_at, job.repair_end)
            if ov_start > t:
                break
            job.oven_start = ov_start
            job.oven_end   = ov_start + OVEN_PROCESS_TIME
            oven_free_at   = job.oven_end + CHANGEOVER_TIME
            oven_starts.append(ov_start)
            oven_ends.append(job.oven_end)
            oven_queue.pop(i)

        # Step 2 â€” fill free tracks
        for i in range(MAX_REPAIR_TRACKS):
            if slots[i] is None and slot_free_at[i] <= t and queue:
                job = queue.pop(0)

                # When a buffer job is dispatched, replenish the buffer with a
                # PERT-sampled arrival (up to MAX_GENERATED_JOBS total).
                if job.is_buffered and generated_count < MAX_GENERATED_JOBS:
                    new_h   = pert_sample(gen_rng, _PERT_A, _PERT_MODE, _PERT_B)
                    new_job = Job(
                        job_id      = f"GEN-{generated_count + 1:03d}",
                        estimated_h = new_h,
                        job_index   = n + generated_count,
                        is_buffered = True,
                    )
                    generated_count += 1
                    queue.append(new_job)   # goes to back of queue (after remaining buffer jobs)
                    jobs.append(new_job)    # tracked for SC-2/SC-3 and Gantt

                aj  = _ActiveSlot(job=job, track_idx=i,
                                  p1_rem=job.phase1_h, p2_rem=job.phase2_h, p3_rem=job.phase3_h)
                job.repair_track = i
                job.repair_start = t
                if job.skip_phase1:
                    job.phase1_end = t  # zero-length phase 1 so Gantt renders correctly
                slots[i] = aj
        enforce_empty_track_fill(slots, slot_free_at, queue, t)

        # Step 3 â€” allocate workers from chromosome (SA-refined if enabled)
        # Generated jobs (job_index >= n) fall back to default_b / default_c.
        active   = [aj for aj in slots if aj is not None]
        b_active = [aj for aj in active if aj.phase == 2]
        c_active = [aj for aj in active if aj.phase in (1, 3)]

        def _w_b(idx: int) -> int:
            ci = n + idx
            return chromosome[ci] if ci < len(chromosome) else default_b

        def _w_c(idx: int) -> int:
            ci = 2 * n + idx
            return chromosome[ci] if ci < len(chromosome) else default_c

        if active:
            raw_b = {aj.job.job_index: _w_b(aj.job.job_index) for aj in b_active}
            raw_c = {aj.job.job_index: _w_c(aj.job.job_index) for aj in c_active}

            oq_times = [j.repair_end for j in oven_queue]
            if use_sa and t >= next_sa_t:
                capped_b, capped_c = _sa_allocate_bc(
                    b_active, c_active, slots, slot_free_at, oven_free_at,
                    queue, t, pool_B, pool_C, raw_b, raw_c, sa_rng, oq_times,
                )
                next_sa_t = t + SA_INTERVAL
            else:
                capped_b = _cap_allocation(raw_b, pool_B)
                capped_c = _cap_allocation(raw_c, pool_C)
                if b_active:
                    capped_b = enforce_full_worker_utilization_B(capped_b, pool_B)
                if c_active:
                    capped_c = enforce_full_worker_utilization_C(capped_c, pool_C)

            for aj in b_active:
                w = capped_b.get(aj.job.job_index, 0)
                aj.workers_B = w
                aj.workers_C = 0
                sc1_total += penalty_worker_band(w) * dt

            for aj in c_active:
                w = capped_c.get(aj.job.job_index, 0)
                aj.workers_C = w
                aj.workers_B = 0
                sc1_total += penalty_worker_band(w) * dt

        total_B = sum(aj.workers_B for aj in slots if aj is not None)
        total_C = sum(aj.workers_C for aj in slots if aj is not None)
        staff_log.append((t, total_B, total_C))

        # Step 4 â€” advance work by dt
        newly_done: list[_ActiveSlot] = []
        for aj in slots:
            if aj is None:
                continue

            phase = aj.phase
            if phase in (1, 3):
                w, worker_type = aj.workers_C, 'C'
            else:
                w, worker_type = aj.workers_B, 'B'

            eff = efficiency_factor(w)
            if eff <= 0.0:
                aj.job.worker_log.append((t, t + dt, 0, worker_type))
                continue

            eff       = min(eff, HC2_EFF_CAP)
            work_done = dt * eff

            if phase == 1:
                if work_done >= aj.p1_rem:
                    phase_end_t       = t + aj.p1_rem / eff
                    aj.job.worker_log.append((t, phase_end_t, w, 'C'))
                    aj.job.phase1_end = phase_end_t
                    aj.p1_rem         = 0.0
                    if aj.p2_rem <= 1e-9 and aj.p3_rem <= 1e-9:
                        min_end           = aj.job.repair_start + MIN_PROCESS_TIME_RATIO * aj.job.estimated_h
                        aj.job.repair_end = max(phase_end_t, min_end)
                        newly_done.append(aj)
                else:
                    aj.p1_rem -= work_done
                    aj.job.worker_log.append((t, t + dt, w, 'C'))

            elif phase == 2:
                if work_done >= aj.p2_rem:
                    phase_end_t       = t + aj.p2_rem / eff
                    aj.job.worker_log.append((t, phase_end_t, w, 'B'))
                    aj.job.phase2_end = phase_end_t
                    aj.p2_rem         = 0.0
                    if aj.p3_rem <= 1e-9:
                        min_end           = aj.job.repair_start + MIN_PROCESS_TIME_RATIO * aj.job.estimated_h
                        aj.job.repair_end = max(phase_end_t, min_end)
                        newly_done.append(aj)
                else:
                    aj.p2_rem -= work_done
                    aj.job.worker_log.append((t, t + dt, w, 'B'))

            else:
                if work_done >= aj.p3_rem:
                    raw_end           = t + aj.p3_rem / eff
                    min_end           = aj.job.repair_start + MIN_PROCESS_TIME_RATIO * aj.job.estimated_h
                    actual_end        = max(raw_end, min_end)
                    aj.job.worker_log.append((t, actual_end, w, 'C'))
                    aj.job.repair_end = actual_end
                    aj.p3_rem         = 0.0
                    newly_done.append(aj)
                else:
                    aj.p3_rem -= work_done
                    aj.job.worker_log.append((t, t + dt, w, 'C'))

        for aj in newly_done:
            oven_queue.append(aj.job)
            slot_free_at[aj.track_idx] = aj.job.repair_end + CHANGEOVER_TIME
            slots[aj.track_idx]        = None

        if not queue and not oven_queue and all(s is None for s in slots):
            break

        t += dt

    # Safety drain
    for job in sorted(oven_queue, key=lambda j: j.repair_end):
        ov_start       = max(oven_free_at, job.repair_end)
        job.oven_start = ov_start
        job.oven_end   = ov_start + OVEN_PROCESS_TIME
        oven_free_at   = job.oven_end + CHANGEOVER_TIME
        oven_starts.append(ov_start)
        oven_ends.append(job.oven_end)

    done = [j for j in jobs if j.oven_start is not None and j.repair_end is not None]
    sc2_total = sum(max(0.0, j.oven_start - j.repair_end) for j in done)

    if len(oven_starts) > 1:
        pairs    = sorted(zip(oven_starts, oven_ends))
        s_sorted = [p[0] for p in pairs]
        e_sorted = [p[1] for p in pairs]
        sc3_total = penalty_oven_idle(e_sorted, s_sorted)
    else:
        sc3_total = 0.0

    return staff_log, sc1_total, sc2_total, sc3_total


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------

def _fresh_jobs(template: list[Job]) -> list[Job]:
    return [Job(job_id=j.job_id, estimated_h=j.estimated_h,
                job_index=j.job_index, is_buffered=j.is_buffered,
                skip_phase1=j.skip_phase1)
            for j in template]


def evaluate(chromosome: list[int],
             jobs_template: list[Job],
             pool_B: int,
             pool_C: int) -> float:
    jobs_copy = _fresh_jobs(jobs_template)
    _, sc1, sc2, sc3 = simulate_ga_bc(jobs_copy, chromosome, pool_B, pool_C)
    done = [j for j in jobs_copy if j.oven_end is not None]
    makespan = max((j.oven_end for j in done), default=float('inf'))
    return makespan + SC2_WEIGHT * sc2 + SC3_WEIGHT * sc3


# ---------------------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------------------

def run_ga(
    jobs_template: list[Job],
    pool_B:        int   = 0,
    pool_C:        int   = 0,
    pop_size:      int   = 0,
    n_gen:         int   = 0,
    cx_rate:       float = GA_CROSSOVER_RATE,
    mut_rate:      float = GA_MUTATION_RATE,
    elitism_n:     int   = GA_ELITISM_N,
    tournament_k:  int   = GA_TOURNAMENT_K,
    seed:          int   = GA_SEED,
) -> tuple[list[int], float, list[float]]:
    if pool_B   <= 0: pool_B   = N_WORKERS_B
    if pool_C   <= 0: pool_C   = N_WORKERS_C
    if pop_size <= 0: pop_size = GA_POP_SIZE
    if n_gen    <= 0: n_gen    = GA_GENERATIONS

    rng = random.Random(seed)
    n   = len(jobs_template)

    pop  = [random_chromosome(jobs_template, rng) for _ in range(pop_size)]
    fits = [evaluate(c, jobs_template, pool_B, pool_C) for c in pop]

    history: list[float] = [min(fits)]
    print(f"  Gen   0/{n_gen}  best = {history[0]:.1f}")

    for gen in range(1, n_gen + 1):
        ranked   = sorted(range(pop_size), key=lambda i: fits[i])
        next_pop = [pop[i][:] for i in ranked[:elitism_n]]

        while len(next_pop) < pop_size:
            p1 = tournament_select(pop, fits, tournament_k, rng)
            p2 = tournament_select(pop, fits, tournament_k, rng)
            if rng.random() < cx_rate:
                c1, c2 = two_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1[:], p2[:]
            next_pop.append(mutate(c1, mut_rate, n, rng))
            if len(next_pop) < pop_size:
                next_pop.append(mutate(c2, mut_rate, n, rng))

        pop  = next_pop
        fits = [evaluate(c, jobs_template, pool_B, pool_C) for c in pop]

        best = min(fits)
        history.append(best)
        if gen % 10 == 0 or gen == n_gen:
            print(f"  Gen {gen:3d}/{n_gen}  best = {best:.1f}")

    best_idx = min(range(pop_size), key=lambda i: fits[i])
    return pop[best_idx], fits[best_idx], history


# ---------------------------------------------------------------------------
# Gantt chart
# ---------------------------------------------------------------------------

def _make_palette(n: int) -> list:
    base = [cm.tab20(i / 20) for i in range(20)] + \
           [cm.tab20b(i / 20) for i in range(20)]
    while len(base) < n:
        base += base
    return base[:n]


def plot_gantt(
    jobs:      list[Job],
    staff_log: list[tuple[float, int, int]],
    pool_B:    int = N_WORKERS_B,
    pool_C:    int = N_WORKERS_C,
    save_path: str = "gantt_ga_bc_current.png",
) -> None:

    done = [j for j in jobs if j.oven_end is not None]
    if not done:
        print("No finished jobs â€” nothing to plot.")
        return

    makespan = max(j.oven_end for j in done)
    palette  = _make_palette(len(jobs))
    colors   = {j.job_id: palette[k] for k, j in enumerate(jobs)}
    bar_h    = 0.72

    n_primary = sum(1 for j in jobs if not j.is_buffered)
    n_buf     = sum(1 for j in jobs if j.is_buffered and not j.job_id.startswith("GEN-"))
    n_gen     = sum(1 for j in jobs if j.job_id.startswith("GEN-"))

    fig, axes = plt.subplots(
        4, 1,
        figsize=(16, 14),
        gridspec_kw={"height_ratios": [MAX_REPAIR_TRACKS * 0.6, 1.8, 2.0, 2.0]},
    )
    fig.suptitle(
        f"GA (B/C workers) â€” current_status.csv  "
        f"({n_primary} primary + {n_buf} buffer + {n_gen} PERT-generated = {len(done)} completed)",
        fontsize=13, fontweight="bold",
    )

    # ---- Panel 1: Repair tracks ------------------------------------------
    ax1 = axes[0]
    ax1.set_title(
        "Repair tracks  "
        "(solid = Phase 1/3: C workers | hatched = Phase 2: B workers | "
        "BUF-xxx = buffer job, dashed outline)",
        fontsize=9,
    )
    ax1.set_xlim(0, makespan * 1.02)
    ax1.set_ylim(-0.5, MAX_REPAIR_TRACKS - 0.5)
    ax1.set_yticks(range(MAX_REPAIR_TRACKS))
    ax1.set_yticklabels([f"Track {i}" for i in range(MAX_REPAIR_TRACKS)], fontsize=8)

    for j in done:
        if j.repair_track is None:
            continue
        y        = j.repair_track
        col      = colors[j.job_id]
        is_gen   = j.job_id.startswith("GEN-")
        edge_col = "darkgreen" if is_gen else ("navy" if j.is_buffered else "white")
        edge_lw  = 1.2 if j.is_buffered else 0.4

        p1_s = j.repair_start
        p1_e = j.phase1_end if j.phase1_end is not None else j.repair_end
        if p1_s is not None and p1_e is not None and p1_e > p1_s:
            ax1.barh(y, p1_e - p1_s, left=p1_s,
                     height=bar_h, color=col,
                     edgecolor=edge_col, linewidth=edge_lw)

        p2_s = p1_e
        p2_e = j.phase2_end if (j.phase2_h > 0 and j.phase2_end is not None) else None
        if p2_s is not None and p2_e is not None and p2_e > p2_s:
            p2_w = p2_e - p2_s
            ax1.barh(y, p2_w, left=p2_s,
                     height=bar_h, color=col, alpha=0.45,
                     edgecolor=edge_col, linewidth=edge_lw, hatch="///")
            if j.phase2_h > 0:
                ratio = p2_w / j.phase2_h
                ax1.text(p2_s + p2_w / 2, y,
                         f"{j.job_id}\n{ratio:.0%}",
                         ha="center", va="center",
                         fontsize=6, color="black", fontweight="bold",
                         clip_on=True)

        p3_s = p2_e if (p2_e is not None and j.phase2_h > 0) else p1_e
        p3_e = j.repair_end
        if p3_s is not None and p3_e is not None and p3_e > p3_s and j.phase3_h > 0:
            ax1.barh(y, p3_e - p3_s, left=p3_s,
                     height=bar_h, color=col, alpha=0.70,
                     edgecolor=edge_col, linewidth=edge_lw)

    p1_patch  = mpatches.Patch(color="grey",             label=f"Phase 1: C workers ({PHASE1_H:.0f} h)")
    p2_patch  = mpatches.Patch(color="grey", alpha=0.45, hatch="///", label="Phase 2: B workers")
    p3_patch  = mpatches.Patch(color="grey", alpha=0.70, label=f"Phase 3: C workers ({PHASE3_H:.0f} h)")
    buf_patch = mpatches.Patch(facecolor="white", edgecolor="navy", linewidth=1.2,
                               label="Buffer job (BUF-xxx)")
    gen_patch = mpatches.Patch(facecolor="white", edgecolor="darkgreen", linewidth=1.2,
                               linestyle="--", label="PERT-generated job (GEN-xxx)")
    ax1.axvline(WEEK_HOURS, color="red", linestyle="--", linewidth=1.2)
    ax1.legend(handles=[p1_patch, p2_patch, p3_patch, buf_patch, gen_patch],
               fontsize=7, loc="upper right")
    ax1.grid(axis="x", alpha=0.3)

    # ---- Panel 2: Oven ---------------------------------------------------
    ax2 = axes[1]
    ax2.set_title(
        f"Oven track  ({int(OVEN_PROCESS_TIME)} h/job  +  {int(CHANGEOVER_TIME)} h changeover)  "
        "â€” navy border = buffer job",
        fontsize=10,
    )
    ax2.set_xlim(0, makespan * 1.02)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks([0])
    ax2.set_yticklabels(["Oven"], fontsize=8)

    for j in done:
        if j.oven_start is None:
            continue
        w        = j.oven_end - j.oven_start
        is_gen   = j.job_id.startswith("GEN-")
        edge_col = "darkgreen" if is_gen else ("navy" if j.is_buffered else "white")
        edge_lw  = 1.5 if j.is_buffered else 0.4
        ax2.barh(0, w, left=j.oven_start, height=bar_h,
                 color=colors[j.job_id], edgecolor=edge_col, linewidth=edge_lw)
        if w > 4:
            ax2.text(j.oven_start + w / 2, 0, j.job_id[:7],
                     ha="center", va="center",
                     fontsize=5.5, color="black", fontweight="bold", rotation=90)

    ax2.axvline(WEEK_HOURS, color="red", linestyle="--", linewidth=1.2)
    ax2.grid(axis="x", alpha=0.3)

    # ---- Panel 3: C worker utilization -----------------------------------
    ax3 = axes[2]
    ax3.set_title("C worker utilization", fontsize=10)
    if staff_log:
        ts, bs, cs = zip(*staff_log)
        ax3.bar(ts, cs, width=WORKER_REASSIGN_INTERVAL * 0.85,
                align="edge", color="#5BA4CF", edgecolor="none")
    ax3.axhline(pool_C, color="red", linestyle="--",
                linewidth=1.2, label=f"C pool limit ({pool_C})")
    ax3.axvline(WEEK_HOURS, color="red", linestyle="--", linewidth=1.2)
    ax3.set_xlim(0, makespan * 1.02)
    ax3.set_ylim(0, pool_C * 1.15)
    ax3.set_ylabel("C workers in use")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(axis="y", alpha=0.3)

    # ---- Panel 4: B worker utilization -----------------------------------
    ax4 = axes[3]
    ax4.set_title("B worker utilization", fontsize=10)
    if staff_log:
        ts, bs, cs = zip(*staff_log)
        ax4.bar(ts, bs, width=WORKER_REASSIGN_INTERVAL * 0.85,
                align="edge", color="#F5A623", edgecolor="none")
    ax4.axhline(pool_B, color="red", linestyle="--",
                linewidth=1.2, label=f"B pool limit ({pool_B})")
    ax4.axvline(WEEK_HOURS, color="red", linestyle="--", linewidth=1.2)
    ax4.set_xlim(0, makespan * 1.02)
    ax4.set_ylim(0, pool_B * 1.15)
    ax4.set_xlabel("Hours")
    ax4.set_ylabel("B workers in use")
    ax4.legend(fontsize=8, loc="upper right")
    ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Gantt chart saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------
# Generated-job duration chart
# ---------------------------------------------------------------------------

def plot_generated_jobs(
    jobs:      list[Job],
    save_path: str = "ga_bc_generated_jobs.png",
) -> None:
    """Bar chart of PERT-generated job durations with the PERT PDF overlaid."""
    gen_jobs = [j for j in jobs if j.job_id.startswith("GEN-")]
    if not gen_jobs:
        print("No generated jobs to plot.")
        return

    durations = [j.estimated_h for j in gen_jobs]
    labels    = [j.job_id for j in gen_jobs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"PERT-generated jobs  (n={len(gen_jobs)}, "
        f"a={_PERT_A:.1f}, mode={_PERT_MODE:.1f}, b={_PERT_B:.1f}, gamma={PERT_GAMMA})",
        fontsize=12, fontweight="bold",
    )

    # ---- Left: individual job durations as a bar chart -------------------
    ax1 = axes[0]
    x = range(len(durations))
    ax1.bar(x, durations, color="#5BA4CF", edgecolor="darkgreen", linewidth=0.8)
    ax1.axhline(sum(durations) / len(durations), color="crimson", linestyle="--",
                linewidth=1.4, label=f"mean = {sum(durations)/len(durations):.1f} h")
    ax1.axhline(_PERT_MODE, color="orange", linestyle=":", linewidth=1.4,
                label=f"PERT mode = {_PERT_MODE:.1f} h")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("Estimated hours")
    ax1.set_title("Duration of each generated job")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # ---- Right: histogram + PERT PDF ------------------------------------
    ax2 = axes[1]
    n_bins = max(5, len(durations) // 3)
    ax2.hist(durations, bins=n_bins, density=True,
             color="#5BA4CF", edgecolor="white", alpha=0.7, label="Generated jobs")

    # PERT PDF via beta distribution
    x_vals = np.linspace(_PERT_A, _PERT_B, 400)
    alpha1  = 1.0 + PERT_GAMMA * (_PERT_MODE - _PERT_A) / (_PERT_B - _PERT_A)
    alpha2  = 1.0 + PERT_GAMMA * (_PERT_B - _PERT_MODE) / (_PERT_B - _PERT_A)
    from scipy.stats import beta as _beta
    z_vals  = (x_vals - _PERT_A) / (_PERT_B - _PERT_A)
    pdf     = _beta.pdf(z_vals, alpha1, alpha2) / (_PERT_B - _PERT_A)
    ax2.plot(x_vals, pdf, color="crimson", linewidth=2, label="PERT PDF")
    ax2.axvline(_PERT_MODE, color="orange", linestyle=":", linewidth=1.4,
                label=f"mode = {_PERT_MODE:.1f} h")
    ax2.set_xlabel("Estimated hours")
    ax2.set_ylabel("Density")
    ax2.set_title("Histogram vs PERT PDF")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Generated-jobs chart saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------

def plot_convergence(history: list[float],
                     save_path: str = "ga_bc_current_convergence.png") -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history, color="#2C7BB6", linewidth=1.8)
    ax.set_title("GA convergence (current_status) â€” best fitness per generation", fontsize=12)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness  (makespan + SC penalties)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Convergence plot saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GA blade repair simulation using current_status.csv (B/C workers)"
    )
    parser.add_argument("--workers_b",   type=int,   default=N_WORKERS_B,
                        help=f"B worker pool (default: {N_WORKERS_B})")
    parser.add_argument("--workers_c",   type=int,   default=N_WORKERS_C,
                        help=f"C worker pool (default: {N_WORKERS_C})")
    parser.add_argument("--generations", type=int,   default=GA_GENERATIONS,
                        help=f"GA generations (default: {GA_GENERATIONS})")
    parser.add_argument("--popsize",     type=int,   default=GA_POP_SIZE,
                        help=f"GA population size (default: {GA_POP_SIZE})")
    parser.add_argument("--csv",         type=str,   default=STATUS_CSV,
                        help="Path to current_status.csv")
    args = parser.parse_args()

    jobs = load_jobs(args.csv)
    n_primary = sum(1 for j in jobs if not j.is_buffered)
    n_buf     = sum(1 for j in jobs if j.is_buffered)

    primary_ests = [j.estimated_h for j in jobs if not j.is_buffered]
    buf_ests     = [j.estimated_h for j in jobs if j.is_buffered]

    print(f"Loaded {len(jobs)} blades from {args.csv}")
    print(f"  Primary jobs (buffer=0) : {n_primary}")
    if primary_ests:
        print(f"    min/avg/max remaining : "
              f"{min(primary_ests):.1f} / {sum(primary_ests)/len(primary_ests):.1f} / "
              f"{max(primary_ests):.1f} h")
    print(f"  Buffer  jobs (buffer=1) : {n_buf}  (GA decides when to pull these in)")
    if buf_ests:
        print(f"    min/avg/max remaining : "
              f"{min(buf_ests):.1f} / {sum(buf_ests)/len(buf_ests):.1f} / "
              f"{max(buf_ests):.1f} h")
    print(f"  B workers: {args.workers_b}  |  C workers: {args.workers_c}")
    print(f"\nRunning GA (pop={args.popsize}, generations={args.generations}) ...")

    best_chrom, best_fit, history = run_ga(
        jobs,
        pool_B=args.workers_b,
        pool_C=args.workers_c,
        pop_size=args.popsize,
        n_gen=args.generations,
    )

    print(f"\nBest fitness : {best_fit:.1f}")

    # Show the GA-chosen dispatch order and flag where buffer jobs ended up
    n = len(jobs)
    order = sorted(jobs, key=lambda j: best_chrom[j.job_index])
    print("\nGA dispatch order:")
    for rank, j in enumerate(order):
        tag = " [BUFFER]" if j.is_buffered else ""
        print(f"  {rank+1:2d}. {j.job_id}  {j.estimated_h:.1f} h{tag}")

    # Final simulation with SA worker refinement
    final_jobs = load_jobs(args.csv)
    print("\nRunning final simulation with SA worker refinement every 4 h ...")
    staff_log, sc1, sc2, sc3 = simulate_ga_bc(
        final_jobs, best_chrom,
        args.workers_b, args.workers_c,
        use_sa=True, use_lookahead=True,
    )

    done       = [j for j in final_jobs if j.oven_end is not None]
    n_gen_done = sum(1 for j in done if j.job_id.startswith("GEN-"))
    print(f"\nCompleted  : {len(done)} jobs total")
    print(f"  Primary  : {sum(1 for j in done if not j.is_buffered)}")
    print(f"  Buffer   : {sum(1 for j in done if j.is_buffered and not j.job_id.startswith('GEN-'))}")
    print(f"  PERT-gen : {n_gen_done}  (max allowed: {MAX_GENERATED_JOBS})")
    if n_gen_done:
        gen_ests = [j.estimated_h for j in final_jobs if j.job_id.startswith("GEN-")]
        print(f"  PERT params : a={_PERT_A:.1f}  mode={_PERT_MODE:.1f}  b={_PERT_B:.1f}  gamma={PERT_GAMMA}")
        print(f"  Generated h : min={min(gen_ests):.1f}  avg={sum(gen_ests)/len(gen_ests):.1f}  max={max(gen_ests):.1f}")
    if done:
        makespan = max(j.oven_end for j in done)
        avg_B    = sum(j.avg_workers_B for j in done) / len(done)
        avg_C    = sum(j.avg_workers_C for j in done) / len(done)
        print(f"\nMakespan          : {makespan:.1f} h")
        print(f"Avg B workers/job : {avg_B:.2f}")
        print(f"Avg C workers/job : {avg_C:.2f}")
        print(f"SC-1 penalty      : {sc1:.1f}")
        print(f"SC-2 penalty      : {sc2:.1f}")
        print(f"SC-3 penalty      : {sc3:.1f}")
        total_fit = makespan + SC1_WEIGHT * sc1 + SC2_WEIGHT * sc2 + SC3_WEIGHT * sc3
        print(f"Total fitness     : {total_fit:.1f}")

        ratios = [(j.repair_end - j.repair_start) / j.estimated_h
                  for j in done if j.repair_end is not None]
        if ratios:
            violations = sum(1 for r in ratios if r < MIN_PROCESS_TIME_RATIO - 1e-6)
            print(f"\nHC-2 check : min ratio = {min(ratios):.4f}  "
                  f"(limit >= {MIN_PROCESS_TIME_RATIO:.2f})  violations = {violations}")

    plot_gantt(final_jobs, staff_log,
               pool_B=args.workers_b, pool_C=args.workers_c,
               save_path="gantt_ga_bc_current.png")
    plot_convergence(history, save_path="ga_bc_current_convergence.png")
    plot_generated_jobs(final_jobs, save_path="ga_bc_generated_jobs.png")


