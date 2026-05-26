"""
ga_simulationBC.py
==================
Blade repair simulation with a Genetic Algorithm that jointly optimises
B-worker and C-worker counts for each job, with workers split into two groups.

Each repair job is divided into three sequential phases:
  Phase 1 (first 9 h)      â€” C workers only
  Phase 2 (middle hours)   â€” B workers only  (= max(0, estimated_h - 36 h))
  Phase 3 (last 27 h)      â€” C workers only

Chromosome encoding
-------------------
Length 3 * N_JOBS integers:
  chrom[i]         â†’ dispatch priority for job i  (integer in [0, 2N-1]; lower = earlier)
  chrom[N + i]     â†’ target B workers for job i   (used during phase 2)
  chrom[2*N + i]   â†’ target C workers for job i   (used during phases 1 and 3)
Priority values determine the queue sort order at simulation start.  HC-9 still
limits track-filling to the first BLADE_QUEUE_LOOKAHEAD jobs in that sorted queue.
Worker values are integers in [0, MAX_WORKERS_PER_BLADE].

All hard constraints (HC-1..HC-12) from constraintsBC.py are enforced.
All soft constraints (SC-1..SC-3) are active in the fitness function:
  SC-1  penalty for B or C allocations outside optimal band [4, 6]
  SC-2  penalty for hours a finished blade waits in the staging queue for oven
  SC-3  penalty for oven idle time between consecutive jobs

Fitness (minimised)
-------------------
    fitness = makespan
            + SC1_WEIGHT * (sc1_B + sc1_C)
            + SC2_WEIGHT * sc2_blocking_hours
            + SC3_WEIGHT * sc3_idle_hours

Outputs a four-panel Gantt chart and a convergence curve.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass, field
from typing import Optional

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
N_JOBS:      int   = 50

PHASE1_H: float = 9.0
PHASE3_H: float = 27.0

DATA_CSV: str = os.path.join(os.path.dirname(__file__), "data_real.csv")

HC2_EFF_CAP: float = 1.0 / MIN_PROCESS_TIME_RATIO  # ~1.1765

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
# Phase helper
# ---------------------------------------------------------------------------

def _compute_phases(estimated_h: float) -> tuple[float, float, float]:
    """Return (phase1_h, phase2_h, phase3_h) that sum to estimated_h."""
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
    job_index:   int = 0

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

    worker_log: list = field(default_factory=list)  # (t0, t1, n_workers, 'B'|'C')

    def __post_init__(self):
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
    """Fast-forward simulation (no logging) returning estimated makespan.

    oven_queue: repair_end times of blades already staging for the oven
    (passed from the main simulation so the forward estimate accounts for them).
    """
    dt    = WORKER_REASSIGN_INTERVAL * 2
    max_t = t + 15_000.0
    last_oven_end = oven_free_at
    pending: list[float] = sorted(oven_queue) if oven_queue else []

    while t < max_t:
        # Transfer staged blades to oven (sorted by repair_end = arrival order)
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

        # Advance work; immediately stage completed slots for the oven
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

    # Drain remaining staged blades
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
    """Return the candidate index that yields the lowest forward makespan."""
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

        p1, p2, p3 = _compute_phases(cand.estimated_h)
        trial_slots[track_i] = _TrialSlot(p1_rem=p1, p2_rem=p2, p3_rem=p3,
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

def random_chromosome(n: int, rng: random.Random) -> list[int]:
    """Length 3*n: [0..n-1]=dispatch priorities [0..2n-1], [n..2n-1]=B targets, [2n..3n-1]=C targets."""
    prios   = [rng.randint(0, 2 * n - 1) for _ in range(n)]
    workers = [rng.randint(0, MAX_WORKERS_PER_BLADE) for _ in range(2 * n)]
    return prios + workers


def two_point_crossover(p1: list[int], p2: list[int],
                        rng: random.Random) -> tuple[list[int], list[int]]:
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))
    return p1[:a] + p2[a:b] + p1[b:], p2[:a] + p1[a:b] + p2[b:]


def mutate(chrom: list[int], rate: float, rng: random.Random) -> list[int]:
    n = len(chrom) // 3   # number of jobs
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
    """Clamp each value to [0, MAX_WORKERS_PER_BLADE], then scale down if total > pool."""
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
    """Apply one epoch of alloc_b/alloc_c then forward-simulate; return makespan."""
    dt = WORKER_REASSIGN_INTERVAL
    trial_slots  = []
    # repair_end times for blades that complete within this single SA epoch
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
                # Blade completes this epoch â€” free the trial slot immediately
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
    """Transfer 1 worker from a random donor to a random receiver within one pool."""
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
    """SA over B and C worker allocations for the current epoch."""
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

def load_jobs(csv_path: str = DATA_CSV) -> list[Job]:
    jobs = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for i, row in enumerate(reader):
            est = float(row["Estimated"])
            jobs.append(Job(job_id=f"JOB-{i+1:03d}", estimated_h=est, job_index=i))
    return jobs


# ---------------------------------------------------------------------------
# Simulation engine (chromosome-driven, B/C workers)
# ---------------------------------------------------------------------------

def simulate_ga_bc(
    jobs:          list[Job],
    chromosome:    list[int],
    pool_B:        int = 0,
    pool_C:        int = 0,
    use_sa:        bool = False,
    use_lookahead: bool = False,
) -> tuple[list[tuple[float, int, int]], float, float, float]:
    """
    Run one simulation driven by chromosome worker targets.
    chromosome[i]       = dispatch priority for job i  (lower value = scheduled first)
    chromosome[N + i]   = B worker target for job i    (used during phase 2)
    chromosome[2*N + i] = C worker target for job i    (used during phases 1 and 3)
    Returns (staff_log, sc1_total, sc2_total, sc3_total).
    staff_log entries: (time, b_workers_used, c_workers_used)
    """
    n = len(jobs)
    if pool_B <= 0: pool_B = N_WORKERS_B
    if pool_C <= 0: pool_C = N_WORKERS_C

    sa_rng    = random.Random(GA_SEED) if use_sa else None
    next_sa_t = 0.0
    dt = WORKER_REASSIGN_INTERVAL

    slots:        list[Optional[_ActiveSlot]] = [None] * MAX_REPAIR_TRACKS
    slot_free_at: list[float]                 = [0.0]  * MAX_REPAIR_TRACKS
    oven_free_at: float = 0.0

    # Sort queue by chromosome dispatch priority (index 0..n-1); HC-9 still limits
    # track-filling to the first BLADE_QUEUE_LOOKAHEAD jobs of this sorted queue.
    queue:       list[Job]                    = sorted(jobs, key=lambda j: chromosome[j.job_index])
    # HC-12: blades done with repair staged here; track freed immediately on completion
    oven_queue:  list[Job]                    = []
    staff_log:   list[tuple[float, int, int]] = []
    sc1_total:   float = 0.0
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

        # Step 2 â€” fill free tracks (HC-12: every ready empty track must get a blade)
        for i in range(MAX_REPAIR_TRACKS):
            if slots[i] is None and slot_free_at[i] <= t and queue:
                # HC-9: only consider first BLADE_QUEUE_LOOKAHEAD jobs.
                # Queue is pre-sorted by chromosome priority so position 0 is highest priority.
                job = queue.pop(0)
                p1, p2, p3 = _compute_phases(job.estimated_h)
                aj  = _ActiveSlot(job=job, track_idx=i, p1_rem=p1, p2_rem=p2, p3_rem=p3)
                job.repair_track = i
                job.repair_start = t
                slots[i] = aj
        enforce_empty_track_fill(slots, slot_free_at, queue, t)

        # Step 3 â€” allocate workers from chromosome (SA-refined if enabled)
        active   = [aj for aj in slots if aj is not None]
        b_active = [aj for aj in active if aj.phase == 2]
        c_active = [aj for aj in active if aj.phase in (1, 3)]

        if active:
            raw_b = {aj.job.job_index: chromosome[n + aj.job.job_index]       for aj in b_active}
            raw_c = {aj.job.job_index: chromosome[2 * n + aj.job.job_index]   for aj in c_active}

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

        # Step 4 â€” advance work by dt; enforce HC-2 (85% floor)
        # Blades that finish are immediately staged for the oven and their track freed (HC-12)
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

            else:  # phase 3
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

        # Free completed tracks immediately â€” blade moves to staging queue
        for aj in newly_done:
            oven_queue.append(aj.job)
            slot_free_at[aj.track_idx] = aj.job.repair_end + CHANGEOVER_TIME
            slots[aj.track_idx]        = None

        # Step 5 â€” terminate when repair queue, staging queue, and all tracks are clear
        if not queue and not oven_queue and all(s is None for s in slots):
            break

        t += dt

    # Safety drain: blades staged but not yet transferred to oven
    for job in sorted(oven_queue, key=lambda j: j.repair_end):
        ov_start       = max(oven_free_at, job.repair_end)
        job.oven_start = ov_start
        job.oven_end   = ov_start + OVEN_PROCESS_TIME
        oven_free_at   = job.oven_end + CHANGEOVER_TIME
        oven_starts.append(ov_start)
        oven_ends.append(job.oven_end)

    # SC-2: total hours blades blocked their track waiting for oven
    done = [j for j in jobs if j.oven_start is not None and j.repair_end is not None]
    sc2_total = sum(max(0.0, j.oven_start - j.repair_end) for j in done)

    # SC-3: oven idle time between consecutive jobs
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
    return [Job(job_id=j.job_id, estimated_h=j.estimated_h, job_index=j.job_index)
            for j in template]


def evaluate(chromosome: list[int],
             jobs_template: list[Job],
             pool_B: int,
             pool_C: int) -> float:
    jobs_copy = _fresh_jobs(jobs_template)
    _, sc1, sc2, sc3 = simulate_ga_bc(jobs_copy, chromosome, pool_B, pool_C)
    done = [j for j in jobs_copy if j.oven_end is not None]
    makespan = max((j.oven_end for j in done), default=float('inf'))
    return makespan + SC1_WEIGHT * sc1 + SC2_WEIGHT * sc2 + SC3_WEIGHT * sc3


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
    """Run the GA. Returns (best_chromosome, best_fitness, history)."""
    if pool_B   <= 0: pool_B   = N_WORKERS_B
    if pool_C   <= 0: pool_C   = N_WORKERS_C
    if pop_size <= 0: pop_size = GA_POP_SIZE
    if n_gen    <= 0: n_gen    = GA_GENERATIONS

    rng = random.Random(seed)
    n   = len(jobs_template)

    pop  = [random_chromosome(n, rng) for _ in range(pop_size)]
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
            next_pop.append(mutate(c1, mut_rate, rng))
            if len(next_pop) < pop_size:
                next_pop.append(mutate(c2, mut_rate, rng))

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
    save_path: str = "gantt_ga_bc.png",
) -> None:

    done = [j for j in jobs if j.oven_end is not None]
    if not done:
        print("No finished jobs â€” nothing to plot.")
        return

    makespan = max(j.oven_end for j in done)
    palette  = _make_palette(len(jobs))
    colors   = {j.job_id: palette[k] for k, j in enumerate(jobs)}
    bar_h    = 0.72

    fig, axes = plt.subplots(
        4, 1,
        figsize=(16, 14),
        gridspec_kw={"height_ratios": [MAX_REPAIR_TRACKS * 0.6, 1.8, 2.0, 2.0]},
    )
    fig.suptitle(
        f"GA (B/C workers)  â€”  {len(done)} jobs  (all HC + SC active)",
        fontsize=13, fontweight="bold",
    )

    # ---- Panel 1: Repair tracks (three-phase segments) -------------------
    ax1 = axes[0]
    ax1.set_title(
        "Repair tracks  "
        "(solid = Phase 1/3: C workers | hatched = Phase 2: B workers)",
        fontsize=9,
    )
    ax1.set_xlim(0, makespan * 1.02)
    ax1.set_ylim(-0.5, MAX_REPAIR_TRACKS - 0.5)
    ax1.set_yticks(range(MAX_REPAIR_TRACKS))
    ax1.set_yticklabels([f"Track {i}" for i in range(MAX_REPAIR_TRACKS)], fontsize=8)

    for j in done:
        if j.repair_track is None:
            continue
        y   = j.repair_track
        col = colors[j.job_id]

        p1_s = j.repair_start
        p1_e = j.phase1_end if j.phase1_end is not None else j.repair_end
        if p1_s is not None and p1_e is not None and p1_e > p1_s:
            ax1.barh(y, p1_e - p1_s, left=p1_s,
                     height=bar_h, color=col, edgecolor="white", linewidth=0.4)

        p2_s = p1_e
        p2_e = j.phase2_end if (j.phase2_h > 0 and j.phase2_end is not None) else None
        if p2_s is not None and p2_e is not None and p2_e > p2_s:
            p2_w = p2_e - p2_s
            ax1.barh(y, p2_w, left=p2_s,
                     height=bar_h, color=col, alpha=0.45,
                     edgecolor="white", linewidth=0.4, hatch="///")
            if j.phase2_h > 0:
                ratio = p2_w / j.phase2_h
                ax1.text(p2_s + p2_w / 2, y,
                         f"{j.job_id}\n{ratio:.0%} of est.",
                         ha="center", va="center",
                         fontsize=6, color="black", fontweight="bold",
                         clip_on=True)

        p3_s = p2_e if (p2_e is not None and j.phase2_h > 0) else p1_e
        p3_e = j.repair_end
        if p3_s is not None and p3_e is not None and p3_e > p3_s and j.phase3_h > 0:
            ax1.barh(y, p3_e - p3_s, left=p3_s,
                     height=bar_h, color=col, alpha=0.70,
                     edgecolor="white", linewidth=0.4)

    p1_patch = mpatches.Patch(color="grey",             label=f"Phase 1: C workers ({PHASE1_H:.0f} h)")
    p2_patch = mpatches.Patch(color="grey", alpha=0.45,
                               hatch="///",              label="Phase 2: B workers (middle)")
    p3_patch = mpatches.Patch(color="grey", alpha=0.70, label=f"Phase 3: C workers ({PHASE3_H:.0f} h)")
    ax1.axvline(WEEK_HOURS, color="red", linestyle="--", linewidth=1.2)
    ax1.legend(handles=[p1_patch, p2_patch, p3_patch], fontsize=7, loc="upper right")
    ax1.grid(axis="x", alpha=0.3)

    # ---- Panel 2: Oven ---------------------------------------------------
    ax2 = axes[1]
    ax2.set_title(
        f"Oven track  ({int(OVEN_PROCESS_TIME)} h/job  +  {int(CHANGEOVER_TIME)} h changeover)",
        fontsize=10,
    )
    ax2.set_xlim(0, makespan * 1.02)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks([0])
    ax2.set_yticklabels(["Oven"], fontsize=8)

    for j in done:
        if j.oven_start is None:
            continue
        w = j.oven_end - j.oven_start
        ax2.barh(0, w, left=j.oven_start, height=bar_h,
                 color=colors[j.job_id], edgecolor="white", linewidth=0.4)
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
        ax3.bar(ts, cs,
                width=WORKER_REASSIGN_INTERVAL * 0.85,
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
        ax4.bar(ts, bs,
                width=WORKER_REASSIGN_INTERVAL * 0.85,
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

def plot_convergence(history: list[float],
                     save_path: str = "ga_bc_convergence.png") -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history, color="#2C7BB6", linewidth=1.8)
    ax.set_title("GA (B/C workers) convergence â€” best fitness per generation", fontsize=12)
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
    parser = argparse.ArgumentParser(description="GA blade repair simulation (B/C workers)")
    parser.add_argument("--workers_b",   type=int,   default=N_WORKERS_B,
                        help=f"B worker pool (default: {N_WORKERS_B})")
    parser.add_argument("--workers_c",   type=int,   default=N_WORKERS_C,
                        help=f"C worker pool (default: {N_WORKERS_C})")
    parser.add_argument("--jobs",        type=int,   default=N_JOBS,
                        help=f"Jobs to load from data_real.csv (default: {N_JOBS})")
    parser.add_argument("--generations", type=int,   default=GA_GENERATIONS,
                        help=f"GA generations (default: {GA_GENERATIONS})")
    parser.add_argument("--popsize",     type=int,   default=GA_POP_SIZE,
                        help=f"GA population size (default: {GA_POP_SIZE})")
    args = parser.parse_args()

    jobs = load_jobs()
    if args.jobs is not None:
        jobs = jobs[:args.jobs]
    n = len(jobs)

    ests = [j.estimated_h for j in jobs]
    print(f"Loaded {n} jobs  |  B workers: {args.workers_b}  |  C workers: {args.workers_c}")
    print(f"  min estimated = {min(ests):.1f} h")
    print(f"  max estimated = {max(ests):.1f} h")
    print(f"  avg estimated = {sum(ests)/len(ests):.1f} h")
    print(f"\nRunning GA (pop={args.popsize}, generations={args.generations}) ...")

    best_chrom, best_fit, history = run_ga(
        jobs,
        pool_B=args.workers_b,
        pool_C=args.workers_c,
        pop_size=args.popsize,
        n_gen=args.generations,
    )

    print(f"\nBest fitness : {best_fit:.1f}")

    final_jobs = load_jobs()
    if args.jobs is not None:
        final_jobs = final_jobs[:args.jobs]
    print("\nRunning final simulation with SA worker refinement every 4 h ...")
    staff_log, sc1, sc2, sc3 = simulate_ga_bc(
        final_jobs, best_chrom,
        args.workers_b, args.workers_c,
        use_sa=True, use_lookahead=True,
    )

    done = [j for j in final_jobs if j.oven_end is not None]
    print(f"\nCompleted  : {len(done)}/{n} jobs")
    if done:
        makespan = max(j.oven_end for j in done)
        avg_B    = sum(j.avg_workers_B for j in done) / len(done)
        avg_C    = sum(j.avg_workers_C for j in done) / len(done)
        print(f"Makespan          : {makespan:.1f} h")
        print(f"Avg B workers/job : {avg_B:.2f}")
        print(f"Avg C workers/job : {avg_C:.2f}")
        print(f"SC-1 penalty      : {sc1:.1f}")
        print(f"SC-2 penalty      : {sc2:.1f}")
        print(f"SC-3 penalty      : {sc3:.1f}")
        total_fit = makespan + SC1_WEIGHT * sc1 + SC2_WEIGHT * sc2 + SC3_WEIGHT * sc3
        print(f"Total fitness     : {total_fit:.1f}")

        ratios = [(j.repair_end - j.repair_start) / j.estimated_h
                  for j in done if j.repair_end is not None]
        violations = sum(1 for r in ratios if r < MIN_PROCESS_TIME_RATIO - 1e-6)
        print(f"\nHC-2 check : min ratio = {min(ratios):.4f}  "
              f"(limit >= {MIN_PROCESS_TIME_RATIO:.2f})  violations = {violations}")

    plot_gantt(final_jobs, staff_log,
               pool_B=args.workers_b, pool_C=args.workers_c,
               save_path=f"gantt_ga_bc_{n}jobs.png")
    plot_convergence(history, save_path=f"ga_bc_convergence_{n}jobs.png")


