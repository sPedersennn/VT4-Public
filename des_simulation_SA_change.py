"""
des_simulationBC.py
===================
Discrete-Event Simulation (DES) of the blade repair scheduling from ga_simulationBC_current.py.

Instead of advancing in fixed 4-h epochs, an event heap (heapq) drives the
simulation. Events are processed in chronological order; state changes only
when something actually happens.

Event types (lower priority value = processed first at equal time):
  PHASE_DONE  (0)  -  exact moment a repair phase ends on a track
  EPOCH       (1)  -  every WORKER_REASSIGN_INTERVAL h; reallocate workers
  OVEN_FREE   (2)  -  oven + changeover done; accept next waiting blade
  TRACK_FREE  (3)  -  track + changeover done; assign next job from queue

Key differences from the time-stepped ga_simulationBC_current.simulate_ga_bc():
  - Oven transfers and track fills happen at the exact available instant,
    not at the next 4-h epoch boundary.
  - Phase-completion times are computed exactly and recorded precisely.
  - Stale PHASE_DONE events (version-mismatched) are silently discarded,
    so rescheduling after a worker reallocation is cheap and correct.
  - Tracks are freed immediately at repair completion (HC-12), matching
    ga_simulationBC_current.simulate_ga_bc() semantics.

The GA optimiser, chromosome encoding, constraints, and Gantt/convergence
plots are all reused unchanged from ga_simulationBC_current.py / constraintsBC.py.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import math
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as _scipy_beta

from ga_simulationBC_current import (
    Job,
    _ActiveSlot,
    _cap_allocation,
    _sa_allocate_bc,
    _sa_fitness_bc,
    _sa_neighbor_pool,
    _fresh_jobs,
    load_jobs,
    plot_convergence,
    run_ga,
    evaluate,
    pert_sample,
    _PERT_A,
    _PERT_MODE,
    _PERT_B,
    N_WORKERS_B,
    N_WORKERS_C,
    MAX_GENERATED_JOBS,
    SC1_WEIGHT,
    SC2_WEIGHT,
    SC3_WEIGHT,
    GA_SEED,
    HC2_EFF_CAP,
    SA_ITERATIONS,
    SA_TEMP_INIT,
    SA_COOLING,
    GA_POP_SIZE,
    GA_GENERATIONS,
    WEEK_HOURS,
    PHASE1_H,
    PHASE3_H,
)
from constraintsBC import (
    BLADE_QUEUE_LOOKAHEAD,
    CHANGEOVER_TIME,
    MAX_REPAIR_TRACKS,
    OVEN_PROCESS_TIME,
    WORKER_REASSIGN_INTERVAL,
    MIN_PROCESS_TIME_RATIO,
    effective_repair_hours,
    efficiency_factor,
    penalty_worker_band,
    penalty_oven_idle,
    enforce_full_worker_utilization_B,
    enforce_full_worker_utilization_C,
)

# SA parameter overrides for this file
SA_TEMP_INIT  = 1000
SA_ITERATIONS = 5000
SA_COOLING    = 0.99

# ---------------------------------------------------------------------------
# Event priorities (lower = fires first at equal timestamp)
# ---------------------------------------------------------------------------
_P_PHASE_DONE = 0
_P_EPOCH      = 1
_P_OVEN_FREE  = 2
_P_TRACK_FREE = 3

_MAX_T     = 15_000.0
DELAY_SEED = 7   # RNG seed for Monte Carlo delay sampling (separate from GA seed)


# ---------------------------------------------------------------------------
# Monte Carlo delay (Modified PERT distribution from delay_distrubution.py)
# ---------------------------------------------------------------------------

_PERT_CACHE: Optional[tuple[float, float, float, float]] = None


def _load_pert_params() -> tuple[float, float, float, float]:
    """Compute (a, mode, b, gamma=4) from Actual-Estimated delays in data_real.csv."""
    csv_path = os.path.join(os.path.dirname(__file__), "data_real.csv")
    delays: list[float] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            try:
                delays.append(float(row["Actual"]) - float(row["Estimated"]))
            except (KeyError, ValueError):
                pass
    a     = float(min(delays))
    b     = float(max(delays))
    counts, edges = np.histogram(delays, bins="auto")
    mode  = float((edges[np.argmax(counts)] + edges[np.argmax(counts) + 1]) / 2)
    return a, mode, b, 60.0


def _get_pert_params() -> tuple[float, float, float, float]:
    global _PERT_CACHE
    if _PERT_CACHE is None:
        _PERT_CACHE = _load_pert_params()
    return _PERT_CACHE


def _sample_pert_delay(rng: random.Random, job=None) -> float:
    """
    Draw one delay sample (hours) via inverse-CDF Monte Carlo on the Modified PERT.

    PERT is parameterised by (a=min, mode, b=max, gamma=4) fitted to the
    historical (Actual - Estimated) data.  A positive sample means the job
    takes longer than estimated; negative means it finishes early.

    If job has a _preset_delay attribute (set externally for paired MC
    experiments), that value is returned directly without consuming the RNG.
    """
    preset = getattr(job, "_preset_delay", None)
    if preset is not None:
        return preset
    a, mode, b, gamma = _get_pert_params()
    alpha1 = 1.0 + gamma * (mode - a) / (b - a)
    alpha2 = 1.0 + gamma * (b - mode) / (b - a)
    u = rng.random()                              # uniform on (0, 1)
    z = float(_scipy_beta.ppf(u, alpha1, alpha2)) # inverse Beta CDF
    return a + z * (b - a)


# ---------------------------------------------------------------------------
# SA allocation with per-step convergence history
# ---------------------------------------------------------------------------

def _sa_allocate_bc_tracked(
    active_b:     list,
    active_c:     list,
    slots:        list,
    slot_free_at: list[float],
    oven_free_at: float,
    queue:        list,
    t:            float,
    pool_B:       int,
    pool_C:       int,
    seed_alloc_b: dict,
    seed_alloc_c: dict,
    rng:          random.Random,
    oven_queue:   Optional[list[float]] = None,
) -> tuple[dict, dict, list[float]]:
    """Identical to _sa_allocate_bc but also returns best_fit at every step."""
    keys_b = [s.job.job_index for s in active_b]
    keys_c = [s.job.job_index for s in active_c]

    current_b = {k: seed_alloc_b.get(k, 0) for k in keys_b}
    current_c = {k: seed_alloc_c.get(k, 0) for k in keys_c}
    current_b = _cap_allocation(current_b, pool_B)
    current_c = _cap_allocation(current_c, pool_C)
    if keys_b:
        from constraintsBC import enforce_full_worker_utilization_B
        current_b = enforce_full_worker_utilization_B(current_b, pool_B)
    if keys_c:
        from constraintsBC import enforce_full_worker_utilization_C
        current_c = enforce_full_worker_utilization_C(current_c, pool_C)

    current_fit = _sa_fitness_bc(current_b, current_c, slots, slot_free_at,
                                  oven_free_at, queue, t, pool_B, pool_C, oven_queue)
    best_b, best_c = dict(current_b), dict(current_c)
    best_fit = current_fit
    temp = SA_TEMP_INIT
    history: list[float] = [best_fit]

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
        history.append(best_fit)

    return best_b, best_c, history


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------
@dataclass
class _Ev:
    time: float
    prio: int
    data: dict = field(default_factory=dict, compare=False)

    def __lt__(self, other: _Ev) -> bool:
        return (self.time, self.prio) < (other.time, other.prio)


# ---------------------------------------------------------------------------
# Mutable simulation state
# ---------------------------------------------------------------------------
@dataclass
class _State:
    slots:         list                         # Optional[_ActiveSlot] * MAX_REPAIR_TRACKS
    slot_free_at:  list[float]                  # earliest time each track is usable
    oven_free_at:  float
    queue:         list[Job]                    # jobs not yet started
    pending_oven:  list                         # (repair_end, seq, job) sorted by repair_end
    oven_seq:      int                          # tie-break counter for pending_oven
    track_version: list[int]                    # PHASE_DONE invalidation token per track
    alloc_since:   list[float]                  # time of last worker assignment per track
    heap:          list                         # heapq of _Ev
    staff_log:     list[tuple[float, int, int]]
    sc1_total:     float
    oven_starts:   list[float]
    oven_ends:     list[float]
    chromosome:    list[int]
    n_chrom:       int                          # len(chromosome) // 3  (number of jobs)
    pool_B:        int
    pool_C:        int
    use_sa:        bool
    use_lookahead: bool
    sa_rng:        Optional[random.Random]
    sa_trigger:    bool   # True when a job finished or a delay was applied since last SA run
    use_delays:      bool
    delay_rng:       Optional[random.Random]
    delay_applied:   set   # job_index values that have already been delayed
    jobs_list:       list  # master jobs list; generated jobs are appended here
    generated_count: int
    gen_rng:         Optional[random.Random]
    default_b:        int   # fallback B workers for generated jobs (no chromosome gene)
    default_c:        int   # fallback C workers for generated jobs
    primary_remaining: int  # primary jobs not yet dispatched; buffer/generated jobs are
                            # blocked from repair tracks until this reaches 0
    forced_workers:   dict  # job_index -> fixed worker count (overrides GA/SA allocation)
    sa_history:       Optional[list]  # None = not collecting; list = [(epoch_t, [best_fit...])]


def _push(state: _State, ev: _Ev) -> None:
    heapq.heappush(state.heap, ev)


# ---------------------------------------------------------------------------
# Work-advancement helper
# ---------------------------------------------------------------------------

def _advance_work(slot: _ActiveSlot, from_t: float, to_t: float) -> None:
    """Reduce the current-phase remainder by elapsed wall-clock time."""
    dt = to_t - from_t
    if dt <= 1e-12:
        return
    phase = slot.phase
    w = slot.workers_C if phase in (1, 3) else slot.workers_B
    eff = min(efficiency_factor(w), HC2_EFF_CAP)
    if eff <= 0:
        return
    work = dt * eff
    if phase == 1:
        slot.p1_rem = max(0.0, slot.p1_rem - work)
    elif phase == 2:
        slot.p2_rem = max(0.0, slot.p2_rem - work)
    else:
        slot.p3_rem = max(0.0, slot.p3_rem - work)


# ---------------------------------------------------------------------------
# PHASE_DONE scheduler
# ---------------------------------------------------------------------------

def _schedule_phase_done(state: _State, i: int, t: float) -> None:
    """Increment version and push a PHASE_DONE event for track i's current phase."""
    slot = state.slots[i]
    if slot is None or slot.remaining_h <= 1e-9:
        return
    state.track_version[i] += 1
    phase = slot.phase
    w = slot.workers_C if phase in (1, 3) else slot.workers_B
    eff = min(efficiency_factor(w), HC2_EFF_CAP)
    if eff <= 0:
        return  # workers=0 â†’ job paused; no PHASE_DONE scheduled
    rem = slot.p1_rem if phase == 1 else (slot.p2_rem if phase == 2 else slot.p3_rem)
    if rem <= 1e-9:
        return
    _push(state, _Ev(
        t + rem / eff, _P_PHASE_DONE,
        {'track': i, 'ver': state.track_version[i], 'phase': phase},
    ))


# ---------------------------------------------------------------------------
# Oven intake
# ---------------------------------------------------------------------------

def _try_oven_intake(state: _State, t: float) -> None:
    """
    Transfer pending blades to the oven in arrival order.
    Tracks are already freed at repair completion; this only schedules oven runs.
    """
    state.pending_oven.sort()
    i = 0
    while i < len(state.pending_oven):
        rep_end, _seq, job = state.pending_oven[i]
        ov_start = max(state.oven_free_at, rep_end)
        if ov_start > t:
            break
        job.oven_start     = ov_start
        job.oven_end       = ov_start + OVEN_PROCESS_TIME
        state.oven_free_at = job.oven_end + CHANGEOVER_TIME
        state.oven_starts.append(ov_start)
        state.oven_ends.append(job.oven_end)
        _push(state, _Ev(state.oven_free_at, _P_OVEN_FREE))
        state.pending_oven.pop(i)
        # do NOT increment i  -  list shortened


# ---------------------------------------------------------------------------
# SA-based buffer job selection
# ---------------------------------------------------------------------------

def _sa_select_buffer_job(state: _State, t: float) -> None:
    """
    Reorder the buffer portion of the queue so the SA-chosen job is first.
    Considers the first BLADE_QUEUE_LOOKAHEAD buffer jobs.

    Fitness: estimated track-blocking time  -  how long the track will sit idle
    waiting for the oven after this job finishes repair.  A job whose estimated
    repair-end aligns with oven availability scores best (blocking â†’ 0).
    """
    buf_positions: list[tuple[int, Job]] = []
    for i, job in enumerate(state.queue):
        if job.is_buffered:
            buf_positions.append((i, job))
        if len(buf_positions) >= BLADE_QUEUE_LOOKAHEAD:
            break

    if len(buf_positions) <= 1:
        return

    n_active = sum(1 for s in state.slots if s is not None and s.remaining_h > 1e-9)
    avg_w    = max(1, (state.pool_B + state.pool_C) // max(1, n_active + 1))

    def _score(job: Job) -> float:
        repair_h = effective_repair_hours(job.estimated_h, avg_w)
        # Penalise jobs that finish repair well before the oven is free
        # (positive gap = track blocked waiting for oven)
        return max(0.0, state.oven_free_at - (t + repair_h))

    indices  = [i for i, _ in buf_positions]
    scores   = [_score(j) for _, j in buf_positions]

    rng         = state.sa_rng or random.Random(GA_SEED)
    current_k   = 0
    current_fit = scores[0]
    best_k      = 0
    best_fit    = current_fit
    temp        = SA_TEMP_INIT * 0.1

    for _ in range(SA_ITERATIONS // 5):
        k     = rng.randrange(len(buf_positions))
        fit   = scores[k]
        delta = fit - current_fit
        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
            current_k, current_fit = k, fit
            if current_fit < best_fit:
                best_k, best_fit = k, fit
        temp *= SA_COOLING

    if best_k != 0:
        state.queue.insert(indices[0], state.queue.pop(indices[best_k]))


# ---------------------------------------------------------------------------
# Track filling
# ---------------------------------------------------------------------------

def _try_fill_tracks(state: _State, t: float) -> None:
    """Assign jobs from the queue to every free repair track."""
    for i in range(MAX_REPAIR_TRACKS):
        if state.slots[i] is not None or state.slot_free_at[i] > t or not state.queue:
            continue
        # Hard constraint: all 12 primary jobs (first in CSV) must be dispatched
        # to repair tracks before any buffer or PERT-generated job may enter.
        if state.primary_remaining > 0 and state.queue[0].is_buffered:
            break
        # HC-9: SA selects the best of the first BLADE_QUEUE_LOOKAHEAD buffer jobs;
        # for primary jobs the chromosome order already ranks position 0 as best.
        if state.use_sa and state.queue[0].is_buffered:
            _sa_select_buffer_job(state, t)
        job = state.queue.pop(0)
        if not job.is_buffered:
            state.primary_remaining -= 1

        # When a buffer job is dispatched, replenish the buffer with a PERT-sampled
        # arrival (up to MAX_GENERATED_JOBS total), mirroring ga_simulationBC_current.
        if job.is_buffered and state.generated_count < MAX_GENERATED_JOBS:
            new_h   = pert_sample(state.gen_rng, _PERT_A, _PERT_MODE, _PERT_B)
            new_job = Job(
                job_id      = f"GEN-{state.generated_count + 1:03d}",
                estimated_h = new_h,
                job_index   = state.n_chrom + state.generated_count,
                is_buffered = True,
            )
            state.generated_count += 1
            state.queue.append(new_job)
            state.jobs_list.append(new_job)

        aj = _ActiveSlot(job=job, track_idx=i,
                         p1_rem=job.phase1_h, p2_rem=job.phase2_h, p3_rem=job.phase3_h)
        job.repair_track     = i
        job.repair_start     = t
        if job.skip_phase1:
            job.phase1_end = t  # zero-length phase 1 so Gantt renders correctly
        state.slots[i]       = aj
        state.alloc_since[i] = t


# ---------------------------------------------------------------------------
# EPOCH handler
# ---------------------------------------------------------------------------

def _on_epoch(state: _State, t: float) -> bool:
    """
    Worker reallocation checkpoint every WORKER_REASSIGN_INTERVAL hours.
    Returns False when the simulation is finished.
    """
    dt = WORKER_REASSIGN_INTERVAL

    # 1. Transfer any finished blades to oven
    _try_oven_intake(state, t)

    # 2. Fill empty tracks from queue
    _try_fill_tracks(state, t)

    # 3. Advance remaining work (from last alloc time to now) for every active slot
    for i, slot in enumerate(state.slots):
        if slot is not None and slot.remaining_h > 1e-9:
            _advance_work(slot, state.alloc_since[i], t)
            state.alloc_since[i] = t

    # 4. Allocate workers from chromosome (optionally SA-refined)
    active = [s for s in state.slots if s is not None and s.remaining_h > 1e-9]
    for s in active:
        s.workers_B = 0
        s.workers_C = 0

    b_jobs: list = []
    c_jobs: list = []

    if active:
        b_jobs = [s for s in active if s.phase == 2]
        c_jobs = [s for s in active if s.phase in (1, 3)]
        n = state.n_chrom

        # chromosome[n+i]  = B-worker target for job i  (phase 2)
        # chromosome[2n+i] = C-worker target for job i  (phases 1 and 3)
        # Generated jobs (job_index >= n) have no chromosome gene; fall back to defaults.
        raw_b = {
            s.job.job_index: (state.chromosome[n + s.job.job_index]
                              if n + s.job.job_index < len(state.chromosome)
                              else state.default_b)
            for s in b_jobs
        }
        raw_c = {
            s.job.job_index: (state.chromosome[2 * n + s.job.job_index]
                              if 2 * n + s.job.job_index < len(state.chromosome)
                              else state.default_c)
            for s in c_jobs
        }

        if state.use_sa and state.sa_trigger:
            state.sa_trigger = False
            oq_times = [rep_end for rep_end, _, _ in state.pending_oven]
            if state.sa_history is not None:
                capped_b, capped_c, step_hist = _sa_allocate_bc_tracked(
                    b_jobs, c_jobs, state.slots, state.slot_free_at, state.oven_free_at,
                    state.queue, t, state.pool_B, state.pool_C,
                    raw_b, raw_c, state.sa_rng, oq_times,
                )
                state.sa_history.append((t, step_hist))
            else:
                capped_b, capped_c = _sa_allocate_bc(
                    b_jobs, c_jobs, state.slots, state.slot_free_at, state.oven_free_at,
                    state.queue, t, state.pool_B, state.pool_C,
                    raw_b, raw_c, state.sa_rng, oq_times,
                )
        else:
            capped_b = _cap_allocation(raw_b, state.pool_B)
            capped_c = _cap_allocation(raw_c, state.pool_C)
            if b_jobs:
                capped_b = enforce_full_worker_utilization_B(capped_b, state.pool_B)
            if c_jobs:
                capped_c = enforce_full_worker_utilization_C(capped_c, state.pool_C)

        # Forced-worker override: pin allocation for specific jobs regardless of
        # GA/SA output.  Capped at the pool size to stay physically valid.
        for s in b_jobs:
            if s.job.job_index in state.forced_workers:
                capped_b[s.job.job_index] = min(state.forced_workers[s.job.job_index],
                                                 state.pool_B)
        for s in c_jobs:
            if s.job.job_index in state.forced_workers:
                capped_c[s.job.job_index] = min(state.forced_workers[s.job.job_index],
                                                 state.pool_C)

        for s in b_jobs:
            w = capped_b.get(s.job.job_index, 0)
            s.workers_B = w
            state.sc1_total += penalty_worker_band(w) * dt
            s.job.worker_log.append((t, t + dt, w, 'B'))

        for s in c_jobs:
            w = capped_c.get(s.job.job_index, 0)
            s.workers_C = w
            s.job.worker_log.append((t, t + dt, w, 'C'))

    # 4b. Final-epoch Monte Carlo delay: sample a PERT delay in the last epoch
    #     before phase 2 completes.  Detected when allocated B-workers will
    #     exhaust p2_rem within this epoch (p2_rem / eff <= dt).
    if state.use_delays:
        for slot in state.slots:
            if slot is None or slot.p2_rem <= 1e-9:
                continue
            if slot.job.job_index in state.delay_applied:
                continue
            w   = slot.workers_B
            eff = min(efficiency_factor(w), HC2_EFF_CAP)
            if eff <= 0:
                continue
            if slot.p2_rem / eff > dt:
                continue  # phase 2 will not finish this epoch; wait
            state.delay_applied.add(slot.job.job_index)
            state.sa_trigger = True
            d      = _sample_pert_delay(state.delay_rng, slot.job)
            old_p2 = slot.p2_rem
            # HC-2 floor: total phase-2 duration must stay >= 0.85*estimated - 9 - 27.
            # Account for phase-2 work already completed so the floor shrinks as
            # the job progresses (at the final epoch p2_done >> 0, floor â†’ 0).
            p2_initial   = slot.job.phase2_h
            p2_done      = max(0.0, p2_initial - old_p2)
            min_p2_total = max(0.0,
                               MIN_PROCESS_TIME_RATIO * slot.job.estimated_h
                               - slot.job.phase1_h - slot.job.phase3_h)
            floor_p2 = max(0.0, min_p2_total - p2_done)
            slot.p2_rem       = max(old_p2 + d, floor_p2)
            slot.job.mc_delay = getattr(slot.job, "mc_delay", 0.0) + (slot.p2_rem - old_p2)
            # Negative delay can zero out p2_rem before _schedule_phase_done
            # runs, so _on_phase_done never fires and phase2_end stays None.
            # Record it here so the Gantt bar and annotation are not lost.
            if slot.p2_rem <= 1e-9:
                slot.p2_rem = 0.0
                slot.job.phase2_end = t

    # 5. Schedule PHASE_DONE for each active track with current allocation
    for i in range(MAX_REPAIR_TRACKS):
        if state.slots[i] is not None and state.slots[i].remaining_h > 1e-9:
            _schedule_phase_done(state, i, t)

    # 6. Staff log entry
    total_B = sum(s.workers_B for s in state.slots if s is not None)
    total_C = sum(s.workers_C for s in state.slots if s is not None)
    state.staff_log.append((t, total_B, total_C))

    # 7. Termination check
    if (not state.queue
            and not state.pending_oven
            and all(s is None for s in state.slots)):
        return False

    # 8. Next epoch
    if t + dt < _MAX_T:
        _push(state, _Ev(t + dt, _P_EPOCH))
    return True


# ---------------------------------------------------------------------------
# PHASE_DONE handler
# ---------------------------------------------------------------------------

def _on_phase_done(state: _State, t: float, track: int, ver: int, phase: int) -> None:
    """
    Exact phase completion.  Stale events (version mismatch) are discarded.
    When all phases complete the track is freed immediately (HC-12).
    """
    if ver != state.track_version[track]:
        return

    slot = state.slots[track]
    if slot is None or slot.remaining_h <= 1e-9:
        return

    job = slot.job

    # Advance work exactly up to this phase-end time
    _advance_work(slot, state.alloc_since[track], t)
    state.alloc_since[track] = t

    if phase == 1:
        slot.p1_rem    = 0.0
        job.phase1_end = t
        if slot.p2_rem <= 1e-9 and slot.p3_rem <= 1e-9:
            min_end        = job.repair_start + MIN_PROCESS_TIME_RATIO * job.estimated_h
            job.repair_end = max(t, min_end)
    elif phase == 2:
        slot.p2_rem    = 0.0
        job.phase2_end = t
        if slot.p3_rem <= 1e-9:
            min_end        = job.repair_start + MIN_PROCESS_TIME_RATIO * job.estimated_h
            job.repair_end = max(t, min_end)
    else:  # phase 3
        slot.p3_rem    = 0.0
        min_end        = job.repair_start + MIN_PROCESS_TIME_RATIO * job.estimated_h
        job.repair_end = max(t, min_end)

    if slot.remaining_h <= 1e-9 and job.repair_end is not None:
        # HC-12: free the track immediately; changeover starts at repair_end
        state.slots[track]        = None
        state.slot_free_at[track] = job.repair_end + CHANGEOVER_TIME
        _push(state, _Ev(state.slot_free_at[track], _P_TRACK_FREE, {'track': track}))
        state.pending_oven.append((job.repair_end, state.oven_seq, job))
        state.oven_seq += 1
        _try_oven_intake(state, t)
        state.sa_trigger = True
    else:
        # Phase boundary only; workers stay idle until next epoch (HC-3)
        slot.workers_B = 0
        slot.workers_C = 0


# ---------------------------------------------------------------------------
# OVEN_FREE / TRACK_FREE handlers
# ---------------------------------------------------------------------------

def _on_oven_free(state: _State, t: float) -> None:
    _try_oven_intake(state, t)


def _on_track_free(state: _State, t: float) -> None:
    _try_fill_tracks(state, t)


# ---------------------------------------------------------------------------
# Public DES simulation entry point
# ---------------------------------------------------------------------------

def simulate_des_bc(
    jobs:             list[Job],
    chromosome:       list[int],
    pool_B:           int  = 0,
    pool_C:           int  = 0,
    use_sa:           bool = False,
    use_lookahead:    bool = False,
    use_delays:       bool = True,
    delay_seed:       int  = DELAY_SEED,
    gen_seed:         int  = GA_SEED + 1,
    forced_workers:   dict | None = None,
    oven_offset:      float = 0.0,
    sa_history_out:   list | None = None,
) -> tuple[list[tuple[float, int, int]], float, float, float]:
    """
    DES equivalent of ga_simulationBC_current.simulate_ga_bc().

    Primary (buffer=0) jobs are always dispatched before buffer (buffer=1) jobs.
    Each time a buffer job is placed on a repair track a new PERT-sampled job is
    appended to the queue (up to MAX_GENERATED_JOBS total), mirroring
    ga_simulationBC_current.simulate_ga_bc().

    use_delays: if True, a Monte Carlo PERT delay is added to each blade on
                its first active epoch (sampled from the historical
                Actual-Estimated distribution in data_real.csv).

    Returns (staff_log, sc1_total, sc2_total, sc3_total).
    staff_log entries: (epoch_time, b_workers_used, c_workers_used).
    """
    if pool_B <= 0:
        pool_B = N_WORKERS_B
    if pool_C <= 0:
        pool_C = N_WORKERS_C

    n = len(chromosome) // 3

    # Fallback worker counts for generated jobs that have no chromosome gene,
    # computed as the average of buffer-job genes (same logic as ga_simulationBC_current).
    buf_indices = [j.job_index for j in jobs if j.is_buffered]
    if buf_indices:
        default_b = max(1, round(sum(chromosome[n + i] for i in buf_indices) / len(buf_indices)))
        default_c = max(1, round(sum(chromosome[2 * n + i] for i in buf_indices) / len(buf_indices)))
    else:
        from constraintsBC import MAX_WORKERS_PER_BLADE
        default_b = MAX_WORKERS_PER_BLADE // 2
        default_c = MAX_WORKERS_PER_BLADE // 2

    # Primary jobs always dispatched before buffer jobs; chromosome priority
    # controls ordering within each group.
    primary_q = sorted([j for j in jobs if not j.is_buffered], key=lambda j: chromosome[j.job_index])
    buffer_q  = sorted([j for j in jobs if j.is_buffered],     key=lambda j: chromosome[j.job_index])

    state = _State(
        slots              = [None] * MAX_REPAIR_TRACKS,
        slot_free_at       = [0.0]  * MAX_REPAIR_TRACKS,
        oven_free_at       = oven_offset,
        queue              = primary_q + buffer_q,
        pending_oven       = [],
        oven_seq           = 0,
        track_version      = [0] * MAX_REPAIR_TRACKS,
        alloc_since        = [0.0] * MAX_REPAIR_TRACKS,
        heap               = [],
        staff_log          = [],
        sc1_total          = 0.0,
        oven_starts        = [],
        oven_ends          = [],
        chromosome         = chromosome,
        n_chrom            = n,
        pool_B             = pool_B,
        pool_C             = pool_C,
        use_sa             = use_sa,
        use_lookahead      = use_lookahead,
        sa_rng             = random.Random(GA_SEED) if use_sa else None,
        sa_trigger         = False,
        use_delays         = use_delays,
        delay_rng          = random.Random(delay_seed) if use_delays else None,
        delay_applied      = set(),
        jobs_list          = jobs,
        generated_count    = 0,
        gen_rng            = random.Random(gen_seed),
        default_b          = default_b,
        default_c          = default_c,
        primary_remaining  = len(primary_q),
        forced_workers     = forced_workers or {},
        sa_history         = sa_history_out,
    )

    # If generation is enabled but no buffer blades were supplied, seed the
    # queue with a small initial batch so the replenishment chain can start.
    # Each seed job dispatched to a track will trigger one more PERT-sampled
    # job to be appended (up to MAX_GENERATED_JOBS total).
    if MAX_GENERATED_JOBS > 0 and not buffer_q:
        n_seed = min(3, MAX_GENERATED_JOBS)
        for _ in range(n_seed):
            new_h = pert_sample(state.gen_rng, _PERT_A, _PERT_MODE, _PERT_B)
            seed = Job(
                job_id      = f"GEN-{state.generated_count + 1:03d}",
                estimated_h = new_h,
                job_index   = state.n_chrom + state.generated_count,
                is_buffered = True,
            )
            state.generated_count += 1
            state.queue.append(seed)
            state.jobs_list.append(seed)

    _push(state, _Ev(0.0, _P_EPOCH))

    while state.heap:
        ev = heapq.heappop(state.heap)
        t = ev.time
        if t >= _MAX_T:
            break

        if ev.prio == _P_PHASE_DONE:
            d = ev.data
            _on_phase_done(state, t, d['track'], d['ver'], d['phase'])

        elif ev.prio == _P_EPOCH:
            if not _on_epoch(state, t):
                break

        elif ev.prio == _P_OVEN_FREE:
            _on_oven_free(state, t)

        elif ev.prio == _P_TRACK_FREE:
            _on_track_free(state, t)

    # Safety drain: blades that finished repair but were not yet moved to oven
    for rep_end, _, job in sorted(state.pending_oven):
        ov_start           = max(state.oven_free_at, rep_end)
        job.oven_start     = ov_start
        job.oven_end       = ov_start + OVEN_PROCESS_TIME
        state.oven_free_at = job.oven_end + CHANGEOVER_TIME
        state.oven_starts.append(ov_start)
        state.oven_ends.append(job.oven_end)

    # SC-2: total hours blades blocked waiting for oven
    done = [j for j in jobs if j.oven_start is not None and j.repair_end is not None]
    sc2_total = sum(max(0.0, j.oven_start - j.repair_end) for j in done)

    # SC-3: total oven idle hours between consecutive jobs
    if len(state.oven_starts) > 1:
        pairs    = sorted(zip(state.oven_starts, state.oven_ends))
        s_sorted = [p[0] for p in pairs]
        e_sorted = [p[1] for p in pairs]
        sc3_total = penalty_oven_idle(e_sorted, s_sorted)
    else:
        sc3_total = 0.0

    return state.staff_log, state.sc1_total, sc2_total, sc3_total


# ---------------------------------------------------------------------------
# Fitness wrapper (mirrors ga_simulationBC_current.evaluate)
# ---------------------------------------------------------------------------

def evaluate_des(chromosome: list[int],
                 jobs_template: list[Job],
                 pool_B: int,
                 pool_C: int) -> float:
    jobs_copy = _fresh_jobs(jobs_template)
    _, sc1, sc2, sc3 = simulate_des_bc(jobs_copy, chromosome, pool_B, pool_C)
    done = [j for j in jobs_copy if j.oven_end is not None]
    makespan = max((j.oven_end for j in done), default=float('inf'))
    return makespan + SC2_WEIGHT * sc2 + SC3_WEIGHT * sc3


# ---------------------------------------------------------------------------
# Gantt chart (DES version with MC delay annotations)
# ---------------------------------------------------------------------------

def _palette(n: int) -> list:
    base = [cm.tab20(i / 20) for i in range(20)] + \
           [cm.tab20b(i / 20) for i in range(20)]
    while len(base) < n:
        base += base
    return base[:n]


def plot_gantt_des(
    jobs:      list[Job],
    staff_log: list[tuple[float, int, int]],
    pool_B:    int = N_WORKERS_B,
    pool_C:    int = N_WORKERS_C,
    save_path: str = "sa_gantt_des_bc.png",
) -> None:
    """
    Four-panel Gantt chart for the DES run.
    Panel 1 annotates every repair bar with the Monte Carlo delay that was
    applied to that blade (D+X.Xh red = longer, D-X.Xh green = shorter).
    """
    done = [j for j in jobs if j.oven_end is not None]
    if not done:
        print("No finished jobs  -  nothing to plot.")
        return

    makespan  = max(j.oven_end for j in done)
    palette   = _palette(len(jobs))
    colors    = {j.job_id: palette[k] for k, j in enumerate(jobs)}
    bar_h     = 0.72

    n_primary = sum(1 for j in jobs if not j.is_buffered)
    n_buf     = sum(1 for j in jobs if j.is_buffered and not j.job_id.startswith("GEN-"))
    n_gen     = sum(1 for j in jobs if j.job_id.startswith("GEN-"))

    fig, axes = plt.subplots(
        4, 1,
        figsize=(16, 14),
        gridspec_kw={"height_ratios": [MAX_REPAIR_TRACKS * 0.6, 1.8, 2.0, 2.0]},
    )
    fig.suptitle(
        f"DES (B/C workers + MC delays)\n"
        f"{n_primary + n_buf} real jobs ({n_primary} primary + {n_buf} buffer)"
        f" + {n_gen} PERT-generated = {len(done)} completed",
        fontsize=12, fontweight="bold",
    )

    # ---- Panel 1: Repair tracks ------------------------------------------
    ax1 = axes[0]
    ax1.set_title(
        "Repair tracks  "
        "(solid = Phase 1/3: C workers | hatched = Phase 2: B workers | "
        "D label = MC delay | BUF-xxx = navy border, GEN-xxx = darkgreen border)",
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

        # Phase 1 bar
        p1_s = j.repair_start
        p1_e = j.phase1_end if j.phase1_end is not None else j.repair_end
        if p1_s is not None and p1_e is not None and p1_e > p1_s:
            ax1.barh(y, p1_e - p1_s, left=p1_s,
                     height=bar_h, color=col, edgecolor=edge_col, linewidth=edge_lw)

        # Phase 2 bar
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
                         f"{j.job_id}\n{ratio:.0%} of est.",
                         ha="center", va="center",
                         fontsize=6, color="black", fontweight="bold",
                         clip_on=True)

        # Phase 3 bar
        p3_s = p2_e if (p2_e is not None and j.phase2_h > 0) else p1_e
        p3_e = j.repair_end
        if p3_s is not None and p3_e is not None and p3_e > p3_s and j.phase3_h > 0:
            ax1.barh(y, p3_e - p3_s, left=p3_s,
                     height=bar_h, color=col, alpha=0.70,
                     edgecolor=edge_col, linewidth=edge_lw)

        # MC delay annotation  -  placed just above the centre of the full repair bar
        delay = getattr(j, "mc_delay", None)
        if delay is not None and j.repair_start is not None and j.repair_end is not None:
            cx = j.repair_start + (j.repair_end - j.repair_start) / 2
            if abs(delay) < 0.05:
                dcol = "#888888"
            elif delay > 0:
                dcol = "#cc2200"   # red  â†’ job took longer than estimated
            else:
                dcol = "#007722"   # green â†’ job finished faster than estimated
            ax1.text(cx, y + bar_h / 2 + 0.04,
                     f"D{delay:+.1f}h",
                     ha="center", va="bottom", fontsize=5,
                     color=dcol, clip_on=True, zorder=5)

    p1_patch  = mpatches.Patch(color="grey",
                                label=f"Phase 1: C workers ({PHASE1_H:.0f} h)")
    p2_patch  = mpatches.Patch(color="grey", alpha=0.45, hatch="///",
                                label="Phase 2: B workers (middle)")
    p3_patch  = mpatches.Patch(color="grey", alpha=0.70,
                                label=f"Phase 3: C workers ({PHASE3_H:.0f} h)")
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
        " -  navy border = buffer job, darkgreen border = generated job",
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
        ts, _, cs = zip(*staff_log)
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
        ts, bs, _ = zip(*staff_log)
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

    fig.subplots_adjust(top=0.93)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Gantt chart (DES) saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Weekly throughput chart
# ---------------------------------------------------------------------------

def plot_weekly_throughput(
    jobs:      list[Job],
    makespan:  float,
    save_path: str = "sa_weekly_throughput.png",
) -> None:
    """
    Bar chart of blades completing (oven_end) per 144-hour simulation week,
    with a cumulative total as a secondary axis.
    """
    done = [j for j in jobs if j.oven_end is not None]
    if not done:
        print("No finished jobs  -  nothing to plot.")
        return

    n_weeks = int(np.ceil(makespan / WEEK_HOURS))
    counts  = np.zeros(n_weeks, dtype=int)
    for j in done:
        wk = min(int(j.oven_end / WEEK_HOURS), n_weeks - 1)
        counts[wk] += 1

    week_labels = [f"Wk {i + 1}" for i in range(n_weeks)]
    cumulative  = np.cumsum(counts)
    avg         = len(done) / n_weeks

    fig, ax1 = plt.subplots(figsize=(max(8, n_weeks * 0.9), 5))

    bars = ax1.bar(week_labels, counts,
                   color="#5BA4CF", edgecolor="white", linewidth=0.6, zorder=3)
    ax1.axhline(avg, color="#E05C2A", linestyle="--", linewidth=1.4,
                label=f"Average  {avg:.1f} blades/week", zorder=4)

    for bar, val in zip(bars, counts):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.15,
                     str(val), ha="center", va="bottom", fontsize=8)

    ax1.set_xlabel("Simulation week  (1 week = 144 h)")
    ax1.set_ylabel("Blades completed", color="#5BA4CF")
    ax1.tick_params(axis="y", labelcolor="#5BA4CF")
    ax1.set_ylim(0, max(counts) * 1.25)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.legend(loc="upper left", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(week_labels, cumulative,
             color="#2E7D32", marker="o", linewidth=2.0,
             markersize=5, label="Cumulative total", zorder=5)
    ax2.set_ylabel("Cumulative blades completed", color="#2E7D32")
    ax2.tick_params(axis="y", labelcolor="#2E7D32")
    ax2.set_ylim(0, len(done) * 1.15)
    ax2.legend(loc="upper right", fontsize=9)

    ax1.set_title(
        f"Weekly throughput  -  {len(done)} blades  |  makespan {makespan:.0f} h  "
        f"({n_weeks} weeks)",
        fontsize=11, fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Weekly throughput chart saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _export_csv(
    jobs:       list[Job],
    makespan:   float,
    sc1:        float,
    sc2:        float,
    sc3:        float,
    pool_B:     int,
    pool_C:     int,
    use_delays: bool,
    save_path:  str = "sa_des_simulation_results.csv",
) -> None:
    """Append one row of KPIs to a CSV file (creates file + header on first run)."""
    done = [j for j in jobs if j.oven_end is not None and j.repair_end is not None]

    repair_times   = [j.repair_end - j.repair_start for j in done if j.repair_start is not None]
    avg_repair_time = sum(repair_times) / len(repair_times) if repair_times else 0.0

    # Total changeover (blocking) time consumed across all tracks
    track_blocking = CHANGEOVER_TIME * len(done)

    # Total idle time: for each track, makespan minus (repair + changeover) on that track
    total_track_idle = 0.0
    for i in range(MAX_REPAIR_TRACKS):
        track_done = [j for j in done if j.repair_track == i]
        occupied   = sum((j.repair_end - j.repair_start) + CHANGEOVER_TIME for j in track_done)
        total_track_idle += max(0.0, makespan - occupied)

    # Oven utilisation: fraction of makespan the oven is actively processing
    oven_utilization = (len(done) * OVEN_PROCESS_TIME) / makespan if makespan > 0 else 0.0

    row = {
        "n_jobs":            len(done),
        "pool_B":            pool_B,
        "pool_C":            pool_C,
        "mc_delays":         use_delays,
        "makespan_h":        round(makespan, 2),
        "avg_repair_time_h": round(avg_repair_time, 2),
        "track_idle_time_h": round(total_track_idle, 2),
        "track_blocking_h":  round(track_blocking, 2),
        "oven_utilization":  round(oven_utilization, 4),
        "sc1_penalty":       round(sc1, 2),
        "sc2_penalty":       round(sc2, 2),
        "sc3_penalty":       round(sc3, 2),
    }

    write_header = not os.path.exists(save_path)
    with open(save_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"  Results CSV saved -> {save_path}")


# ---------------------------------------------------------------------------
# Worker distribution chart (post-simulation)
# ---------------------------------------------------------------------------

def plot_worker_distribution(
    jobs:      list[Job],
    pool_B:    int = N_WORKERS_B,
    pool_C:    int = N_WORKERS_C,
    save_path: str = "sa_worker_distribution.png",
) -> None:
    """
    Stacked bar chart showing per-blade worker allocation at each 4-h epoch.
    Panel 1: B-workers (phase 2) per blade.
    Panel 2: C-workers (phases 1 & 3) per blade.
    """
    epochs_b: dict[float, dict[str, int]] = {}
    epochs_c: dict[float, dict[str, int]] = {}

    for j in jobs:
        for t_start, _t_end, w, wtype in j.worker_log:
            if wtype == 'B':
                epochs_b.setdefault(t_start, {})[j.job_id] = w
            else:
                epochs_c.setdefault(t_start, {})[j.job_id] = w

    if not epochs_b and not epochs_c:
        print("No worker allocation data  -  skipping worker distribution chart.")
        return

    all_epochs  = sorted(t for t in set(epochs_b) | set(epochs_c)
                         if t < 6 * WORKER_REASSIGN_INTERVAL)
    all_job_ids = [j.job_id for j in jobs]
    palette     = _palette(len(jobs))
    colors      = {j.job_id: palette[k] for k, j in enumerate(jobs)}
    bar_w       = WORKER_REASSIGN_INTERVAL * 0.82

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(max(10, len(all_epochs) * 0.55), 9),
        sharex=True,
    )
    fig.suptitle(
        "Recommended worker distribution  -  first 24 h  (6 Ã- 4-h epochs)",
        fontsize=13, fontweight="bold",
    )

    def _draw_stacked(ax: plt.Axes, data: dict, pool: int, title: str) -> None:
        ax.set_title(title, fontsize=10)
        for t in all_epochs:
            alloc  = data.get(t, {})
            bottom = 0
            for jid in all_job_ids:
                w = alloc.get(jid, 0)
                if w <= 0:
                    continue
                ax.bar(t, w, bottom=bottom, width=bar_w,
                       color=colors[jid], align="edge",
                       edgecolor="white", linewidth=0.4)
                ax.text(t + bar_w / 2, bottom + w / 2,
                        jid[:7],
                        ha="center", va="center",
                        fontsize=5, color="white", clip_on=True)
                bottom += w
        ax.axhline(pool, color="red", linestyle="--",
                   linewidth=1.3, label=f"Pool limit ({pool})")
        ax.set_ylim(0, pool * 1.2)
        ax.set_ylabel("Workers assigned")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    _draw_stacked(ax1, epochs_b, pool_B,
                  f"B-workers  (phase 2)   -   pool = {pool_B}")
    _draw_stacked(ax2, epochs_c, pool_C,
                  f"C-workers  (phases 1 & 3)   -   pool = {pool_C}")

    ax2.set_xlabel("Simulation time (h)")
    if all_epochs:
        ax2.set_xlim(
            max(0, min(all_epochs) - 2),
            max(all_epochs) + WORKER_REASSIGN_INTERVAL + 2,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Worker distribution chart saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# SA convergence chart
# ---------------------------------------------------------------------------

def plot_sa_convergence_des(
    sa_history: list[tuple[float, list[float]]],
    save_path:  str = "sa_sa_convergence_des.png",
) -> None:
    """
    Two-panel convergence chart for the SA worker-allocation calls made
    during a DES run.

    Panel 1  -  Within-call convergence:
        Each SA call (one per active epoch) is drawn as a semi-transparent
        line: best fitness vs. iteration step within that call.  Lines are
        colour-mapped from blue (early epoch) to orange (late epoch).

    Panel 2  -  Across-call improvement:
        Bar chart of the absolute improvement (initial âˆ’ final best fitness)
        achieved by each SA call, grouped by epoch time.  A positive bar
        means SA found a strictly better allocation than the GA seed.
    """
    if not sa_history:
        print("No SA history recorded  -  was use_sa=True and sa_history_out passed?")
        return

    epoch_times  = [t for t, _ in sa_history]
    step_curves  = [hist for _, hist in sa_history]
    n_calls      = len(sa_history)

    cmap   = plt.get_cmap("plasma")
    colors = [cmap(i / max(n_calls - 1, 1)) for i in range(n_calls)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(
        f"SA worker-allocation convergence  ({n_calls} SA calls during DES run)\n"
        f"SA params: Tâ‚€={SA_TEMP_INIT}  cooling={SA_COOLING}  iterations={SA_ITERATIONS}",
        fontsize=12, fontweight="bold",
    )

    # ---- Panel 1: within-call best-fitness curves -----------------------
    ax1.set_title("Within-call convergence  -  best fitness vs. SA iteration", fontsize=10)
    for i, (curve, col) in enumerate(zip(step_curves, colors)):
        ax1.plot(range(len(curve)), curve,
                 color=col, alpha=0.55, linewidth=0.9)

    # Overlay the mean best-fitness across all calls at each step
    max_steps = max(len(c) for c in step_curves)
    mean_curve = []
    for s in range(max_steps):
        vals = [c[s] for c in step_curves if s < len(c)]
        mean_curve.append(sum(vals) / len(vals))
    ax1.plot(range(len(mean_curve)), mean_curve,
             color="black", linewidth=2.0, label="Mean across all calls", zorder=5)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=min(epoch_times),
                                                   vmax=max(epoch_times)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, pad=0.01)
    cbar.set_label("Epoch time (h)", fontsize=8)

    ax1.set_xlabel("SA iteration step")
    ax1.set_ylabel("Best SA fitness (lower = better)")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(alpha=0.3)

    # ---- Panel 2: per-call improvement -----------------------------------
    ax2.set_title("Per-call improvement  (seed fitness âˆ’ best found)", fontsize=10)
    improvements = [curve[0] - min(curve) for curve in step_curves]
    bar_cols     = [("#2196F3" if imp > 0 else "#BDBDBD") for imp in improvements]
    ax2.bar(range(n_calls), improvements, color=bar_cols, edgecolor="none")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("SA call index  (chronological order during simulation)")
    ax2.set_ylabel("Fitness improvement")

    # Annotate epoch times on every 5th bar to avoid clutter
    for i, (imp, et) in enumerate(zip(improvements, epoch_times)):
        if i % max(1, n_calls // 10) == 0:
            ax2.text(i, max(imp, 0) + max(improvements, default=1) * 0.01,
                     f"t={et:.0f}h", ha="center", va="bottom",
                     fontsize=6, rotation=45)

    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  SA convergence chart saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DES blade repair simulation (B/C workers)  -  "
                    "uses GA chromosome from ga_simulationBC_current"
    )
    parser.add_argument("--workers_b",   type=int, default=N_WORKERS_B)
    parser.add_argument("--workers_c",   type=int, default=N_WORKERS_C)
    parser.add_argument("--jobs",        type=int, default=None)
    parser.add_argument("--generations", type=int, default=GA_GENERATIONS)
    parser.add_argument("--popsize",     type=int, default=GA_POP_SIZE)
    parser.add_argument("--compare",    action="store_true",
                        help="Also run original GA sim for side-by-side comparison")
    parser.add_argument("--no_delays", action="store_true",
                        help="Disable Monte Carlo PERT delays (deterministic run)")
    parser.add_argument("--delay_seed", type=int, default=DELAY_SEED,
                        help=f"RNG seed for delay sampling (default: {DELAY_SEED})")
    parser.add_argument("--max_gen_jobs", type=int, default=MAX_GENERATED_JOBS,
                        help=f"Max PERT-generated jobs spawned by buffer replenishment "
                             f"(default: {MAX_GENERATED_JOBS})")
    parser.add_argument("--force_workers", type=str, default="",
                        help="Comma-separated job_index:workers pairs, e.g. '0:5,2:8'. "
                             "Pins worker count for those jobs for their entire repair.")
    parser.add_argument("--oven_offset", type=float, default=0.0,
                        help="Hours from t=0 until the oven is free for its first blade "
                             "(simulates the oven already being mid-cycle at start; default: 0)")
    args = parser.parse_args()

    # Override MAX_GENERATED_JOBS in both modules so _try_fill_tracks and
    # evaluate (in ga_simulationBC_current) both respect the user-supplied cap.
    import ga_simulationBC_current as _ga_mod
    _ga_mod.MAX_GENERATED_JOBS = args.max_gen_jobs
    MAX_GENERATED_JOBS = args.max_gen_jobs  # rebind this module's global

    # Parse forced-worker assignments: "0:5,2:8" -> {0: 5, 2: 8}
    forced_workers: dict = {}
    if args.force_workers.strip():
        for pair in args.force_workers.split(","):
            idx_str, w_str = pair.split(":")
            forced_workers[int(idx_str.strip())] = int(w_str.strip())
    if forced_workers:
        print(f"Forced workers: { {f'job {k}': v for k, v in forced_workers.items()} }")

    pool_B = args.workers_b
    pool_C = args.workers_c

    jobs_template = load_jobs()
    if args.jobs is not None:
        jobs_template = jobs_template[:args.jobs]
    n = len(jobs_template)

    ests = [j.estimated_h for j in jobs_template]
    print(f"Loaded {n} jobs  |  B workers: {pool_B}  |  C workers: {pool_C}")
    print(f"  min est = {min(ests):.1f} h  |  max est = {max(ests):.1f} h  "
          f"|  avg est = {sum(ests)/n:.1f} h")

    print(f"\nRunning GA  (pop={args.popsize}, gen={args.generations}) ...")
    best_chrom, best_fit_ga, history = run_ga(
        jobs_template,
        pool_B=pool_B, pool_C=pool_C,
        pop_size=args.popsize, n_gen=args.generations,
    )
    print(f"Best GA fitness : {best_fit_ga:.1f}")

    print("\nRunning final DES with SA worker refinement every 4 h ...")
    des_jobs = load_jobs()
    if args.jobs is not None:
        des_jobs = des_jobs[:args.jobs]

    use_delays = not args.no_delays
    sa_hist: list = []
    staff_log, sc1, sc2, sc3 = simulate_des_bc(
        des_jobs, best_chrom, pool_B, pool_C,
        use_sa=True, use_lookahead=True,
        use_delays=use_delays, delay_seed=args.delay_seed,
        forced_workers=forced_workers,
        oven_offset=args.oven_offset,
        sa_history_out=sa_hist,
    )
    print(f"  Monte Carlo delays : {'ON' if use_delays else 'OFF'}  "
          f"(seed={args.delay_seed})")

    done_des = [j for j in des_jobs if j.oven_end is not None]
    print(f"\n=== DES results ===")
    print(f"Completed  : {len(done_des)}/{n} jobs")
    if done_des:
        makespan_des = max(j.oven_end for j in done_des)
        avg_B        = sum(j.avg_workers_B for j in done_des) / len(done_des)
        avg_C        = sum(j.avg_workers_C for j in done_des) / len(done_des)
        fitness_des  = makespan_des + SC2_WEIGHT*sc2 + SC3_WEIGHT*sc3
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

    if args.compare:
        from ga_simulationBC_current import simulate_ga_bc
        print("\nRunning original GA sim for comparison ...")
        ga_jobs = load_jobs()
        if args.jobs is not None:
            ga_jobs = ga_jobs[:args.jobs]
        staff_log_ga, sc1_ga, sc2_ga, sc3_ga = simulate_ga_bc(
            ga_jobs, best_chrom, pool_B, pool_C,
            use_sa=True, use_lookahead=True,
        )
        done_ga = [j for j in ga_jobs if j.oven_end is not None]
        if done_ga:
            makespan_ga = max(j.oven_end for j in done_ga)
            fitness_ga  = makespan_ga + SC2_WEIGHT*sc2_ga + SC3_WEIGHT*sc3_ga
            print(f"\n=== GA sim results (original) ===")
            print(f"Makespan  : {makespan_ga:.1f} h")
            print(f"Fitness   : {fitness_ga:.1f}")
            print(f"\nDES improvement : "
                  f"{fitness_ga - fitness_des:+.1f}  "
                  f"({'better' if fitness_des < fitness_ga else 'worse or equal'})")

    if done_des:
        _export_csv(
            des_jobs, makespan_des, sc1, sc2, sc3,
            pool_B=pool_B, pool_C=pool_C,
            use_delays=use_delays,
            save_path=f"sa_des_simulation_results_{n}jobs.csv",
        )
        plot_weekly_throughput(
            des_jobs, makespan_des,
            save_path=f"sa_weekly_throughput_{n}jobs.png",
        )

    plot_gantt_des(des_jobs, staff_log, pool_B=pool_B, pool_C=pool_C,
                   save_path=f"sa_gantt_des_bc_{n}jobs.png")
    plot_convergence(history, save_path=f"sa_des_bc_convergence_{n}jobs.png")
    plot_sa_convergence_des(sa_hist, save_path=f"sa_sa_convergence_des_{n}jobs.png")
    plot_worker_distribution(des_jobs, pool_B=pool_B, pool_C=pool_C,
                             save_path=f"sa_worker_distribution_{n}jobs.png")

