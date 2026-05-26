"""
des_sot_baseline.py
===================
Discrete-Event Simulation (DES) of the SOT blade repair scheduling from
sot_simulationBC_baseline.py.

Instead of advancing in fixed 4-h epochs, an event heap (heapq) drives the
simulation. Events are processed in chronological order; state changes only
when something actually happens.

Event types (lower priority value = processed first at equal time):
  PHASE_DONE  (0) - exact moment a repair phase ends on a track
  EPOCH       (1) - every WORKER_REASSIGN_INTERVAL h; reallocate workers
  OVEN_FREE   (2) - oven + changeover done; accept next waiting blade
  TRACK_FREE  (3) - track + changeover done; assign next job from queue

Key differences from the time-stepped sot_simulationBC_baseline.simulate_sot_bc():
  - Oven transfers and track fills happen at the exact available instant,
    not at the next 4-h epoch boundary.
  - Phase-completion times are computed exactly and recorded precisely.
  - Stale PHASE_DONE events (version-mismatched) are silently discarded,
    so rescheduling after a worker reallocation is cheap and correct.
  - Tracks are freed immediately at repair completion (HC-12), matching
    sot_simulationBC_baseline.simulate_sot_bc() semantics.

No GA, no chromosome, no SA -- stochastic DES run using pure SOT with
Monte Carlo PERT delays applied to phase 2 of each blade (sampled from the
historical Actual-Estimated distribution in data_real.csv).
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

from sot_simulationBC_baseline import (
    Job,
    _ActiveSlot,
    load_jobs,
    N_WORKERS_B,
    N_WORKERS_C,
    WEEK_HOURS,
    PHASE1_H,
    PHASE3_H,
    HC2_EFF_CAP,
    SC1_WEIGHT,
    SC2_WEIGHT,
    SC3_WEIGHT,
)
from constraintsBC import (
    BLADE_QUEUE_LOOKAHEAD,
    CHANGEOVER_TIME,
    MAX_REPAIR_TRACKS,
    MAX_WORKERS_PER_BLADE,
    MIN_PROCESS_TIME_RATIO,
    OVEN_PROCESS_TIME,
    WORKER_REASSIGN_INTERVAL,
    efficiency_factor,
    get_candidate_blades,
    penalty_worker_band,
    penalty_oven_idle,
    enforce_full_worker_utilization_B,
    enforce_full_worker_utilization_C,
    enforce_empty_track_fill,
)

# ---------------------------------------------------------------------------
# Event priorities (lower = fires first at equal timestamp)
# ---------------------------------------------------------------------------
_P_PHASE_DONE = 0
_P_EPOCH      = 1
_P_OVEN_FREE  = 2
_P_TRACK_FREE = 3

_MAX_T     = 15_000.0
DELAY_SEED = 7   # RNG seed for Monte Carlo delay sampling


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
    a    = float(min(delays))
    b    = float(max(delays))
    counts, edges = np.histogram(delays, bins="auto")
    mode = float((edges[np.argmax(counts)] + edges[np.argmax(counts) + 1]) / 2)
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
    u = rng.random()
    z = float(_scipy_beta.ppf(u, alpha1, alpha2))
    return a + z * (b - a)


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
    slot_free_at:  list[float]
    oven_free_at:  float
    queue:         list[Job]
    pending_oven:  list                         # (repair_end, seq, job) sorted by repair_end
    oven_seq:      int
    track_version: list[int]                    # PHASE_DONE invalidation token per track
    alloc_since:   list[float]                  # time of last worker assignment per track
    heap:          list                         # heapq of _Ev
    staff_log:     list[tuple[float, int, int]]
    sc1_total:     float
    oven_starts:   list[float]
    oven_ends:     list[float]
    pool_B:        int
    pool_C:        int
    use_delays:    bool
    delay_rng:     Optional[random.Random]
    delay_applied: set                          # job_index values already delayed


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
        return  # workers=0 -> job paused; no PHASE_DONE scheduled
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
    """Transfer pending blades to the oven in repair-end order."""
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
        # do NOT increment i -- list shortened


# ---------------------------------------------------------------------------
# Track filling (SOT dispatching)
# ---------------------------------------------------------------------------

def _try_fill_tracks(state: _State, t: float) -> None:
    """Assign jobs from the queue to every free repair track using SOT selection."""
    for i in range(MAX_REPAIR_TRACKS):
        if state.slots[i] is not None or state.slot_free_at[i] > t or not state.queue:
            continue
        candidates = get_candidate_blades(state.queue, BLADE_QUEUE_LOOKAHEAD)
        job = min(candidates, key=lambda j: j.estimated_h)
        state.queue.remove(job)

        aj = _ActiveSlot(job=job, track_idx=i,
                         p1_rem=job.phase1_h, p2_rem=job.phase2_h, p3_rem=job.phase3_h)
        job.repair_track  = i
        job.repair_start  = t
        if job.skip_phase1:
            job.phase1_end = t
        state.slots[i]       = aj
        state.alloc_since[i] = t

    enforce_empty_track_fill(state.slots, state.slot_free_at, state.queue, t)


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

    # 2. Fill empty tracks from queue (SOT)
    _try_fill_tracks(state, t)

    # 3. Advance remaining work (from last alloc time to now) for every active slot
    for i, slot in enumerate(state.slots):
        if slot is not None and slot.remaining_h > 1e-9:
            _advance_work(slot, state.alloc_since[i], t)
            state.alloc_since[i] = t

    # 4. SOT worker allocation
    active   = [s for s in state.slots if s is not None and s.remaining_h > 1e-9]
    b_active = sorted([s for s in active if s.phase == 2],      key=lambda s: s.remaining_h)
    c_active = sorted([s for s in active if s.phase in (1, 3)], key=lambda s: s.remaining_h)

    for s in b_active: s.workers_B = 0
    for s in c_active: s.workers_C = 0

    # B workers: guarantee 1 to each (shortest first), then top-up
    pool_left_B = state.pool_B
    for s in b_active:
        if pool_left_B <= 0: break
        s.workers_B = 1; pool_left_B -= 1
    for s in b_active:
        if pool_left_B <= 0: break
        give = min(MAX_WORKERS_PER_BLADE - s.workers_B, pool_left_B)
        s.workers_B += give; pool_left_B -= give
    if b_active:
        alloc_b = {s.job.job_index: s.workers_B for s in b_active}
        alloc_b = enforce_full_worker_utilization_B(alloc_b, state.pool_B)
        for s in b_active:
            s.workers_B = alloc_b[s.job.job_index]

    # C workers: guarantee 1 to each (shortest first), then top-up
    pool_left_C = state.pool_C
    for s in c_active:
        if pool_left_C <= 0: break
        s.workers_C = 1; pool_left_C -= 1
    for s in c_active:
        if pool_left_C <= 0: break
        give = min(MAX_WORKERS_PER_BLADE - s.workers_C, pool_left_C)
        s.workers_C += give; pool_left_C -= give
    if c_active:
        alloc_c = {s.job.job_index: s.workers_C for s in c_active}
        alloc_c = enforce_full_worker_utilization_C(alloc_c, state.pool_C)
        for s in c_active:
            s.workers_C = alloc_c[s.job.job_index]

    # SC-1 penalty and worker log
    for s in b_active:
        state.sc1_total += penalty_worker_band(s.workers_B) * dt
        s.job.worker_log.append((t, t + dt, s.workers_B, 'B'))
    for s in c_active:
        s.job.worker_log.append((t, t + dt, s.workers_C, 'C'))

    # 4b. Final-epoch Monte Carlo delay: sample a PERT delay in the last epoch
    #     before phase 2 completes (p2_rem / eff <= dt).
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
            d      = _sample_pert_delay(state.delay_rng, slot.job)
            old_p2 = slot.p2_rem
            # HC-2 floor: total phase-2 duration must stay >= MIN_PROCESS_TIME_RATIO
            # accounting for work already done so the floor shrinks as job progresses.
            p2_initial   = slot.job.phase2_h
            p2_done      = max(0.0, p2_initial - old_p2)
            min_p2_total = max(0.0,
                               MIN_PROCESS_TIME_RATIO * slot.job.estimated_h
                               - slot.job.phase1_h - slot.job.phase3_h)
            floor_p2      = max(0.0, min_p2_total - p2_done)
            slot.p2_rem   = max(old_p2 + d, floor_p2)
            slot.job.mc_delay = getattr(slot.job, "mc_delay", 0.0) + (slot.p2_rem - old_p2)
            if slot.p2_rem <= 1e-9:
                slot.p2_rem         = 0.0
                slot.job.phase2_end = t

    # 5. Schedule PHASE_DONE for each active track with current allocation
    for i in range(MAX_REPAIR_TRACKS):
        if state.slots[i] is not None and state.slots[i].remaining_h > 1e-9:
            _schedule_phase_done(state, i, t)

    # Staff log
    total_B = sum(s.workers_B for s in state.slots if s is not None)
    total_C = sum(s.workers_C for s in state.slots if s is not None)
    state.staff_log.append((t, total_B, total_C))

    # Termination check
    if (not state.queue
            and not state.pending_oven
            and all(s is None for s in state.slots)):
        return False

    # Next epoch
    if t + dt < _MAX_T:
        _push(state, _Ev(t + dt, _P_EPOCH))
    return True


# ---------------------------------------------------------------------------
# PHASE_DONE handler
# ---------------------------------------------------------------------------

def _on_phase_done(state: _State, t: float, track: int, ver: int, phase: int) -> None:
    """
    Exact phase completion. Stale events (version mismatch) are discarded.
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

def simulate_des_sot(
    jobs:        list[Job],
    pool_B:      int  = 0,
    pool_C:      int  = 0,
    use_delays:  bool = True,
    delay_seed:  int  = DELAY_SEED,
) -> tuple[list[tuple[float, int, int]], float, float, float]:
    """
    DES equivalent of sot_simulationBC_baseline.simulate_sot_bc().

    First MAX_REPAIR_TRACKS jobs pre-sorted by estimated_h are placed on tracks
    at t=0. Remaining jobs enter the queue in arrival order.
    Each time a track becomes free the shortest job among the first
    BLADE_QUEUE_LOOKAHEAD candidates is selected (SOT).
    Worker allocation runs at every WORKER_REASSIGN_INTERVAL epoch using the
    SOT guarantee-then-top-up heuristic from sot_simulationBC_baseline.

    use_delays: if True, a Monte Carlo PERT delay is added to each blade in
                the last epoch before phase 2 completes (sampled from the
                historical Actual-Estimated distribution in data_real.csv).

    Returns (staff_log, sc1_total, sc2_total, sc3_total).
    """
    if pool_B <= 0:
        pool_B = N_WORKERS_B
    if pool_C <= 0:
        pool_C = N_WORKERS_C

    slots:        list[Optional[_ActiveSlot]] = [None] * MAX_REPAIR_TRACKS
    slot_free_at: list[float]                 = [0.0]  * MAX_REPAIR_TRACKS
    alloc_since:  list[float]                 = [0.0]  * MAX_REPAIR_TRACKS

    # Pre-fill all tracks with the first MAX_REPAIR_TRACKS jobs sorted SOT at t=0,
    # mirroring sot_simulationBC_baseline.simulate_sot_bc().
    for i, job in enumerate(sorted(jobs[:MAX_REPAIR_TRACKS], key=lambda j: j.estimated_h)):
        aj = _ActiveSlot(job=job, track_idx=i,
                         p1_rem=job.phase1_h, p2_rem=job.phase2_h, p3_rem=job.phase3_h)
        job.repair_track = i
        job.repair_start = 0.0
        job.phase1_end   = 0.0   # skip_phase1=True for all initial track jobs
        slots[i]         = aj

    state = _State(
        slots         = slots,
        slot_free_at  = slot_free_at,
        oven_free_at  = 0.0,
        queue         = list(jobs[MAX_REPAIR_TRACKS:]),
        pending_oven  = [],
        oven_seq      = 0,
        track_version = [0] * MAX_REPAIR_TRACKS,
        alloc_since   = alloc_since,
        heap          = [],
        staff_log     = [],
        sc1_total     = 0.0,
        oven_starts   = [],
        oven_ends     = [],
        pool_B        = pool_B,
        pool_C        = pool_C,
        use_delays    = use_delays,
        delay_rng     = random.Random(delay_seed) if use_delays else None,
        delay_applied = set(),
    )

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
# Gantt chart (DES version)
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
    save_path: str = "gantt_des_sot_bc_baseline.png",
) -> None:
    """Four-panel Gantt chart for the DES SOT run."""
    done = [j for j in jobs if j.oven_end is not None]
    if not done:
        print("No finished jobs -- nothing to plot.")
        return

    makespan = max(j.oven_end for j in done)
    palette  = _palette(len(jobs))
    colors   = {j.job_id: palette[k] for k, j in enumerate(jobs)}
    bar_h    = 0.72

    fig, axes = plt.subplots(
        4, 1,
        figsize=(16, 14),
        gridspec_kw={"height_ratios": [MAX_REPAIR_TRACKS * 0.6, 1.8, 2.0, 2.0]},
    )
    fig.suptitle(
        f"DES SOT (B/C workers) -- baseline (data_real.csv)  "
        f"({len(done)} jobs completed from {len(jobs)} loaded)",
        fontsize=13, fontweight="bold",
    )

    # ---- Panel 1: Repair tracks ------------------------------------------
    ax1 = axes[0]
    ax1.set_title(
        "Repair tracks  "
        "(solid = Phase 1/3: C workers | hatched = Phase 2: B workers | "
        "Delta label = MC delay)",
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

        # MC delay annotation -- placed just above the centre of the full repair bar
        delay = getattr(j, "mc_delay", None)
        if delay is not None and j.repair_start is not None and j.repair_end is not None:
            cx = j.repair_start + (j.repair_end - j.repair_start) / 2
            if abs(delay) < 0.05:
                dcol = "#888888"
            elif delay > 0:
                dcol = "#cc2200"   # red  -> job took longer than estimated
            else:
                dcol = "#007722"   # green -> job finished faster than estimated
            ax1.text(cx, y + bar_h / 2 + 0.04,
                     f"D{delay:+.1f}h",
                     ha="center", va="bottom", fontsize=5,
                     color=dcol, clip_on=True, zorder=5)

    p1_patch = mpatches.Patch(color="grey",             label=f"Phase 1: C workers ({PHASE1_H:.0f} h)")
    p2_patch = mpatches.Patch(color="grey", alpha=0.45, hatch="///", label="Phase 2: B workers")
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
            ax2.text(j.oven_start + w / 2, 0, j.job_id[:8],
                     ha="center", va="center",
                     fontsize=5.5, color="black", fontweight="bold", rotation=90)

    ax2.axvline(WEEK_HOURS, color="red", linestyle="--", linewidth=1.2)
    ax2.grid(axis="x", alpha=0.3)

    # ---- Panel 3: C worker utilization -----------------------------------
    ax3 = axes[2]
    ax3.set_title("C worker utilization  (SOT allocation)", fontsize=10)
    if staff_log:
        ts, _, cs = zip(*staff_log)
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
    ax4.set_title("B worker utilization  (SOT allocation)", fontsize=10)
    if staff_log:
        ts, bs, _ = zip(*staff_log)
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
    print(f"  Gantt chart (DES) saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Weekly throughput chart
# ---------------------------------------------------------------------------

def plot_weekly_throughput(
    jobs:      list[Job],
    makespan:  float,
    save_path: str = "des_sot_weekly_throughput.png",
) -> None:
    """
    Bar chart of blades completing (oven_end) per 144-hour simulation week,
    with a cumulative total as a secondary axis.
    """
    done = [j for j in jobs if j.oven_end is not None]
    if not done:
        print("No finished jobs -- nothing to plot.")
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
        f"DES SOT -- Weekly throughput  |  {len(done)} blades  |  makespan {makespan:.0f} h  "
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
    jobs:        list[Job],
    makespan:    float,
    sc1:         float,
    sc2:         float,
    sc3:         float,
    pool_B:      int,
    pool_C:      int,
    use_delays:  bool = True,
    save_path:   str = "des_sot_simulation_results.csv",
) -> None:
    """Append one row of KPIs to a CSV file (creates file + header on first run)."""
    done = [j for j in jobs if j.oven_end is not None and j.repair_end is not None]

    repair_times    = [j.repair_end - j.repair_start for j in done if j.repair_start is not None]
    avg_repair_time = sum(repair_times) / len(repair_times) if repair_times else 0.0

    track_blocking = CHANGEOVER_TIME * len(done)

    total_track_idle = 0.0
    for i in range(MAX_REPAIR_TRACKS):
        track_done = [j for j in done if j.repair_track == i]
        occupied   = sum((j.repair_end - j.repair_start) + CHANGEOVER_TIME for j in track_done)
        total_track_idle += max(0.0, makespan - occupied)

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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DES SOT blade repair simulation using data_real.csv (B/C workers)"
    )
    parser.add_argument("--workers_b", type=int, default=N_WORKERS_B,
                        help=f"B worker pool (default: {N_WORKERS_B})")
    parser.add_argument("--workers_c", type=int, default=N_WORKERS_C,
                        help=f"C worker pool (default: {N_WORKERS_C})")
    parser.add_argument("--csv",        type=str, default=None,
                        help="Path to data_real.csv (default: auto-detected)")
    parser.add_argument("--no_delays", action="store_true",
                        help="Disable Monte Carlo PERT delays (deterministic run)")
    parser.add_argument("--delay_seed", type=int, default=DELAY_SEED,
                        help=f"RNG seed for delay sampling (default: {DELAY_SEED})")
    parser.add_argument("--compare",   action="store_true",
                        help="Also run original time-stepped SOT sim for side-by-side comparison")
    args = parser.parse_args()

    pool_B = args.workers_b
    pool_C = args.workers_c

    load_kwargs = {"csv_path": args.csv} if args.csv else {}
    jobs = load_jobs(**load_kwargs)
    ests = [j.estimated_h for j in jobs]
    n    = len(jobs)

    print(f"Loaded {n} jobs from data_real.csv")
    print(f"  min/avg/max estimated : "
          f"{min(ests):.1f} / {sum(ests)/n:.1f} / {max(ests):.1f} h")
    print(f"  B workers: {pool_B}  |  C workers: {pool_C}")
    print(f"  Lookahead : {BLADE_QUEUE_LOOKAHEAD}")

    use_delays = not args.no_delays
    print(f"\nRunning DES SOT baseline simulation ...")
    print(f"  Monte Carlo delays : {'ON' if use_delays else 'OFF'}  "
          f"(seed={args.delay_seed})")
    staff_log, sc1, sc2, sc3 = simulate_des_sot(
        jobs, pool_B=pool_B, pool_C=pool_C,
        use_delays=use_delays, delay_seed=args.delay_seed,
    )

    done = [j for j in jobs if j.oven_end is not None]
    print(f"\nCompleted : {len(done)} / {n} jobs")

    if done:
        makespan = max(j.oven_end for j in done)
        avg_B    = sum(j.avg_workers_B for j in done) / len(done)
        avg_C    = sum(j.avg_workers_C for j in done) / len(done)
        fitness  = makespan + SC2_WEIGHT * sc2 + SC3_WEIGHT * sc3
        print(f"\nMakespan          : {makespan:.1f} h")
        print(f"Avg B workers/job : {avg_B:.2f}")
        print(f"Avg C workers/job : {avg_C:.2f}")
        print(f"SC-1 penalty      : {sc1:.1f}")
        print(f"SC-2 penalty      : {sc2:.1f}")
        print(f"SC-3 penalty      : {sc3:.1f}")
        print(f"Total fitness     : {fitness:.1f}")

        ratios = [(j.repair_end - j.repair_start) / j.estimated_h
                  for j in done if j.repair_end is not None]
        if ratios:
            violations = sum(1 for r in ratios if r < MIN_PROCESS_TIME_RATIO - 1e-6)
            print(f"\nHC-2 check : min ratio = {min(ratios):.4f}  "
                  f"(limit >= {MIN_PROCESS_TIME_RATIO:.2f})  violations = {violations}")

    if args.compare:
        from sot_simulationBC_baseline import simulate_sot_bc
        print("\nRunning original time-stepped SOT sim for comparison ...")
        from sot_simulationBC_baseline import load_jobs as _load
        jobs_ts = _load(**({"csv_path": args.csv} if args.csv else {}))
        sl_ts, sc1_ts, sc2_ts, sc3_ts = simulate_sot_bc(jobs_ts, pool_B=pool_B, pool_C=pool_C)
        done_ts = [j for j in jobs_ts if j.oven_end is not None]
        if done_ts:
            makespan_ts = max(j.oven_end for j in done_ts)
            fitness_ts  = makespan_ts + SC2_WEIGHT * sc2_ts + SC3_WEIGHT * sc3_ts
            print(f"\n=== Original SOT (time-stepped) results ===")
            print(f"Makespan  : {makespan_ts:.1f} h")
            print(f"Fitness   : {fitness_ts:.1f}")
            if done:
                print(f"\nDES improvement : "
                      f"{fitness_ts - fitness:+.1f}  "
                      f"({'better' if fitness < fitness_ts else 'worse or equal'})")

    if done:
        _export_csv(
            jobs, makespan, sc1, sc2, sc3,
            pool_B=pool_B, pool_C=pool_C,
            use_delays=use_delays,
            save_path=f"des_sot_simulation_results_{n}jobs.csv",
        )
        plot_weekly_throughput(
            jobs, makespan,
            save_path=f"des_sot_weekly_throughput_{n}jobs.png",
        )

    plot_gantt_des(jobs, staff_log, pool_B=pool_B, pool_C=pool_C,
                   save_path=f"gantt_des_sot_bc_baseline_{n}jobs.png")
