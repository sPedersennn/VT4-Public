"""
des_simulationBC_sot_current.py
================================
Discrete-Event Simulation (DES) of blade repair using SOT (Shortest
Operating Time) dispatching â€” the DES counterpart of sot_simulationBC_current.py.

Combines the event-driven engine from des_simulationBCtester_current.py
with the SOT worker allocation from sot_simulationBC_current.py, plus
Monte Carlo PERT delays (from the historical Actual-Estimated distribution
in data_real.csv).

Dispatching
-----------
Primary jobs (buffer=0) fill repair tracks first, sorted SOT (shortest
estimated_h first).  Buffer jobs (buffer=1) follow, also sorted SOT.
Each time a buffer job is dispatched a new PERT-sampled job is inserted
into the buffer queue in SOT order (up to MAX_GENERATED_JOBS total).

Worker allocation (every WORKER_REASSIGN_INTERVAL hours)
---------------------------------------------------------
  1. Guarantee one B worker to every phase-2 job (shortest remaining first).
  2. Top up remaining B workers to phase-2 jobs, shortest remaining first.
  3. HC-11: redistribute so the full pool is used when capacity allows.
  4-6. Same for C workers on phase-1/3 jobs.

Monte Carlo delays
------------------
Same as des_simulationBCtester_current.py: a PERT delay sampled from
data_real.csv (Actual-Estimated) is applied to each blade in the last epoch
before its phase-2 completes.

The first 12 primary jobs skip phase 1 (handled by load_jobs via
sot_simulationBC_current).

Outputs
-------
  gantt_des_sot_bc_current_Njobs.png   â€” four-panel Gantt chart
  weekly_throughput_sot_Njobs.png      â€” blade throughput per week
  des_sot_simulation_results_Njobs.csv â€” KPI row appended per run
"""
from __future__ import annotations

import argparse
import csv
import heapq
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as _scipy_beta

from sot_simulationBC_current import (
    Job,
    _ActiveSlot,
    load_jobs,
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
    WEEK_HOURS,
    PHASE1_H,
    PHASE3_H,
)
from constraintsBC import (
    CHANGEOVER_TIME,
    MAX_REPAIR_TRACKS,
    MAX_WORKERS_PER_BLADE,
    OVEN_PROCESS_TIME,
    WORKER_REASSIGN_INTERVAL,
    MIN_PROCESS_TIME_RATIO,
    efficiency_factor,
    penalty_worker_band,
    penalty_oven_idle,
    enforce_full_worker_utilization_B,
    enforce_full_worker_utilization_C,
)

HC2_EFF_CAP: float = 1.0 / MIN_PROCESS_TIME_RATIO

# ---------------------------------------------------------------------------
# Event priorities (lower = fires first at equal timestamp)
# ---------------------------------------------------------------------------
_P_PHASE_DONE = 0
_P_EPOCH      = 1
_P_OVEN_FREE  = 2
_P_TRACK_FREE = 3

_MAX_T     = 15_000.0
DELAY_SEED = 7


# ---------------------------------------------------------------------------
# Monte Carlo delay  (Modified PERT fitted to data_real.csv Actual-Estimated)
# ---------------------------------------------------------------------------

_PERT_DELAY_CACHE: Optional[tuple[float, float, float, float]] = None


def _load_delay_pert_params() -> tuple[float, float, float, float]:
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


def _get_delay_pert_params() -> tuple[float, float, float, float]:
    global _PERT_DELAY_CACHE
    if _PERT_DELAY_CACHE is None:
        _PERT_DELAY_CACHE = _load_delay_pert_params()
    return _PERT_DELAY_CACHE


def _sample_pert_delay(rng: random.Random) -> float:
    a, mode, b, gamma = _get_delay_pert_params()
    alpha1 = 1.0 + gamma * (mode - a) / (b - a)
    alpha2 = 1.0 + gamma * (b - mode) / (b - a)
    z = float(_scipy_beta.ppf(rng.random(), alpha1, alpha2))
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
    slots:           list
    slot_free_at:    list[float]
    oven_free_at:    float
    queue:           list[Job]
    pending_oven:    list
    oven_seq:        int
    track_version:   list[int]
    alloc_since:     list[float]
    heap:            list
    staff_log:       list[tuple[float, int, int]]
    sc1_total:       float
    oven_starts:     list[float]
    oven_ends:       list[float]
    pool_B:          int
    pool_C:          int
    use_delays:      bool
    delay_rng:       Optional[random.Random]
    delay_applied:   set
    jobs_list:       list
    generated_count: int
    gen_rng:         random.Random


def _push(state: _State, ev: _Ev) -> None:
    heapq.heappush(state.heap, ev)


# ---------------------------------------------------------------------------
# Work-advancement helper
# ---------------------------------------------------------------------------

def _advance_work(slot: _ActiveSlot, from_t: float, to_t: float) -> None:
    dt = to_t - from_t
    if dt <= 1e-12:
        return
    phase = slot.phase
    w   = slot.workers_C if phase in (1, 3) else slot.workers_B
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
    slot = state.slots[i]
    if slot is None or slot.remaining_h <= 1e-9:
        return
    state.track_version[i] += 1
    phase = slot.phase
    w   = slot.workers_C if phase in (1, 3) else slot.workers_B
    eff = min(efficiency_factor(w), HC2_EFF_CAP)
    if eff <= 0:
        return
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


# ---------------------------------------------------------------------------
# Track filling
# ---------------------------------------------------------------------------

def _try_fill_tracks(state: _State, t: float) -> None:
    for i in range(MAX_REPAIR_TRACKS):
        if state.slots[i] is not None or state.slot_free_at[i] > t or not state.queue:
            continue
        job = state.queue.pop(0)

        if job.is_buffered and state.generated_count < MAX_GENERATED_JOBS:
            new_h   = pert_sample(state.gen_rng, _PERT_A, _PERT_MODE, _PERT_B)
            new_job = Job(
                job_id      = f"GEN-{state.generated_count + 1:03d}",
                estimated_h = new_h,
                job_index   = len(state.jobs_list),
                is_buffered = True,
            )
            state.generated_count += 1
            # Insert in SOT position within the remaining buffer tail
            insert_at = len(state.queue)
            for k, q in enumerate(state.queue):
                if q.is_buffered and new_h < q.estimated_h:
                    insert_at = k
                    break
            state.queue.insert(insert_at, new_job)
            state.jobs_list.append(new_job)

        aj = _ActiveSlot(job=job, track_idx=i,
                         p1_rem=job.phase1_h, p2_rem=job.phase2_h, p3_rem=job.phase3_h)
        job.repair_track     = i
        job.repair_start     = t
        if job.skip_phase1:
            job.phase1_end = t
        state.slots[i]       = aj
        state.alloc_since[i] = t


# ---------------------------------------------------------------------------
# EPOCH handler
# ---------------------------------------------------------------------------

def _on_epoch(state: _State, t: float) -> bool:
    dt = WORKER_REASSIGN_INTERVAL

    # 1. Transfer staged blades to oven
    _try_oven_intake(state, t)

    # 2. Fill empty tracks from queue
    _try_fill_tracks(state, t)

    # 3. Advance remaining work for every active slot
    for i, slot in enumerate(state.slots):
        if slot is not None and slot.remaining_h > 1e-9:
            _advance_work(slot, state.alloc_since[i], t)
            state.alloc_since[i] = t

    # 4. SOT worker allocation
    active = [s for s in state.slots if s is not None and s.remaining_h > 1e-9]
    for s in active:
        s.workers_B = 0
        s.workers_C = 0

    b_jobs = sorted([s for s in active if s.phase == 2],      key=lambda s: s.remaining_h)
    c_jobs = sorted([s for s in active if s.phase in (1, 3)], key=lambda s: s.remaining_h)

    # B workers: guarantee 1, top-up SOT, HC-11 full-utilisation
    pool_left_B = state.pool_B
    for s in b_jobs:
        if pool_left_B <= 0: break
        s.workers_B = 1; pool_left_B -= 1
    for s in b_jobs:
        if pool_left_B <= 0: break
        give = min(MAX_WORKERS_PER_BLADE - s.workers_B, pool_left_B)
        s.workers_B += give; pool_left_B -= give
    if b_jobs:
        alloc_b = {s.job.job_index: s.workers_B for s in b_jobs}
        alloc_b = enforce_full_worker_utilization_B(alloc_b, state.pool_B)
        for s in b_jobs:
            s.workers_B = alloc_b[s.job.job_index]

    # C workers: guarantee 1, top-up SOT, HC-11 full-utilisation
    pool_left_C = state.pool_C
    for s in c_jobs:
        if pool_left_C <= 0: break
        s.workers_C = 1; pool_left_C -= 1
    for s in c_jobs:
        if pool_left_C <= 0: break
        give = min(MAX_WORKERS_PER_BLADE - s.workers_C, pool_left_C)
        s.workers_C += give; pool_left_C -= give
    if c_jobs:
        alloc_c = {s.job.job_index: s.workers_C for s in c_jobs}
        alloc_c = enforce_full_worker_utilization_C(alloc_c, state.pool_C)
        for s in c_jobs:
            s.workers_C = alloc_c[s.job.job_index]

    # SC-1 penalty and worker log
    for s in b_jobs:
        state.sc1_total += penalty_worker_band(s.workers_B) * dt
        s.job.worker_log.append((t, t + dt, s.workers_B, 'B'))
    for s in c_jobs:
        state.sc1_total += penalty_worker_band(s.workers_C) * dt
        s.job.worker_log.append((t, t + dt, s.workers_C, 'C'))

    # 4b. Monte Carlo delay: apply in the last epoch before phase 2 completes
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
                continue
            state.delay_applied.add(slot.job.job_index)
            d            = _sample_pert_delay(state.delay_rng)
            old_p2       = slot.p2_rem
            p2_initial   = slot.job.phase2_h
            p2_done      = max(0.0, p2_initial - old_p2)
            min_p2_total = max(0.0,
                               MIN_PROCESS_TIME_RATIO * slot.job.estimated_h
                               - slot.job.phase1_h - slot.job.phase3_h)
            floor_p2          = max(0.0, min_p2_total - p2_done)
            slot.p2_rem       = max(old_p2 + d, floor_p2)
            slot.job.mc_delay = slot.p2_rem - old_p2
            if slot.p2_rem <= 1e-9:
                slot.p2_rem         = 0.0
                slot.job.phase2_end = t

    # 5. Schedule PHASE_DONE for each active track
    for i in range(MAX_REPAIR_TRACKS):
        if state.slots[i] is not None and state.slots[i].remaining_h > 1e-9:
            _schedule_phase_done(state, i, t)

    # 6. Staff log
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
    if ver != state.track_version[track]:
        return
    slot = state.slots[track]
    if slot is None or slot.remaining_h <= 1e-9:
        return
    job = slot.job

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
    else:
        slot.p3_rem    = 0.0
        min_end        = job.repair_start + MIN_PROCESS_TIME_RATIO * job.estimated_h
        job.repair_end = max(t, min_end)

    if slot.remaining_h <= 1e-9 and job.repair_end is not None:
        state.slots[track]        = None
        state.slot_free_at[track] = job.repair_end + CHANGEOVER_TIME
        _push(state, _Ev(state.slot_free_at[track], _P_TRACK_FREE, {'track': track}))
        state.pending_oven.append((job.repair_end, state.oven_seq, job))
        state.oven_seq += 1
        _try_oven_intake(state, t)
    else:
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
# Public simulation entry point
# ---------------------------------------------------------------------------

def simulate_des_sot_bc(
    jobs:       list[Job],
    pool_B:     int  = 0,
    pool_C:     int  = 0,
    use_delays: bool = True,
    delay_seed: int  = DELAY_SEED,
    gen_seed:   int  = 100,
) -> tuple[list[tuple[float, int, int]], float, float, float]:
    """
    DES-SOT simulation over all jobs.

    Primary jobs dispatched SOT, then buffer jobs SOT.  Each dispatched
    buffer job spawns a new PERT-sampled arrival inserted in SOT order.
    Workers reallocated every WORKER_REASSIGN_INTERVAL hours using the
    SOT heuristic.  Optional Monte Carlo PERT delays on phase-2.

    Returns (staff_log, sc1_total, sc2_total, sc3_total).
    """
    if pool_B <= 0: pool_B = N_WORKERS_B
    if pool_C <= 0: pool_C = N_WORKERS_C

    primary_q = sorted([j for j in jobs if not j.is_buffered], key=lambda j: j.estimated_h)
    buffer_q  = sorted([j for j in jobs if j.is_buffered],     key=lambda j: j.estimated_h)

    state = _State(
        slots           = [None] * MAX_REPAIR_TRACKS,
        slot_free_at    = [0.0]  * MAX_REPAIR_TRACKS,
        oven_free_at    = 0.0,
        queue           = primary_q + buffer_q,
        pending_oven    = [],
        oven_seq        = 0,
        track_version   = [0] * MAX_REPAIR_TRACKS,
        alloc_since     = [0.0] * MAX_REPAIR_TRACKS,
        heap            = [],
        staff_log       = [],
        sc1_total       = 0.0,
        oven_starts     = [],
        oven_ends       = [],
        pool_B          = pool_B,
        pool_C          = pool_C,
        use_delays      = use_delays,
        delay_rng       = random.Random(delay_seed) if use_delays else None,
        delay_applied   = set(),
        jobs_list       = jobs,
        generated_count = 0,
        gen_rng         = random.Random(gen_seed),
    )

    _push(state, _Ev(0.0, _P_EPOCH))

    while state.heap:
        ev = heapq.heappop(state.heap)
        t  = ev.time
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

    # Safety drain
    for rep_end, _, job in sorted(state.pending_oven):
        ov_start           = max(state.oven_free_at, rep_end)
        job.oven_start     = ov_start
        job.oven_end       = ov_start + OVEN_PROCESS_TIME
        state.oven_free_at = job.oven_end + CHANGEOVER_TIME
        state.oven_starts.append(ov_start)
        state.oven_ends.append(job.oven_end)

    done      = [j for j in jobs if j.oven_start is not None and j.repair_end is not None]
    sc2_total = sum(max(0.0, j.oven_start - j.repair_end) for j in done)

    if len(state.oven_starts) > 1:
        pairs    = sorted(zip(state.oven_starts, state.oven_ends))
        s_sorted = [p[0] for p in pairs]
        e_sorted = [p[1] for p in pairs]
        sc3_total = penalty_oven_idle(e_sorted, s_sorted)
    else:
        sc3_total = 0.0

    return state.staff_log, state.sc1_total, sc2_total, sc3_total


# ---------------------------------------------------------------------------
# Gantt chart
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
    save_path: str = "gantt_des_sot_bc_current.png",
) -> None:
    done = [j for j in jobs if j.oven_end is not None]
    if not done:
        print("No finished jobs â€” nothing to plot.")
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
        f"DES-SOT (B/C workers + MC delays)  â€”  "
        f"{n_primary} primary + {n_buf} buffer + {n_gen} PERT-generated = {len(done)} completed",
        fontsize=13, fontweight="bold",
    )

    # ---- Panel 1: Repair tracks ------------------------------------------
    ax1 = axes[0]
    ax1.set_title(
        "Repair tracks  "
        "(solid = Phase 1/3: C workers | hatched = Phase 2: B workers | "
        "Î” label = MC delay | BUF-xxx = navy border, GEN-xxx = darkgreen border)",
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
                     height=bar_h, color=col, edgecolor=edge_col, linewidth=edge_lw)

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

        p3_s = p2_e if (p2_e is not None and j.phase2_h > 0) else p1_e
        p3_e = j.repair_end
        if p3_s is not None and p3_e is not None and p3_e > p3_s and j.phase3_h > 0:
            ax1.barh(y, p3_e - p3_s, left=p3_s,
                     height=bar_h, color=col, alpha=0.70,
                     edgecolor=edge_col, linewidth=edge_lw)

        delay = getattr(j, "mc_delay", None)
        if delay is not None and j.repair_start is not None and j.repair_end is not None:
            cx   = j.repair_start + (j.repair_end - j.repair_start) / 2
            dcol = "#888888" if abs(delay) < 0.05 else ("#cc2200" if delay > 0 else "#007722")
            ax1.text(cx, y + bar_h / 2 + 0.04, f"Î”{delay:+.1f}h",
                     ha="center", va="bottom", fontsize=5,
                     color=dcol, clip_on=True, zorder=5)

    p1_patch  = mpatches.Patch(color="grey",             label=f"Phase 1: C workers ({PHASE1_H:.0f} h)")
    p2_patch  = mpatches.Patch(color="grey", alpha=0.45, hatch="///", label="Phase 2: B workers (middle)")
    p3_patch  = mpatches.Patch(color="grey", alpha=0.70, label=f"Phase 3: C workers ({PHASE3_H:.0f} h)")
    buf_patch = mpatches.Patch(facecolor="white", edgecolor="navy",      linewidth=1.2, label="Buffer job (BUF-xxx)")
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
        "â€” navy border = buffer job, darkgreen border = generated job",
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
    print(f"  Gantt chart (DES-SOT) saved -> {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Weekly throughput chart
# ---------------------------------------------------------------------------

def plot_weekly_throughput(
    jobs:      list[Job],
    makespan:  float,
    save_path: str = "weekly_throughput_sot.png",
) -> None:
    done = [j for j in jobs if j.oven_end is not None]
    if not done:
        print("No finished jobs â€” nothing to plot.")
        return

    n_weeks     = int(np.ceil(makespan / WEEK_HOURS))
    counts      = np.zeros(n_weeks, dtype=int)
    for j in done:
        wk = min(int(j.oven_end / WEEK_HOURS), n_weeks - 1)
        counts[wk] += 1
    week_labels = [f"Wk {i + 1}" for i in range(n_weeks)]
    cumulative  = np.cumsum(counts)
    avg         = len(done) / n_weeks

    fig, ax1 = plt.subplots(figsize=(max(8, n_weeks * 0.9), 5))
    bars = ax1.bar(week_labels, counts, color="#5BA4CF", edgecolor="white",
                   linewidth=0.6, zorder=3)
    ax1.axhline(avg, color="#E05C2A", linestyle="--", linewidth=1.4,
                label=f"Average  {avg:.1f} blades/week", zorder=4)
    for bar, val in zip(bars, counts):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.15, str(val),
                     ha="center", va="bottom", fontsize=8)
    ax1.set_xlabel("Simulation week  (1 week = 144 h)")
    ax1.set_ylabel("Blades completed", color="#5BA4CF")
    ax1.tick_params(axis="y", labelcolor="#5BA4CF")
    ax1.set_ylim(0, max(counts) * 1.25)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.legend(loc="upper left", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(week_labels, cumulative, color="#2E7D32", marker="o",
             linewidth=2.0, markersize=5, label="Cumulative total", zorder=5)
    ax2.set_ylabel("Cumulative blades completed", color="#2E7D32")
    ax2.tick_params(axis="y", labelcolor="#2E7D32")
    ax2.set_ylim(0, len(done) * 1.15)
    ax2.legend(loc="upper right", fontsize=9)
    ax1.set_title(
        f"Weekly throughput (DES-SOT) â€” {len(done)} blades  |  makespan {makespan:.0f} h  "
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
    save_path:  str = "des_sot_simulation_results.csv",
) -> None:
    done = [j for j in jobs if j.oven_end is not None and j.repair_end is not None]
    repair_times     = [j.repair_end - j.repair_start for j in done if j.repair_start is not None]
    avg_repair_time  = sum(repair_times) / len(repair_times) if repair_times else 0.0
    track_blocking   = CHANGEOVER_TIME * len(done)
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
        description="DES-SOT blade repair simulation (B/C workers, current_status.csv)"
    )
    parser.add_argument("--workers_b",  type=int,  default=N_WORKERS_B,
                        help=f"B worker pool (default: {N_WORKERS_B})")
    parser.add_argument("--workers_c",  type=int,  default=N_WORKERS_C,
                        help=f"C worker pool (default: {N_WORKERS_C})")
    parser.add_argument("--no_delays",  action="store_true",
                        help="Disable Monte Carlo PERT delays (deterministic run)")
    parser.add_argument("--delay_seed", type=int,  default=DELAY_SEED,
                        help=f"RNG seed for delay sampling (default: {DELAY_SEED})")
    parser.add_argument("--gen_seed",   type=int,  default=100,
                        help="RNG seed for PERT buffer-replenishment (default: 100)")
    parser.add_argument("--csv",        type=str,  default=None,
                        help="Path to current_status.csv (default: auto-detect)")
    args = parser.parse_args()

    pool_B = args.workers_b
    pool_C = args.workers_c

    jobs      = load_jobs() if args.csv is None else load_jobs(args.csv)
    n         = len(jobs)
    n_primary = sum(1 for j in jobs if not j.is_buffered)
    n_buf     = sum(1 for j in jobs if j.is_buffered)
    ests      = [j.estimated_h for j in jobs]

    print(f"Loaded {n} jobs from current_status.csv")
    print(f"  Primary jobs (buffer=0) : {n_primary}  "
          f"(first 12 skip phase 1)")
    print(f"  Buffer  jobs (buffer=1) : {n_buf}")
    print(f"  min/avg/max est : {min(ests):.1f} / {sum(ests)/n:.1f} / {max(ests):.1f} h")
    print(f"  B workers: {pool_B}  |  C workers: {pool_C}")

    use_delays = not args.no_delays
    print(f"\nRunning DES-SOT simulation  "
          f"(MC delays: {'ON' if use_delays else 'OFF'}, seed={args.delay_seed}) ...")

    staff_log, sc1, sc2, sc3 = simulate_des_sot_bc(
        jobs, pool_B=pool_B, pool_C=pool_C,
        use_delays=use_delays, delay_seed=args.delay_seed, gen_seed=args.gen_seed,
    )

    done       = [j for j in jobs if j.oven_end is not None]
    n_gen_done = sum(1 for j in done if j.job_id.startswith("GEN-"))

    print(f"\nCompleted  : {len(done)} jobs total")
    print(f"  Primary  : {sum(1 for j in done if not j.is_buffered)}")
    print(f"  Buffer   : {sum(1 for j in done if j.is_buffered and not j.job_id.startswith('GEN-'))}")
    print(f"  PERT-gen : {n_gen_done}  (max allowed: {MAX_GENERATED_JOBS})")

    if done:
        makespan = max(j.oven_end for j in done)
        avg_B    = sum(j.avg_workers_B for j in done) / len(done)
        avg_C    = sum(j.avg_workers_C for j in done) / len(done)
        fitness  = makespan + SC1_WEIGHT * sc1 + SC2_WEIGHT * sc2 + SC3_WEIGHT * sc3
        print(f"\nMakespan          : {makespan:.1f} h")
        print(f"Avg B workers/job : {avg_B:.2f}")
        print(f"Avg C workers/job : {avg_C:.2f}")
        print(f"SC-1 penalty      : {sc1:.1f}")
        print(f"SC-2 penalty      : {sc2:.1f}")
        print(f"SC-3 penalty      : {sc3:.1f}")
        print(f"Total fitness     : {fitness:.1f}")

        ratios     = [(j.repair_end - j.repair_start) / j.estimated_h
                      for j in done if j.repair_end is not None]
        violations = sum(1 for r in ratios if r < MIN_PROCESS_TIME_RATIO - 1e-6)
        print(f"HC-2 check        : min ratio = {min(ratios):.4f}  "
              f"violations = {violations}")

        _export_csv(
            jobs, makespan, sc1, sc2, sc3,
            pool_B=pool_B, pool_C=pool_C,
            use_delays=use_delays,
            save_path=f"des_sot_simulation_results_{n}jobs.csv",
        )
        plot_weekly_throughput(
            jobs, makespan,
            save_path=f"weekly_throughput_sot_{n}jobs.png",
        )

    plot_gantt_des(jobs, staff_log, pool_B=pool_B, pool_C=pool_C,
                   save_path=f"gantt_des_sot_bc_current_{n}jobs.png")


