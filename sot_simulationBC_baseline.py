"""
sot_simulationBC_baseline.py
============================
SOT (Shortest Operating Time) blade repair simulation â€” baseline variant.

The first N_JOBS jobs from the *Estimated* column of data_real.csv are used.
No synthetic job generation, no PERT distribution, no buffer queue.

data_real.csv columns used
---------------------------
  Estimated  â€” estimated repair hours for the blade

Dispatching
-----------
The first MAX_REPAIR_TRACKS jobs are pre-sorted by estimated_h (pure SOT)
and placed directly on tracks at t=0.  The remaining jobs follow in arrival
order.  When a repair track becomes free, only the first
BLADE_QUEUE_LOOKAHEAD (3) jobs in the queue are visible; the shortest of
those (SOT) is selected.

Worker allocation â€” SOT heuristic (every WORKER_REASSIGN_INTERVAL hours)
--------------------------------------------------------------------------
  1. Guarantee one B worker to every phase-2 job (shortest remaining first).
  2. Top up remaining B workers to phase-2 jobs, shortest remaining first.
  3. Guarantee one C worker to every phase-1/3 job (shortest remaining first).
  4. Top up remaining C workers to phase-1/3 jobs, shortest remaining first.

No GA, no chromosome, no SA â€” one deterministic simulation run.

Outputs
-------
  gantt_sot_bc_baseline.png  â€” four-panel Gantt chart
"""

from __future__ import annotations

import argparse
import csv
import os
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
# Top-level constants  â† adjust these to change the simulation setup
# ---------------------------------------------------------------------------

N_JOBS:      int   = 50   # total jobs loaded from data_real.csv
N_WORKERS_B: int   = 35   # B worker pool
N_WORKERS_C: int   = 25   # C worker pool

WEEK_HOURS:  float = MAX_WEEKLY_HOURS
PHASE1_H:    float = 9.0
PHASE3_H:    float = 27.0

DATA_REAL_CSV: str = os.path.join(os.path.dirname(__file__), "data_real.csv")

HC2_EFF_CAP: float = 1.0 / MIN_PROCESS_TIME_RATIO

SC1_WEIGHT: float = 0.1
SC2_WEIGHT: float = 10.0
SC3_WEIGHT: float = 10.0


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
    skip_phase1: bool = False   # True = first 12 jobs (no phase 1)

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


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_jobs(csv_path: str = DATA_REAL_CSV) -> list[Job]:
    """Load the first N_JOBS Estimated durations from data_real.csv.
    The first MAX_REPAIR_TRACKS jobs have skip_phase1=True (already on tracks).
    """
    jobs = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for i, row in enumerate(reader):
            if i >= N_JOBS:
                break
            estimated = float(row["Estimated"])
            jobs.append(Job(
                job_id      = f"JOB-{i+1:03d}",
                estimated_h = estimated,
                job_index   = i,
                skip_phase1 = i < MAX_REPAIR_TRACKS,
            ))
    return jobs


# ---------------------------------------------------------------------------
# Simulation engine â€” SOT dispatching and worker allocation
# ---------------------------------------------------------------------------

def simulate_sot_bc(
    jobs:   list[Job],
    pool_B: int = 0,
    pool_C: int = 0,
) -> tuple[list[tuple[float, int, int]], float, float, float]:
    """
    Run one SOT simulation over all jobs from data_real.csv.

    First 12 jobs pre-sorted SOT, then remaining 38 in arrival order.
    Each track fill selects the shortest job among the first
    BLADE_QUEUE_LOOKAHEAD (3) candidates visible in the queue.
    Worker allocation runs every WORKER_REASSIGN_INTERVAL hours.

    Returns (staff_log, sc1_total, sc2_total, sc3_total).
    """
    if pool_B <= 0: pool_B = N_WORKERS_B
    if pool_C <= 0: pool_C = N_WORKERS_C

    dt = WORKER_REASSIGN_INTERVAL

    slots:        list[Optional[_ActiveSlot]] = [None] * MAX_REPAIR_TRACKS
    slot_free_at: list[float]                 = [0.0]  * MAX_REPAIR_TRACKS
    oven_free_at: float = 0.0

    # Pre-fill all tracks with the first MAX_REPAIR_TRACKS jobs sorted SOT at t=0,
    # guaranteeing they all start on a track regardless of arrival-order durations.
    for i, job in enumerate(sorted(jobs[:MAX_REPAIR_TRACKS], key=lambda j: j.estimated_h)):
        aj = _ActiveSlot(job=job, track_idx=i,
                         p1_rem=job.phase1_h, p2_rem=job.phase2_h, p3_rem=job.phase3_h)
        job.repair_track = i
        job.repair_start = 0.0
        job.phase1_end   = 0.0  # skip_phase1=True for all initial track jobs
        slots[i] = aj

    # Remaining jobs in arrival order; lookahead-3 SOT applied at dispatch
    queue: list[Job] = list(jobs[MAX_REPAIR_TRACKS:])

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

        # Step 2 â€” fill free tracks (SOT within lookahead window)
        for i in range(MAX_REPAIR_TRACKS):
            if slots[i] is None and slot_free_at[i] <= t and queue:
                candidates = get_candidate_blades(queue, BLADE_QUEUE_LOOKAHEAD)
                job = min(candidates, key=lambda j: j.estimated_h)
                queue.remove(job)
                aj  = _ActiveSlot(job=job, track_idx=i,
                                  p1_rem=job.phase1_h, p2_rem=job.phase2_h, p3_rem=job.phase3_h)
                job.repair_track = i
                job.repair_start = t
                if job.skip_phase1:
                    job.phase1_end = t
                slots[i] = aj
        enforce_empty_track_fill(slots, slot_free_at, queue, t)

        # Step 3 â€” SOT worker allocation
        active   = [aj for aj in slots if aj is not None]
        b_active = sorted([aj for aj in active if aj.phase == 2],
                          key=lambda s: s.remaining_h)
        c_active = sorted([aj for aj in active if aj.phase in (1, 3)],
                          key=lambda s: s.remaining_h)

        for aj in b_active: aj.workers_B = 0
        for aj in c_active: aj.workers_C = 0

        # B workers: guarantee 1, then top-up SOT, then HC-11 full-utilisation
        pool_left_B = pool_B
        for aj in b_active:
            if pool_left_B <= 0: break
            aj.workers_B = 1; pool_left_B -= 1
        for aj in b_active:
            if pool_left_B <= 0: break
            give = min(MAX_WORKERS_PER_BLADE - aj.workers_B, pool_left_B)
            aj.workers_B += give; pool_left_B -= give
        if b_active:
            alloc_b = {aj.job.job_index: aj.workers_B for aj in b_active}
            alloc_b = enforce_full_worker_utilization_B(alloc_b, pool_B)
            for aj in b_active:
                aj.workers_B = alloc_b[aj.job.job_index]

        # C workers: guarantee 1, then top-up SOT, then HC-11 full-utilisation
        pool_left_C = pool_C
        for aj in c_active:
            if pool_left_C <= 0: break
            aj.workers_C = 1; pool_left_C -= 1
        for aj in c_active:
            if pool_left_C <= 0: break
            give = min(MAX_WORKERS_PER_BLADE - aj.workers_C, pool_left_C)
            aj.workers_C += give; pool_left_C -= give
        if c_active:
            alloc_c = {aj.job.job_index: aj.workers_C for aj in c_active}
            alloc_c = enforce_full_worker_utilization_C(alloc_c, pool_C)
            for aj in c_active:
                aj.workers_C = alloc_c[aj.job.job_index]

        # SC-1 penalty and worker log
        for aj in b_active:
            sc1_total += penalty_worker_band(aj.workers_B) * dt
            aj.job.worker_log.append((t, t + dt, aj.workers_B, 'B'))
        for aj in c_active:
            sc1_total += penalty_worker_band(aj.workers_C) * dt
            aj.job.worker_log.append((t, t + dt, aj.workers_C, 'C'))

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
                w = aj.workers_C
            else:
                w = aj.workers_B

            eff = efficiency_factor(w)
            if eff <= 0.0:
                continue

            eff       = min(eff, HC2_EFF_CAP)
            work_done = dt * eff

            if phase == 1:
                if work_done >= aj.p1_rem:
                    phase_end_t       = t + aj.p1_rem / eff
                    aj.job.phase1_end = phase_end_t
                    aj.p1_rem         = 0.0
                    if aj.p2_rem <= 1e-9 and aj.p3_rem <= 1e-9:
                        min_end           = aj.job.repair_start + MIN_PROCESS_TIME_RATIO * aj.job.estimated_h
                        aj.job.repair_end = max(phase_end_t, min_end)
                        newly_done.append(aj)
                else:
                    aj.p1_rem -= work_done

            elif phase == 2:
                if work_done >= aj.p2_rem:
                    phase_end_t       = t + aj.p2_rem / eff
                    aj.job.phase2_end = phase_end_t
                    aj.p2_rem         = 0.0
                    if aj.p3_rem <= 1e-9:
                        min_end           = aj.job.repair_start + MIN_PROCESS_TIME_RATIO * aj.job.estimated_h
                        aj.job.repair_end = max(phase_end_t, min_end)
                        newly_done.append(aj)
                else:
                    aj.p2_rem -= work_done

            else:
                if work_done >= aj.p3_rem:
                    raw_end           = t + aj.p3_rem / eff
                    min_end           = aj.job.repair_start + MIN_PROCESS_TIME_RATIO * aj.job.estimated_h
                    aj.job.repair_end = max(raw_end, min_end)
                    aj.p3_rem         = 0.0
                    newly_done.append(aj)
                else:
                    aj.p3_rem -= work_done

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

    done      = [j for j in jobs if j.oven_start is not None and j.repair_end is not None]
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
    save_path: str = "gantt_sot_bc_baseline.png",
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
        f"SOT (B/C workers) â€” baseline (data_real.csv)  "
        f"({len(done)} jobs completed from {len(jobs)} loaded)",
        fontsize=13, fontweight="bold",
    )

    # ---- Panel 1: Repair tracks ------------------------------------------
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
    ax4.set_title("B worker utilization  (SOT allocation)", fontsize=10)
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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SOT blade repair simulation using data_real.csv (B/C workers)"
    )
    parser.add_argument("--workers_b", type=int, default=N_WORKERS_B,
                        help=f"B worker pool (default: {N_WORKERS_B})")
    parser.add_argument("--workers_c", type=int, default=N_WORKERS_C,
                        help=f"C worker pool (default: {N_WORKERS_C})")
    parser.add_argument("--csv",       type=str, default=DATA_REAL_CSV,
                        help="Path to data_real.csv")
    args = parser.parse_args()

    jobs = load_jobs(args.csv)
    ests = [j.estimated_h for j in jobs]

    print(f"Loaded {len(jobs)} jobs (first {N_JOBS} rows) from {args.csv}")
    print(f"  min/avg/max estimated : "
          f"{min(ests):.1f} / {sum(ests)/len(ests):.1f} / {max(ests):.1f} h")
    print(f"  B workers: {args.workers_b}  |  C workers: {args.workers_c}")
    print(f"  Lookahead : {BLADE_QUEUE_LOOKAHEAD}")
    print(f"\nRunning SOT baseline simulation ...")

    staff_log, sc1, sc2, sc3 = simulate_sot_bc(
        jobs,
        pool_B = args.workers_b,
        pool_C = args.workers_c,
    )

    done = [j for j in jobs if j.oven_end is not None]

    print(f"\nCompleted : {len(done)} / {len(jobs)} jobs")

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
        total_fit = makespan + SC2_WEIGHT * sc2 + SC3_WEIGHT * sc3
        print(f"Total fitness     : {total_fit:.1f}")

        ratios = [(j.repair_end - j.repair_start) / j.estimated_h
                  for j in done if j.repair_end is not None]
        if ratios:
            violations = sum(1 for r in ratios if r < MIN_PROCESS_TIME_RATIO - 1e-6)
            print(f"\nHC-2 check : min ratio = {min(ratios):.4f}  "
                  f"(limit >= {MIN_PROCESS_TIME_RATIO:.2f})  violations = {violations}")

    plot_gantt(jobs, staff_log,
               pool_B=args.workers_b, pool_C=args.workers_c,
               save_path="gantt_sot_bc_baseline.png")

