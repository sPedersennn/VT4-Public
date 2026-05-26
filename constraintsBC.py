"""
constraintsBC.py
================
Hard and soft constraints for the blade repair scheduling simulation.
Identical to constraints.py except workers are split into two groups:
  B workers and C workers, each governed by their own pool limits.

Hard constraints are enforced by raising ValueError or by clipping/clamping
values so the simulation never operates in an infeasible state.

Soft constraints return a numeric penalty >= 0 that is added to the
optimisation objective.
"""

# ---------------------------------------------------------------------------
# HARD CONSTRAINT PARAMETERS
# ---------------------------------------------------------------------------

MAX_REPAIR_TRACKS: int   = 12      # k <= 12 simultaneous repair tracks
MIN_PROCESS_TIME_RATIO: float = 0.85  # actual repair time >= 85% of estimated
WORKER_REASSIGN_INTERVAL: float = 4.0 # workers may only be reallocated every 4 h
OVEN_PROCESS_TIME: float = 8   # fixed oven (finalization) duration in hours

# --- Worker pool split (B and C groups) ---
MAX_WORKERS_TOTAL_B: int  = 40     # total B workers across all active tracks
MAX_WORKERS_TOTAL_C: int  = 27     # total C workers across all active tracks

MAX_WEEKLY_HOURS: float  = 144.0  # maximum hours one worker may work per week
CHANGEOVER_TIME: float   = 4.0    # minimum idle time after any track or oven job
BLADE_QUEUE_LOOKAHEAD: int = 3    # only the first 3 blades in the queue may be
                                   # selected as the next job for a free track
MAX_WORKERS_PER_BLADE: int = 9    # w_j in {0, 1, ..., 9}; 0 means job is paused
                                   # (applies independently to B and C workers)

# ---------------------------------------------------------------------------
# SOFT CONSTRAINT PARAMETERS
# ---------------------------------------------------------------------------

OPTIMAL_WORKERS_LOW: int  = 4     # penalty if workers per blade < 4
OPTIMAL_WORKERS_HIGH: int = 6     # penalty if workers per blade > 6


# ---------------------------------------------------------------------------
# EFFICIENCY LOOKUP TABLE
# ---------------------------------------------------------------------------
# Maps number of workers -> work-hours completed per clock-hour.
# 0 workers = job is paused (no progress).
# The curve stagnates: each extra worker above 5 gives diminishing returns.

_EFFICIENCY_TABLE: dict[int, float] = {
    0: 0.00,
    1: 0.20,
    2: 0.45,
    3: 0.65,
    4: 0.90,
    5: 1.00,
    6: 1.10,
    7: 1.15,
    8: 1.20,
    9: 1.25,
}


def efficiency_factor(workers: int) -> float:
    """
    Returns the work-hours completed per clock-hour for a given worker count.
    Input is clamped to [0, MAX_WORKERS_PER_BLADE] before lookup.
    """
    w = max(0, min(MAX_WORKERS_PER_BLADE, workers))
    return _EFFICIENCY_TABLE[w]


def effective_repair_hours(estimated_h: float, workers: int) -> float:
    """
    Wall-clock hours needed to complete a blade repair given worker count.
    HC-2 (85% floor) is applied: the result is capped at estimated_h so that
    the minimum repair time never drops below MIN_PROCESS_TIME_RATIO * estimated_h.

    Returns float('inf') when workers == 0 (job is paused).
    """
    eff = efficiency_factor(workers)
    if eff <= 0.0:
        return float('inf')
    raw = estimated_h / eff
    min_time = MIN_PROCESS_TIME_RATIO * estimated_h
    return max(raw, min_time)


# ===========================================================================
# HARD CONSTRAINTS
# ===========================================================================

def check_track_count(n_tracks: int) -> None:
    """HC-1: no more than MAX_REPAIR_TRACKS blades processed simultaneously."""
    if n_tracks > MAX_REPAIR_TRACKS:
        raise ValueError(
            f"Track count {n_tracks} exceeds maximum of {MAX_REPAIR_TRACKS}."
        )


def enforce_min_process_time(estimated_h: float, actual_h: float) -> float:
    """
    HC-2: processing time cannot drop below 85% of the estimated duration.
    Returns the enforced actual processing time (>= 0.85 * estimated_h).
    """
    min_time = MIN_PROCESS_TIME_RATIO * estimated_h
    if actual_h < min_time:
        return min_time
    return actual_h


def is_reassignment_epoch(t: float, dt: float = WORKER_REASSIGN_INTERVAL) -> bool:
    """
    HC-3: returns True only at multiples of WORKER_REASSIGN_INTERVAL.
    Workers may be reallocated only when this returns True.
    """
    return abs(t % dt) < 1e-9 or abs(t % dt - dt) < 1e-9


def check_oven_time(duration: float) -> None:
    """
    HC-5: oven processing time is fixed and cannot be altered by adding workers.
    Raises if the caller attempts to use a different duration.
    """
    if abs(duration - OVEN_PROCESS_TIME) > 1e-9:
        raise ValueError(
            f"Oven duration must be exactly {OVEN_PROCESS_TIME} h "
            f"(got {duration:.2f} h). Adding workers does not reduce oven time."
        )


def enforce_worker_pool_B(allocations: dict[str, int]) -> dict[str, int]:
    """
    HC-6 (B workers): total B workers across all active blades cannot exceed
    MAX_WORKERS_TOTAL_B.  Scales allocations down proportionally if the pool
    is exceeded, then re-applies per-blade bounds so each value stays in
    [0, MAX_WORKERS_PER_BLADE].
    Returns the (possibly adjusted) allocation mapping.
    """
    allocations = {
        jid: max(0, min(MAX_WORKERS_PER_BLADE, w))
        for jid, w in allocations.items()
    }
    total = sum(allocations.values())
    if total > MAX_WORKERS_TOTAL_B:
        scale = MAX_WORKERS_TOTAL_B / total
        allocations = {
            jid: max(0, int(w * scale))
            for jid, w in allocations.items()
        }
    return allocations


def enforce_worker_pool_C(allocations: dict[str, int]) -> dict[str, int]:
    """
    HC-6 (C workers): total C workers across all active blades cannot exceed
    MAX_WORKERS_TOTAL_C.  Scales allocations down proportionally if the pool
    is exceeded, then re-applies per-blade bounds so each value stays in
    [0, MAX_WORKERS_PER_BLADE].
    Returns the (possibly adjusted) allocation mapping.
    """
    allocations = {
        jid: max(0, min(MAX_WORKERS_PER_BLADE, w))
        for jid, w in allocations.items()
    }
    total = sum(allocations.values())
    if total > MAX_WORKERS_TOTAL_C:
        scale = MAX_WORKERS_TOTAL_C / total
        allocations = {
            jid: max(0, int(w * scale))
            for jid, w in allocations.items()
        }
    return allocations


def check_weekly_hours(worker_hours: float) -> None:
    """HC-7: a single worker cannot exceed MAX_WEEKLY_HOURS hours per week."""
    if worker_hours > MAX_WEEKLY_HOURS:
        raise ValueError(
            f"Worker logged {worker_hours:.1f} h, exceeding the weekly "
            f"maximum of {MAX_WEEKLY_HOURS} h."
        )


def enforce_changeover(free_at: float, now: float) -> bool:
    """
    HC-8: a track or oven cannot start a new job until CHANGEOVER_TIME hours
    have elapsed since the previous job ended.
    Returns True when the slot is available (now >= free_at).
    """
    return now >= free_at


def enforce_empty_track_fill(slots: list, slot_free_at: list[float],
                              queue: list, t: float) -> None:
    """
    HC-12: any repair track that is empty and whose changeover period has
    elapsed must immediately receive the next available blade from the queue.
    Leaving a ready track idle while blades are waiting is forbidden.
    Raises ValueError if a free track is found while the queue is non-empty.
    Call this after the track-fill step to validate compliance.
    """
    if not queue:
        return
    violations = [
        i for i, slot in enumerate(slots)
        if slot is None and slot_free_at[i] <= t
    ]
    if violations:
        raise ValueError(
            f"HC-12 violation: tracks {violations} are empty and past "
            f"changeover but the queue still contains {len(queue)} blade(s)."
        )


def get_candidate_blades(queue: list, n: int = BLADE_QUEUE_LOOKAHEAD) -> list:
    """
    HC-9: only the first BLADE_QUEUE_LOOKAHEAD blades in the queue may be
    considered when selecting the next blade for a free repair track.
    """
    return queue[:n]


def enforce_workers_per_blade(w: int) -> int:
    """
    HC-10: clamps worker count for a single blade to [0, MAX_WORKERS_PER_BLADE].
    A value of 0 means the repair job is paused (no progress this epoch).
    Applies independently to B and C worker counts.
    """
    return max(0, min(MAX_WORKERS_PER_BLADE, w))


def enforce_full_worker_utilization_B(allocations: dict, pool: int) -> dict:
    """
    HC-11 (B workers): all B workers in the pool must be assigned every epoch.
    Any workers left unallocated after the initial assignment are distributed
    to active jobs (in the order they appear in the dict) until the pool is
    exhausted or every job has reached MAX_WORKERS_PER_BLADE.
    If the total active capacity (n_active * MAX_WORKERS_PER_BLADE) is smaller
    than the pool, every job is maxed out and the remainder is silently dropped
    (physically impossible to place those workers).
    Returns the augmented allocation mapping.
    """
    keys = list(allocations.keys())
    alloc = {k: max(0, min(MAX_WORKERS_PER_BLADE, allocations[k])) for k in keys}
    remaining = pool - sum(alloc.values())
    if remaining <= 0:
        return alloc
    idx = 0
    while remaining > 0 and idx < len(keys) * MAX_WORKERS_PER_BLADE:
        k = keys[idx % len(keys)]
        if alloc[k] < MAX_WORKERS_PER_BLADE:
            alloc[k] += 1
            remaining -= 1
        idx += 1
    return alloc


def enforce_full_worker_utilization_C(allocations: dict, pool: int) -> dict:
    """
    HC-11 (C workers): all C workers in the pool must be assigned every epoch.
    Any workers left unallocated after the initial assignment are distributed
    to active jobs (in the order they appear in the dict) until the pool is
    exhausted or every job has reached MAX_WORKERS_PER_BLADE.
    If the total active capacity (n_active * MAX_WORKERS_PER_BLADE) is smaller
    than the pool, every job is maxed out and the remainder is silently dropped
    (physically impossible to place those workers).
    Returns the augmented allocation mapping.
    """
    keys = list(allocations.keys())
    alloc = {k: max(0, min(MAX_WORKERS_PER_BLADE, allocations[k])) for k in keys}
    remaining = pool - sum(alloc.values())
    if remaining <= 0:
        return alloc
    idx = 0
    while remaining > 0 and idx < len(keys) * MAX_WORKERS_PER_BLADE:
        k = keys[idx % len(keys)]
        if alloc[k] < MAX_WORKERS_PER_BLADE:
            alloc[k] += 1
            remaining -= 1
        idx += 1
    return alloc


# ===========================================================================
# SOFT CONSTRAINTS  (each returns a non-negative penalty)
# ===========================================================================

def penalty_worker_band(w: int) -> float:
    """
    SC-1: penalise allocations outside the optimal band [4, 6] workers.
    Penalty equals the distance from the nearest boundary (0 inside the band).
    Applies independently to B and C worker counts.
    """
    if w < OPTIMAL_WORKERS_LOW:
        return float(OPTIMAL_WORKERS_LOW - w)
    if w > OPTIMAL_WORKERS_HIGH:
        return float(w - OPTIMAL_WORKERS_HIGH)
    return 0.0


def penalty_track_blocked(repair_end: float, oven_free_at: float,
                           now: float) -> float:
    """
    SC-2: penalise every hour a finished blade cannot be transferred to the
    oven (track blocked).  Returns the blocking duration up to the current
    epoch, or 0 if the oven was already free when the blade finished.
    """
    if repair_end is None:
        return 0.0
    blocking_start = max(repair_end, oven_free_at - OVEN_PROCESS_TIME - CHANGEOVER_TIME)
    blocking = max(0.0, now - blocking_start)
    return blocking


def penalty_oven_idle(oven_end_times: list[float],
                      oven_start_times: list[float]) -> float:
    """
    SC-3: penalise every hour the oven sits idle between consecutive jobs.
    Expects two parallel lists of oven job end and start times sorted
    chronologically.  Returns total idle hours (excluding the mandatory
    changeover, which is legitimate downtime).
    """
    idle = 0.0
    for i in range(len(oven_start_times) - 1):
        gap = oven_start_times[i + 1] - (oven_end_times[i] + CHANGEOVER_TIME)
        if gap > 0:
            idle += gap
    return idle


def total_soft_penalty(
    worker_allocations_B: list[int],
    worker_allocations_C: list[int],
    repair_end_times: list[float | None],
    oven_free_at: float,
    now: float,
    oven_end_times: list[float],
    oven_start_times: list[float],
) -> float:
    """
    Aggregate all soft-constraint penalties into a single scalar.
    SC-1 is evaluated separately for B and C worker allocations and summed.
    Weights are equal (1.0 per unit) and can be adjusted here.
    """
    SC1_WEIGHT = 0.001
    SC2_WEIGHT = 10.0
    SC3_WEIGHT = 10.0

    pen_band = SC1_WEIGHT * sum(penalty_worker_band(w) for w in worker_allocations_B)
    pen_block = SC2_WEIGHT * sum(
        penalty_track_blocked(re, oven_free_at, now)
        for re in repair_end_times
        if re is not None
    )
    pen_idle = SC3_WEIGHT * penalty_oven_idle(oven_end_times, oven_start_times)

    return pen_band + pen_block + pen_idle
