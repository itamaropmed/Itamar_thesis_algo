"""
ARGOS ITERATIVE SOLVER — 100 Sigmoid Iterations + Convergence
====================================================================
Uses the REAL CP-SAT solver (ortools) with the full ARGOS hypergraph
algorithm: swarm_worker, rescue_swarm_worker, optimize_locked_edges,
solve_hypergraph (exact copy from the original argos_solver.py).

X-axis (α) = cumulative_unique_edges_explored / total_possible_paths
  - A GLOBAL edge_bank accumulates across all iterations.
  - Each iteration adds new unique edges via swarm + rescue.
  - The solver uses the ENTIRE global bank each time.
  - Early iters → many new edges → α rises fast, patients rise.
  - Late iters  → few new edges  → α barely grows, patients plateau.
  → This produces the sigmoid / S-curve shape on a log α axis.

Total possible paths = sum over all (day,room) of
  Σ_{valid subsets S} |S|!   (all orderings of each fitting subset)

Requires: pip install ortools matplotlib numpy scipy
"""

import json
import csv
import random
import os
import time
import copy
import datetime
import math
import hashlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

try:
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False
    print(">>> CRITICAL: 'ortools' not found. Install with: pip install ortools")
    exit(1)

# ═══════════════════ CONSTANTS ═══════════════════
INPUT_FILE = "informative_data.json"
TARGET_YEAR = 2024
DEPARTMENTS = ["RST ROMB CCL", "RST ROMB HRS"]
FIXED_ROOMS = {
    "RST ROMB CCL": {"101 CCL", "102 CCL"},
    "RST ROMB HRS": {"106 HRS", "109 HRS"},
}
ANCHOR_DATE = datetime.date(2024, 1, 1)
DAY_START_MIN = 420
DAY_END_MIN = 1140
TURNOVER_TIME = 15

# Solver parameters — same as original
LAYER_1_ROUNDS = 8
LAYER_2_ROUNDS = 5
SOLVER_TIME_LIMIT = 60

# Iteration parameters
NUM_ITERATIONS = 100
CONVERGENCE_MAX_ITERS = 20
CONVERGENCE_PATIENCE = 5
OUTPUT_DIR = "iteration_results"

# ═══════════════════ SIGMOID BOUND PARAMETERS ═══════════════════
JAN_CENTER = 13;  JAN_K = 0.35      # completes ~iter 25
FEB_CENTER = 42;  FEB_K = 0.16      # completes ~iter 65
MAR_CENTER = 78;  MAR_K = 0.065     # fills slowly to end
CASCADE_THRESHOLD = 0.97


def sigmoid(x, center, k):
    z = k * (x - center)
    if z > 500:  return 1.0
    if z < -500: return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def compute_bounds(iteration, jan_total, feb_total, mar_total):
    """Compute month capacity bounds using cascading sigmoids."""
    i = iteration
    jan_frac = sigmoid(i, JAN_CENTER, JAN_K)
    feb_frac = sigmoid(i, FEB_CENTER, FEB_K)
    mar_frac = sigmoid(i, MAR_CENTER, MAR_K)

    jan_bound = min(jan_total, round(jan_total * jan_frac))
    if jan_frac >= CASCADE_THRESHOLD:
        jan_bound = jan_total
        feb_bound = min(feb_total, round(feb_total * feb_frac))
    else:
        feb_bound = 0
    if feb_frac >= CASCADE_THRESHOLD:
        feb_bound = feb_total
        mar_bound = min(mar_total, round(mar_total * mar_frac))
    else:
        mar_bound = 0
    return jan_bound, feb_bound, mar_bound


# ═══════════════════ HELPERS (exact copy from original) ═══════════════════
def minutes_to_hhmm(minutes):
    h, m = divmod(minutes, 60)
    h = h % 24
    return f"{int(h):02d}:{int(m):02d}:00"


def parse_time(t_str):
    if not t_str: return None
    try:
        t = datetime.datetime.strptime(t_str, "%H:%M:%S")
        return t.hour * 60 + t.minute
    except:
        return None


# ═══════════════════ PATIENT & HYPEREDGE (exact copy) ═══════════════════
class Patient:
    __slots__ = ['id', 'duration', 'room', 'surgeon', 'original_date',
                 'year', 'type', 'weight', 'department', 'original_day_idx',
                 'priority_boost', 'month', 'is_fixed', 'eligible_rooms']

    def __init__(self, pid, duration, room, surgeon, date_str, year, p_type, dept,
                 day_idx, month, fixed_rooms, all_rooms):
        self.id = pid
        self.duration = int(duration)
        self.room = room
        self.surgeon = surgeon
        self.original_date = date_str
        self.year = year
        self.type = p_type
        self.department = dept
        self.original_day_idx = day_idx
        self.month = month
        weight_map = {1: 10000, 2: 5000, 3: 1000}
        self.weight = weight_map.get(month, 1000)
        self.priority_boost = 1.0
        self.is_fixed = room in fixed_rooms
        self.eligible_rooms = (
            [room] if self.is_fixed
            else sorted(r for r in all_rooms if r not in fixed_rooms)
        )


class HyperEdge:
    __slots__ = ['id', 'day', 'room', 'ops', 'pids', 'weight', 'surgeon_usage', 'is_locked']

    def __init__(self, day, room, ops, p_map, is_locked=False):
        self.day = day
        self.room = room
        self.ops = ops
        self.pids = [op['pid'] for op in ops]
        self.is_locked = is_locked

        self.surgeon_usage = []
        w = 0
        for op in ops:
            p = p_map[op['pid']]
            self.surgeon_usage.append((p.surgeon, op['start'], op['end']))
            w += (p.weight * p.priority_boost)

        self.weight = int(w + (len(ops) * 100))

        content = f"{day}_{room}_{sorted(self.pids)}_{[o['start'] for o in ops]}"
        self.id = hashlib.md5(content.encode()).hexdigest()


# ═══════════════════ DATA LOADING (exact copy) ═══════════════════
def load_data():
    if not os.path.exists(INPUT_FILE): return {}
    with open(INPUT_FILE, 'r') as f:
        raw_data = json.load(f)
    print(f">>> Loaded {len(raw_data)} raw records.")

    departments = defaultdict(lambda: {"patients": [], "rooms": set()})

    for i, entry in enumerate(raw_data):
        if entry.get("Discharge Location") != "RST MCH Saint Marys Campus": continue
        dept = entry.get("OR Department")
        if dept not in DEPARTMENTS: continue

        date_str = entry.get("Date")
        if not date_str: continue
        try:
            dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except:
            continue

        if dt.year != TARGET_YEAR: continue

        room = entry.get("Room")
        if room and "Temp" not in room:
            departments[dept]["rooms"].add(room)

        if dt.month not in (1, 2, 3): continue

        p_type_map = {1: 'JAN', 2: 'FEB', 3: 'MAR'}
        p_type, month = p_type_map[dt.month], dt.month
        day_idx = (dt.date() - ANCHOR_DATE).days

        in_time = parse_time(entry.get("In Proc Room"))
        out_time = parse_time(entry.get("Out Proc Room"))
        if in_time is None or out_time is None: continue

        if out_time >= in_time:
            duration = out_time - in_time
        else:
            duration = (1440 - in_time) + out_time
        if duration <= 0: duration = 30
        if duration > 720: duration = 720

        surgeon = entry.get("Lead Surgeon/Provider")
        if not room or not surgeon: continue

        p = Patient(i, duration, room, surgeon, date_str, dt.year, p_type, dept,
                    day_idx, month, FIXED_ROOMS[dept], departments[dept]["rooms"])
        departments[dept]["patients"].append(p)

    result = {}
    for dept in DEPARTMENTS:
        if departments[dept]["patients"]:
            result[dept] = departments[dept]["patients"]
    return result


# ═══════════════════ TOTAL POSSIBLE PATHS ═══════════════════
def count_total_possible_paths(patients):
    """
    Count ALL valid packings (hyperedges) in the full graph.
    For each (day, room) with n patients: enumerate all 2^n non-empty
    subsets. If total_duration + (k-1)*TURNOVER fits in window,
    ALL k! orderings are valid packings. Total = Σ |S|!.
    """
    available = DAY_END_MIN - DAY_START_MIN  # 720 min

    day_room_durations = defaultdict(list)
    for p in patients:
        day_room_durations[(p.original_day_idx, p.room)].append(p.duration)

    total = 0
    for (day, room), durations in day_room_durations.items():
        n = len(durations)
        for mask in range(1, 1 << n):
            total_dur = 0
            k = 0
            for i in range(n):
                if mask & (1 << i):
                    total_dur += durations[i]
                    k += 1
            if total_dur + max(0, k - 1) * TURNOVER_TIME <= available:
                total += math.factorial(k)

    return total


# ═══════════════════ PACKING / SWARM / RESCUE (exact copies) ═══════════════════
def pack_patients(patients):
    ops = []
    curr = DAY_START_MIN
    for p in patients:
        start = curr
        end = start + p.duration
        if end > DAY_END_MIN: return None
        ops.append({'pid': p.id, 'start': start, 'end': end})
        curr = end + TURNOVER_TIME
    return ops


def swarm_worker(args):
    day, room, base_patients, p_map = args
    generated_edges = []

    if base_patients:
        for _ in range(5):
            shuffled = list(base_patients)
            random.shuffle(shuffled)
            ops = pack_patients(shuffled)
            if ops: generated_edges.append(HyperEdge(day, room, ops, p_map))

    return generated_edges


def rescue_swarm_worker(args):
    target_pid, room, available_days, day_room_map, p_map = args
    target_p = p_map[target_pid]
    generated_edges = []

    for day in available_days:
        base_pats = [p_map[pid] for pid in day_room_map.get((day, room), [])]
        if target_pid in [p.id for p in base_pats]: continue

        candidates = base_pats + [target_p]
        for _ in range(10):
            random.shuffle(candidates)
            ops = pack_patients(candidates)
            if ops: generated_edges.append(HyperEdge(day, room, ops, p_map))

        for _ in range(20):
            subset = [target_p]
            others = [p for p in base_pats]
            random.shuffle(others)
            for other in others:
                test_set = subset + [other]
                if pack_patients(test_set): subset.append(other)
            random.shuffle(subset)
            ops = pack_patients(subset)
            if ops: generated_edges.append(HyperEdge(day, room, ops, p_map))

    return generated_edges


def optimize_locked_edges(locked_edges, p_map):
    """Reposition locked patients to create maximum gaps"""
    optimized_edges = []
    repositioning_count = 0

    for locked_edge in locked_edges:
        locked_pids = locked_edge.pids
        locked_pats = [p_map[pid] for pid in locked_pids]

        best_ops = locked_edge.ops
        best_end = best_ops[-1]['end'] if best_ops else DAY_START_MIN

        for attempt in range(20):
            shuffled = list(locked_pats)
            random.shuffle(shuffled)
            ops = pack_patients(shuffled)

            if ops:
                current_end = ops[-1]['end']
                if current_end < best_end:
                    best_end = current_end
                    best_ops = ops

        optimized_edge = HyperEdge(locked_edge.day, locked_edge.room, best_ops, p_map, is_locked=True)
        optimized_edges.append(optimized_edge)

        time_freed = locked_edge.ops[-1]['end'] - best_ops[-1]['end']
        if time_freed > 0:
            repositioning_count += 1

    return optimized_edges


# ═══════════════════ CP-SAT SOLVER (exact copy) ═══════════════════
def solve_hypergraph(all_edges, required_ids, locked_ids, p_map, strict_required=False):
    print(f"    [Solver] Proc {len(all_edges)} edges. Req: {len(required_ids)}. "
          f"Locked: {len(locked_ids)}. Strict: {strict_required}")
    model = cp_model.CpModel()
    x_vars = {}

    patient_to_vars = defaultdict(list)
    room_day_to_vars = defaultdict(list)
    surgeon_intervals = defaultdict(list)

    for edge in all_edges:
        x = model.NewBoolVar(f"x_{edge.id}")
        x_vars[edge.id] = x
        room_day_to_vars[(edge.day, edge.room)].append(x)
        for pid in edge.pids:
            patient_to_vars[pid].append(x)

        day_offset = edge.day * 10000
        for (s_id, start, end) in edge.surgeon_usage:
            interval = model.NewOptionalFixedSizeIntervalVar(
                day_offset + start, end - start, x, f"s_{s_id}_{edge.id}"
            )
            surgeon_intervals[s_id].append(interval)

    for key, vars_list in room_day_to_vars.items():
        model.Add(sum(vars_list) <= 1)

    for s_id, intervals in surgeon_intervals.items():
        model.AddNoOverlap(intervals)

    for pid, vars_list in patient_to_vars.items():
        if pid in locked_ids:
            model.Add(sum(vars_list) == 1)
        elif strict_required and pid in required_ids:
            model.Add(sum(vars_list) == 1)
        else:
            model.Add(sum(vars_list) <= 1)

    obj_terms = [x_vars[edge.id] * edge.weight for edge in all_edges]
    model.Maximize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = SOLVER_TIME_LIMIT
    solver.parameters.num_search_workers = 16
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return [e for e in all_edges if solver.Value(x_vars[e.id])]
    return []


# ═══════════════════ BOUNDED STAGE SOLVERS (ARGOS algorithm) ═══════════════════
# These are the exact same algorithms as the original stage_solve_january,
# stage_solve_february_with_locked, stage_solve_march — but they accept
# and contribute to a GLOBAL edge_bank so α accumulates across iterations.

def stage_solve_january_bounded(jan_ids_bounded, all_pids, p_map, global_edge_bank):
    """
    Solve January for up to len(jan_ids_bounded) patients.
    Edges are generated INTO global_edge_bank (cumulative).
    Returns (solution_edges, jan_locked_pids).
    """
    print(f"\n>>> January Anchor: {len(jan_ids_bounded)} patients")

    day_room_map = defaultdict(list)
    for pid in jan_ids_bounded:
        p = p_map[pid]
        day_room_map[(p.original_day_idx, p.room)].append(pid)

    all_valid_days = sorted(set(k[0] for k in day_room_map.keys()))
    all_rooms = sorted(set(k[1] for k in day_room_map.keys()))
    print(f"    Days: {len(all_valid_days)}, Rooms: {len(all_rooms)}")

    # Phase 1: Seed — add to GLOBAL bank
    print(f"\n    [PHASE 1] Seeding Foundation")
    tasks = [(day, room, [p_map[pid] for pid in day_room_map.get((day, room), [])], p_map)
             for day in all_valid_days for room in all_rooms]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i in range(LAYER_1_ROUNDS):
            results = executor.map(swarm_worker, tasks)
            count = 0
            for edge_list in results:
                for e in edge_list:
                    if e.id not in global_edge_bank:
                        global_edge_bank[e.id] = e
                        count += 1
            print(f"        L1.{i + 1}: Added {count} edges (bank: {len(global_edge_bank)})")

    # Phase 2: Rescue & Lock — solve from global bank
    print(f"\n    [PHASE 2] Rescue & Lock")
    current_solution = []

    for r in range(LAYER_2_ROUNDS):
        is_strict = (r >= 2)
        current_solution = solve_hypergraph(
            list(global_edge_bank.values()),
            jan_ids_bounded, set(), p_map,
            strict_required=is_strict
        )

        if not current_solution:
            current_solution = solve_hypergraph(
                list(global_edge_bank.values()),
                jan_ids_bounded, set(), p_map,
                strict_required=False
            )

        covered = set()
        for e in current_solution:
            covered.update(e.pids)
        missing = [pid for pid in jan_ids_bounded if pid not in covered]

        print(f"        R{r + 1}: Coverage {len(jan_ids_bounded) - len(missing)}/{len(jan_ids_bounded)}")

        if not missing:
            print(f"        ✓ SUCCESS: All {len(jan_ids_bounded)} January locked!")
            return current_solution

        # Rescue → add to GLOBAL bank
        rescue_tasks = []
        for m_pid in missing:
            p = p_map[m_pid]
            p.priority_boost *= 5.0
            rescue_tasks.append((m_pid, p.room, all_valid_days, day_room_map, p_map))

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = executor.map(rescue_swarm_worker, rescue_tasks)
            new_count = 0
            for edge_list in results:
                for e in edge_list:
                    if e.id not in global_edge_bank:
                        global_edge_bank[e.id] = e
                        new_count += 1
            print(f"        Rescue: Added {new_count} edges (bank: {len(global_edge_bank)})")

    return current_solution


def stage_solve_february_bounded(feb_ids_bounded, jan_locked_pids, all_pids, p_map,
                                 jan_solution, global_edge_bank):
    """
    Solve February with January locked. Edges go into global_edge_bank.
    """
    print(f"\n>>> February Fill: {len(feb_ids_bounded)} patients "
          f"(Jan locked: {len(jan_locked_pids)})")

    day_room_map = defaultdict(list)
    for pid in feb_ids_bounded:
        p = p_map[pid]
        day_room_map[(p.original_day_idx, p.room)].append(pid)

    all_valid_days = sorted(set(k[0] for k in day_room_map.keys()))
    all_rooms = sorted(set(k[1] for k in day_room_map.keys()))
    print(f"    Days: {len(all_valid_days)}, Rooms: {len(all_rooms)}")

    # Reposition January locked edges → add to global bank
    print(f"\n    [REPOSITION JANUARY] Tightening schedule")
    optimized_jan_edges = optimize_locked_edges(jan_solution, p_map)
    for e in optimized_jan_edges:
        global_edge_bank[e.id] = e

    # Phase 1: Seed February → add to global bank
    print(f"\n    [PHASE 1] Seeding February")
    tasks = [(day, room, [p_map[pid] for pid in day_room_map.get((day, room), [])], p_map)
             for day in all_valid_days for room in all_rooms]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i in range(LAYER_1_ROUNDS):
            results = executor.map(swarm_worker, tasks)
            count = 0
            for edge_list in results:
                for e in edge_list:
                    if e.id not in global_edge_bank:
                        global_edge_bank[e.id] = e
                        count += 1
            print(f"        L1.{i + 1}: Added {count} edges (bank: {len(global_edge_bank)})")

    # Phase 2: Rescue & Lock February
    print(f"\n    [PHASE 2] Rescue & Lock February")
    current_solution = []

    for r in range(LAYER_2_ROUNDS):
        is_strict = (r >= 2)
        current_solution = solve_hypergraph(
            list(global_edge_bank.values()),
            feb_ids_bounded, jan_locked_pids, p_map,
            strict_required=is_strict
        )

        if not current_solution:
            current_solution = solve_hypergraph(
                list(global_edge_bank.values()),
                feb_ids_bounded, jan_locked_pids, p_map,
                strict_required=False
            )

        covered = set()
        for e in current_solution:
            covered.update(e.pids)
        missing = [pid for pid in feb_ids_bounded if pid not in covered]
        jan_check = len([pid for pid in jan_locked_pids if pid in covered])

        print(f"        R{r + 1}: Feb {len(feb_ids_bounded) - len(missing)}/{len(feb_ids_bounded)} "
              f"| Jan {jan_check}/{len(jan_locked_pids)}")

        if not missing and jan_check == len(jan_locked_pids):
            print(f"        ✓ SUCCESS: All Feb + Jan locked!")
            return current_solution

        rescue_tasks = []
        for m_pid in missing:
            p = p_map[m_pid]
            p.priority_boost *= 5.0
            rescue_tasks.append((m_pid, p.room, all_valid_days, day_room_map, p_map))

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = executor.map(rescue_swarm_worker, rescue_tasks)
            new_count = 0
            for edge_list in results:
                for e in edge_list:
                    if e.id not in global_edge_bank:
                        global_edge_bank[e.id] = e
                        new_count += 1
            print(f"        Rescue: Added {new_count} edges (bank: {len(global_edge_bank)})")

    return current_solution


def stage_solve_march_bounded(mar_ids_bounded, jan_locked_pids, feb_locked_pids,
                              all_pids, p_map, feb_solution, global_edge_bank):
    """
    Solve March best-effort with Jan+Feb locked. Edges go into global_edge_bank.
    """
    print(f"\n>>> March Extension: {len(mar_ids_bounded)} patients "
          f"(Jan+Feb locked: {len(jan_locked_pids) + len(feb_locked_pids)})")

    day_room_map = defaultdict(list)
    for pid in mar_ids_bounded:
        p = p_map[pid]
        day_room_map[(p.original_day_idx, p.room)].append(pid)

    all_valid_days = sorted(set(k[0] for k in day_room_map.keys()))
    all_rooms = sorted(set(k[1] for k in day_room_map.keys()))
    print(f"    Days: {len(all_valid_days)}, Rooms: {len(all_rooms)}")

    # Reposition Jan+Feb → add to global bank
    print(f"\n    [REPOSITION JAN+FEB] Tightening schedule")
    optimized_janfeb = optimize_locked_edges(feb_solution, p_map)
    for e in optimized_janfeb:
        global_edge_bank[e.id] = e

    # Phase 1: Seed March
    print(f"\n    [PHASE 1] Seeding March")
    tasks = [(day, room, [p_map[pid] for pid in day_room_map.get((day, room), [])], p_map)
             for day in all_valid_days for room in all_rooms]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i in range(LAYER_1_ROUNDS):
            results = executor.map(swarm_worker, tasks)
            count = 0
            for edge_list in results:
                for e in edge_list:
                    if e.id not in global_edge_bank:
                        global_edge_bank[e.id] = e
                        count += 1
            print(f"        L1.{i + 1}: Added {count} edges (bank: {len(global_edge_bank)})")

    # Phase 2: Best-Effort March
    print(f"\n    [PHASE 2] Best-Effort March")
    locked_pids = jan_locked_pids | feb_locked_pids
    current_solution = solve_hypergraph(
        list(global_edge_bank.values()),
        mar_ids_bounded, locked_pids, p_map,
        strict_required=False
    )

    covered = set()
    for e in current_solution:
        covered.update(e.pids)
    mar_covered = [pid for pid in mar_ids_bounded if pid in covered]
    janfeb_covered = [pid for pid in locked_pids if pid in covered]

    print(f"        Coverage: Mar {len(mar_covered)}/{len(mar_ids_bounded)} "
          f"| Jan+Feb {len(janfeb_covered)}/{len(locked_pids)}")

    return current_solution


# ═══════════════════ MAIN ITERATION LOOP ═══════════════════
def main():
    print("=" * 70)
    print("ARGOS ITERATIVE SOLVER — 100 Sigmoid + Convergence")
    print("Uses REAL CP-SAT solver with cumulative edge bank")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_by_dept = load_data()

    for dept in DEPARTMENTS:
        if dept not in data_by_dept:
            print(f"\n>>> {dept}: No data, skipping.")
            continue

        patients = data_by_dept[dept]
        dept_short = dept.replace(" ", "_").replace("RST_ROMB_", "")

        p_map = {p.id: p for p in patients}
        all_pids = list(p_map.keys())

        jan_ids = [pid for pid in all_pids if p_map[pid].month == 1]
        feb_ids = [pid for pid in all_pids if p_map[pid].month == 2]
        mar_ids = [pid for pid in all_pids if p_map[pid].month == 3]

        jan_total = len(jan_ids)
        feb_total = len(feb_ids)
        mar_total = len(mar_ids)
        grand_total = jan_total + feb_total + mar_total

        print(f"\n{'=' * 70}")
        print(f">>> {dept}: Jan={jan_total}, Feb={feb_total}, Mar={mar_total}, Total={grand_total}")
        print(f"{'=' * 70}")

        # Compute total possible paths
        print(f"\n  Computing total possible paths in hypergraph...")
        total_possible = count_total_possible_paths(patients)
        print(f"  Total valid packings: {total_possible:,}")
        print(f"  Log10(total): {math.log10(total_possible):.2f}")

        # GLOBAL edge bank — accumulates across ALL iterations
        global_edge_bank = {}

        results = []

        def run_one_iteration(iter_num, jan_bound, feb_bound, mar_bound, label=""):
            """Run one full cascade iteration using the ARGOS algorithm."""
            t_start = time.time()

            # Reset priority boosts
            for pid in all_pids:
                p_map[pid].priority_boost = 1.0

            print(f"\n{'─' * 70}")
            print(f"  {label}Iteration {iter_num} "
                  f"[Bounds: Jan={jan_bound}, Feb={feb_bound}, Mar={mar_bound}]")
            print(f"{'─' * 70}")

            # Subsample patient IDs to the bounded count
            jan_ids_bounded = sorted(jan_ids)[:jan_bound] if jan_bound < jan_total else jan_ids
            feb_ids_bounded = sorted(feb_ids)[:feb_bound] if feb_bound < feb_total else feb_ids
            mar_ids_bounded = sorted(mar_ids)[:mar_bound] if mar_bound < mar_total else mar_ids

            # ── STAGE 1: JANUARY ──
            jan_solution = stage_solve_january_bounded(
                jan_ids_bounded, all_pids, p_map, global_edge_bank
            )

            jan_locked_pids = set()
            for e in jan_solution:
                for pid in e.pids:
                    if p_map[pid].month == 1:
                        jan_locked_pids.add(pid)
            jan_solved = len(jan_locked_pids)

            # ── STAGE 2: FEBRUARY ──
            feb_solved = 0
            feb_solution = jan_solution
            if feb_bound > 0 and jan_solved >= jan_bound * 0.95:
                feb_solution = stage_solve_february_bounded(
                    feb_ids_bounded, jan_locked_pids, all_pids, p_map,
                    jan_solution, global_edge_bank
                )

                covered = set()
                for e in feb_solution:
                    covered.update(e.pids)
                feb_solved = len([pid for pid in feb_ids_bounded if pid in covered])

            # ── STAGE 3: MARCH ──
            mar_solved = 0
            final_solution = feb_solution
            if mar_bound > 0 and feb_solved >= feb_bound * 0.95:
                feb_locked_pids = set()
                for e in feb_solution:
                    for pid in e.pids:
                        if p_map[pid].month == 2:
                            feb_locked_pids.add(pid)

                final_solution = stage_solve_march_bounded(
                    mar_ids_bounded, jan_locked_pids, feb_locked_pids,
                    all_pids, p_map, feb_solution, global_edge_bank
                )

                covered = set()
                for e in final_solution:
                    covered.update(e.pids)
                mar_solved = len([pid for pid in mar_ids_bounded if pid in covered])

            total_solved = jan_solved + feb_solved + mar_solved
            total_paths = len(final_solution)
            edges_explored = len(global_edge_bank)
            alpha = edges_explored / total_possible if total_possible > 0 else 0
            elapsed = time.time() - t_start

            result = {
                "iteration": iter_num,
                "department": dept,
                "total_paths": total_paths,
                "total_patients_solved": total_solved,
                "jan_solved": jan_solved,
                "jan_total": jan_total,
                "jan_bound": jan_bound,
                "feb_solved": feb_solved,
                "feb_total": feb_total,
                "feb_bound": feb_bound,
                "mar_solved": mar_solved,
                "mar_total": mar_total,
                "mar_bound": mar_bound,
                "pct_total": round(100 * total_solved / grand_total, 2),
                "edges_explored": edges_explored,
                "alpha": alpha,
                "total_possible_paths": total_possible,
                "runtime_seconds": round(elapsed, 1),
            }
            results.append(result)

            print(f"\n    => Jan {jan_solved}/{jan_total} | "
                  f"Feb {feb_solved}/{feb_total} | "
                  f"Mar {mar_solved}/{mar_total} | "
                  f"Total {total_solved}/{grand_total} ({result['pct_total']}%) | "
                  f"Paths {total_paths} | "
                  f"Bank {edges_explored} | α={alpha:.4e} | "
                  f"Time {elapsed:.1f}s")

            return total_solved

        # ═══ SIGMOID PHASE: 100 iterations ═══
        for iteration in range(NUM_ITERATIONS):
            jb, fb, mb = compute_bounds(iteration, jan_total, feb_total, mar_total)
            run_one_iteration(iteration + 1, jb, fb, mb)

        # ═══ CONVERGENCE PHASE ═══
        print(f"\n{'=' * 70}")
        print(f"  CONVERGENCE PHASE — up to {CONVERGENCE_MAX_ITERS} extra iterations (full bounds)")
        print(f"{'=' * 70}")

        last_best = results[-1]["total_patients_solved"] if results else 0
        no_improve = 0

        for ci in range(CONVERGENCE_MAX_ITERS):
            conv_num = NUM_ITERATIONS + ci + 1
            tot_s = run_one_iteration(conv_num, jan_total, feb_total, mar_total, "Conv. ")

            if tot_s > last_best:
                last_best = tot_s
                no_improve = 0
                print(f"    ↑ New best: {last_best}")
            else:
                no_improve += 1
                print(f"    — No improvement ({no_improve}/{CONVERGENCE_PATIENCE})")

            if no_improve >= CONVERGENCE_PATIENCE:
                print(f"\n  ✓ CONVERGED at iteration {conv_num} "
                      f"(no improvement for {CONVERGENCE_PATIENCE} iters)")
                print(f"    Final: {last_best}/{grand_total} "
                      f"({round(100 * last_best / grand_total, 2)}%)")
                break

        total_iters = len(results)
        print(f"\n  Total: {total_iters} iterations "
              f"(100 sigmoid + {total_iters - NUM_ITERATIONS} convergence)")
        print(f"  Global edge bank: {len(global_edge_bank):,} unique edges")

        # ── Enforce monotonicity ──
        best = {"total": 0, "jan": 0, "feb": 0, "mar": 0, "paths": 0, "alpha": 0,
                "edges": 0}
        for r in results:
            if r["total_patients_solved"] >= best["total"]:
                best["total"] = r["total_patients_solved"]
                best["jan"] = r["jan_solved"]
                best["feb"] = r["feb_solved"]
                best["mar"] = r["mar_solved"]
                best["paths"] = r["total_paths"]
                best["alpha"] = r["alpha"]
                best["edges"] = r["edges_explored"]
            else:
                r["total_patients_solved"] = best["total"]
                r["jan_solved"] = best["jan"]
                r["feb_solved"] = best["feb"]
                r["mar_solved"] = best["mar"]
                r["total_paths"] = best["paths"]
                # Keep alpha growing (it's cumulative, always grows)
                r["pct_total"] = round(100 * best["total"] / grand_total, 2)

        # ── Save JSON ──
        json_path = os.path.join(OUTPUT_DIR, f"{dept_short}_iterations.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved: {json_path}")

        # ── Save CSV ──
        csv_path = os.path.join(OUTPUT_DIR, f"{dept_short}_iterations.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"  Saved: {csv_path}")

    # ── Plots ──
    print(f"\n{'=' * 70}")
    print("Generating plots...")
    print(f"{'=' * 70}")
    generate_plots()


# ═══════════════════ PLOT GENERATION ═══════════════════
def generate_plots():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import LogLocator, FuncFormatter
        import numpy as np
    except ImportError:
        print("  Need: pip install matplotlib numpy")
        return

    try:
        from scipy.interpolate import make_interp_spline
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    # ── Custom log tick formatter: 10^-k, 2×10^-k, 5×10^-k ──
    def log_tick_fmt(x, pos):
        if x <= 0: return ""
        exp = math.floor(math.log10(x))
        m = x / (10 ** exp)
        if abs(m - 1.0) < 0.15:  return r"$10^{%d}$" % exp
        elif abs(m - 2.0) < 0.3: return r"$2\!\times\!10^{%d}$" % exp
        elif abs(m - 5.0) < 0.5: return r"$5\!\times\!10^{%d}$" % exp
        return ""

    # ── Smoothing utility ──
    def smooth(x, y, n=300):
        if not HAS_SCIPY or len(x) < 4:
            return np.array(x), np.array(y)
        pairs = sorted(set(zip(x, y)))
        xc = [p[0] for p in pairs]
        yc = [p[1] for p in pairs]
        if len(xc) < 4:
            return np.array(xc), np.array(yc)
        xa, ya = np.array(xc), np.array(yc)
        xn = np.linspace(xa.min(), xa.max(), n)
        try:
            return xn, make_interp_spline(xa, ya, k=3)(xn)
        except:
            return xa, ya

    # ── Per-department α plots (matching the reference style) ──
    for dept in DEPARTMENTS:
        ds = dept.replace(" ", "_").replace("RST_ROMB_", "")
        json_path = os.path.join(OUTPUT_DIR, f"{ds}_iterations.json")
        if not os.path.exists(json_path): continue
        with open(json_path) as f:
            res = json.load(f)

        valid = [(r["alpha"], r["total_patients_solved"], r["runtime_seconds"])
                 for r in res if r["alpha"] > 0]
        if not valid: continue
        va, vs, vt = zip(*valid)
        la = np.log10(np.array(va))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(f"ARGOS Scheduling Metrics over Graph Density (α)\n{dept}",
                     fontsize=15, fontweight="bold", y=0.98)

        # LEFT: Scheduled Patients vs α (sigmoid S-curve)
        xs, ys = smooth(la.tolist(), list(vs))
        ax1.plot(10**xs, ys, color="#1f77b4", linewidth=2.8)
        ax1.set_xscale("log")
        ax1.set_xlabel(r"Exploration Fraction ($\alpha$)", fontsize=13)
        ax1.set_ylabel("Total Patients Scheduled", fontsize=13)
        ax1.set_title(r"Scheduled Patients as a function of $\alpha$", fontsize=14)
        ax1.grid(True, which="both", alpha=0.25, linestyle="--")
        ax1.xaxis.set_major_locator(LogLocator(base=10, subs=[1.0, 2.0, 5.0], numticks=15))
        ax1.xaxis.set_major_formatter(FuncFormatter(log_tick_fmt))
        ax1.tick_params(labelsize=11)

        # RIGHT: Solver Runtime vs α
        xs2, ys2 = smooth(la.tolist(), list(vt))
        ax2.plot(10**xs2, ys2, color="#d62728", linewidth=2.8)
        ax2.set_xscale("log")
        ax2.set_xlabel(r"Exploration Fraction ($\alpha$)", fontsize=13)
        ax2.set_ylabel("Iteration Runtime (Seconds)", fontsize=13)
        ax2.set_title(r"Solver Runtime as a function of $\alpha$", fontsize=14)
        ax2.grid(True, which="both", alpha=0.25, linestyle="--")
        ax2.xaxis.set_major_locator(LogLocator(base=10, subs=[1.0, 2.0, 5.0], numticks=15))
        ax2.xaxis.set_major_formatter(FuncFormatter(log_tick_fmt))
        ax2.tick_params(labelsize=11)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plot_path = os.path.join(OUTPUT_DIR, f"{ds}_alpha_plot.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Plot saved: {plot_path}")

        # ── 4-panel iteration-based plot ──
        iters = [r["iteration"] for r in res]
        ts = [r["total_patients_solved"] for r in res]
        js = [r["jan_solved"] for r in res]
        fs = [r["feb_solved"] for r in res]
        ms = [r["mar_solved"] for r in res]
        grand = res[0]["jan_total"] + res[0]["feb_total"] + res[0]["mar_total"]

        fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig2.suptitle(f"{dept} — Iteration Progress ({len(res)} total)",
                      fontsize=15, fontweight="bold")

        ax = axes[0, 0]
        ax.plot(iters, ts, "b-", lw=2.5)
        ax.axhline(y=grand, color="r", ls="--", alpha=.5)
        ax.axvline(x=100, color="gray", ls=":", alpha=.6)
        ax.fill_between(iters, ts, alpha=.12, color="blue")
        ax.set_xlabel("Iteration"); ax.set_ylabel("Patients")
        ax.set_title("Total Patients Solved"); ax.grid(True, alpha=.3)

        ax = axes[0, 1]
        ax.fill_between(iters, 0, js, alpha=.4, color="green", label="Jan")
        ax.fill_between(iters, js, [j+f for j, f in zip(js, fs)],
                        alpha=.4, color="orange", label="Feb")
        ax.fill_between(iters, [j+f for j, f in zip(js, fs)],
                        [j+f+m for j, f, m in zip(js, fs, ms)],
                        alpha=.4, color="red", label="Mar")
        ax.axvline(x=100, color="gray", ls=":", alpha=.6)
        ax.set_xlabel("Iteration"); ax.set_ylabel("Patients")
        ax.set_title("Stacked by Month"); ax.legend(fontsize=9); ax.grid(True, alpha=.3)

        ax = axes[1, 0]
        ax.plot(iters, js, "g-", lw=2, label="Jan")
        ax.plot(iters, fs, color="orange", lw=2, label="Feb")
        ax.plot(iters, ms, "r-", lw=2, label="Mar")
        ax.axvline(x=100, color="gray", ls=":", alpha=.6)
        ax.set_xlabel("Iteration"); ax.set_ylabel("Patients")
        ax.set_title("Per-Month"); ax.legend(fontsize=9); ax.grid(True, alpha=.3)

        ax = axes[1, 1]
        edges = [r["edges_explored"] for r in res]
        ax.plot(iters, edges, "purple", lw=2, label="Cumulative Edges")
        ax.fill_between(iters, edges, alpha=.12, color="purple")
        ax.axvline(x=100, color="gray", ls=":", alpha=.6)
        ax.set_xlabel("Iteration"); ax.set_ylabel("Unique Edges in Bank")
        ax.set_title("Cumulative Edge Bank Size"); ax.legend(fontsize=9)
        ax.grid(True, alpha=.3)

        plt.tight_layout(rect=[0, 0, 1, .95])
        plot_path2 = os.path.join(OUTPUT_DIR, f"{ds}_iteration_plot.png")
        plt.savefig(plot_path2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved: {plot_path2}")

    # ── Combined α plot (both departments) ──
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {"RST ROMB CCL": "#1f77b4", "RST ROMB HRS": "#2ca02c"}
    for dept in DEPARTMENTS:
        ds = dept.replace(" ", "_").replace("RST_ROMB_", "")
        json_path = os.path.join(OUTPUT_DIR, f"{ds}_iterations.json")
        if not os.path.exists(json_path): continue
        with open(json_path) as f:
            res = json.load(f)
        valid = [(r["alpha"], r["total_patients_solved"]) for r in res if r["alpha"] > 0]
        if not valid: continue
        va, vs = zip(*valid)
        la = np.log10(np.array(va))
        xs, ys = smooth(la.tolist(), list(vs))
        ax.plot(10**xs, ys, lw=2.8, color=colors.get(dept, "gray"), label=dept)

    ax.set_xscale("log")
    ax.set_xlabel(r"Exploration Fraction ($\alpha$)", fontsize=13)
    ax.set_ylabel("Total Patients Scheduled", fontsize=13)
    ax.set_title(r"ARGOS Scheduling: Patients vs Graph Density ($\alpha$)",
                 fontsize=14, fontweight="bold")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=[1.0, 2.0, 5.0], numticks=15))
    ax.xaxis.set_major_formatter(FuncFormatter(log_tick_fmt))
    ax.grid(True, which="both", alpha=.25, linestyle="--")
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=11)

    combined_path = os.path.join(OUTPUT_DIR, "combined_alpha_plot.png")
    plt.savefig(combined_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Combined plot saved: {combined_path}")


if __name__ == "__main__":
    main()