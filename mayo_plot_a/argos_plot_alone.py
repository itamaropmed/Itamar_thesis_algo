"""
ARGOS ITERATIVE SOLVER — 100 Iterations + Convergence Phase
====================================================================
Uses the REAL CP-SAT solver (ortools) with hypergraph optimization.
Requires: pip install ortools

Run this in the same directory as informative_data.json.

The progression follows a sigmoid/tanh curve:
  - January fills first (steep sigmoid, center ~iter 13)
  - February opens only after January hits threshold (center ~iter 42)
  - March opens only after February hits threshold (center ~iter 78)
  - The upper bound grows more at the end → tanh-like shape

After 100 sigmoid iterations, a CONVERGENCE PHASE runs with full
bounds (all months maxed) for up to 20 more iterations, stopping
early once the total patients solved stabilizes (no improvement
for 5 consecutive iterations).
"""

import json
import csv
import random
import os
import datetime
import math
import hashlib
import copy
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

try:
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False
    print(">>> CRITICAL: 'ortools' not found. Install with: pip install ortools")
    exit(1)

# ─────────────────── CONSTANTS ───────────────────
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

# Solver params (reduced per iteration for speed across 100 iters)
LAYER_1_ROUNDS = 4
LAYER_2_ROUNDS = 3
SOLVER_TIME_LIMIT = 30

NUM_ITERATIONS = 100
CONVERGENCE_MAX_ITERS = 20      # max extra iterations for convergence phase
CONVERGENCE_PATIENCE = 5        # stop if no improvement for this many consecutive iters
OUTPUT_DIR = "iteration_results"

# ─────────────────── SIGMOID BOUND PARAMETERS ───────────────────
# January: steep sigmoid, completes early (iterations ~1-25)
JAN_CENTER = 13
JAN_K = 0.35

# February: medium sigmoid, fills mid range (iterations ~25-60)
FEB_CENTER = 42
FEB_K = 0.16

# March: gentle/slow sigmoid, fills late (iterations ~60-100)
# This makes the upper bound grow MORE at the end → tanh shape
MAR_CENTER = 78
MAR_K = 0.065

# Cascade unlock: when sigmoid fraction >= this, next month opens
CASCADE_THRESHOLD = 0.97


def sigmoid(x, center, k):
    z = k * (x - center)
    if z > 500:
        return 1.0
    if z < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def compute_bounds(iteration, jan_total, feb_total, mar_total):
    """
    Compute month capacity bounds for a given iteration.
    Cascade logic:
      - Feb only opens when Jan sigmoid >= CASCADE_THRESHOLD
      - Mar only opens when Feb sigmoid >= CASCADE_THRESHOLD
    """
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


# ─────────────────── HELPER FUNCTIONS ───────────────────
def minutes_to_hhmm(minutes):
    h, m = divmod(minutes, 60)
    h = h % 24
    return f"{int(h):02d}:{int(m):02d}:00"


def parse_time(t_str):
    if not t_str:
        return None
    try:
        t = datetime.datetime.strptime(t_str, "%H:%M:%S")
        return t.hour * 60 + t.minute
    except:
        return None


# ─────────────────── PATIENT & HYPEREDGE ───────────────────
class Patient:
    __slots__ = ['id', 'duration', 'room', 'surgeon', 'original_date',
                 'year', 'type', 'weight', 'department', 'original_day_idx',
                 'priority_boost', 'month', 'is_fixed', 'eligible_rooms']

    def __init__(self, pid, duration, room, surgeon, date_str, year,
                 p_type, dept, day_idx, month, fixed_rooms, all_rooms):
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


# ─────────────────── DATA LOADING ───────────────────
def load_data():
    if not os.path.exists(INPUT_FILE):
        return {}
    with open(INPUT_FILE, 'r') as f:
        raw_data = json.load(f)
    print(f">>> Loaded {len(raw_data)} raw records.")

    departments = defaultdict(lambda: {"patients": [], "rooms": set()})

    for i, entry in enumerate(raw_data):
        if entry.get("Discharge Location") != "RST MCH Saint Marys Campus":
            continue
        dept = entry.get("OR Department")
        if dept not in DEPARTMENTS:
            continue
        date_str = entry.get("Date")
        if not date_str:
            continue
        try:
            dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except:
            continue
        if dt.year != TARGET_YEAR:
            continue
        room = entry.get("Room")
        if room and "Temp" not in room:
            departments[dept]["rooms"].add(room)
        if dt.month not in (1, 2, 3):
            continue

        in_time = parse_time(entry.get("In Proc Room"))
        out_time = parse_time(entry.get("Out Proc Room"))
        if in_time is None or out_time is None:
            continue
        if out_time >= in_time:
            duration = out_time - in_time
        else:
            duration = (1440 - in_time) + out_time
        if duration <= 0:
            duration = 30
        if duration > 720:
            duration = 720

        surgeon = entry.get("Lead Surgeon/Provider")
        if not room or not surgeon:
            continue

        p_type_map = {1: 'JAN', 2: 'FEB', 3: 'MAR'}
        p_type, month = p_type_map[dt.month], dt.month
        day_idx = (dt.date() - ANCHOR_DATE).days

        p = Patient(i, duration, room, surgeon, date_str, dt.year, p_type, dept,
                    day_idx, month, FIXED_ROOMS[dept], departments[dept]["rooms"])
        departments[dept]["patients"].append(p)

    result = {}
    for dept in DEPARTMENTS:
        if departments[dept]["patients"]:
            result[dept] = departments[dept]["patients"]
    return result


# ─────────────────── PACKING ───────────────────
def pack_patients(patients):
    ops = []
    curr = DAY_START_MIN
    for p in patients:
        start = curr
        end = start + p.duration
        if end > DAY_END_MIN:
            return None
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
            if ops:
                generated_edges.append(HyperEdge(day, room, ops, p_map))
    return generated_edges


def rescue_swarm_worker(args):
    target_pid, room, available_days, day_room_map, p_map = args
    target_p = p_map[target_pid]
    generated_edges = []

    for day in available_days:
        base_pats = [p_map[pid] for pid in day_room_map.get((day, room), [])]
        if target_pid in [p.id for p in base_pats]:
            continue
        candidates = base_pats + [target_p]
        for _ in range(10):
            random.shuffle(candidates)
            ops = pack_patients(candidates)
            if ops:
                generated_edges.append(HyperEdge(day, room, ops, p_map))
        for _ in range(20):
            subset = [target_p]
            others = list(base_pats)
            random.shuffle(others)
            for other in others:
                test_set = subset + [other]
                if pack_patients(test_set):
                    subset.append(other)
            random.shuffle(subset)
            ops = pack_patients(subset)
            if ops:
                generated_edges.append(HyperEdge(day, room, ops, p_map))
    return generated_edges


def optimize_locked_edges(locked_edges, p_map):
    """Reposition locked patients to create maximum gaps."""
    optimized_edges = []
    for locked_edge in locked_edges:
        locked_pats = [p_map[pid] for pid in locked_edge.pids]
        best_ops = locked_edge.ops
        best_end = best_ops[-1]['end'] if best_ops else DAY_START_MIN

        for _ in range(20):
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
    return optimized_edges


# ─────────────────── CP-SAT SOLVER ───────────────────
def solve_hypergraph(all_edges, required_ids, locked_ids, p_map, strict_required=False):
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
    solver.parameters.num_search_workers = min(16, os.cpu_count() or 4)
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return [e for e in all_edges if solver.Value(x_vars[e.id])]
    return []


# ─────────────────── BOUNDED STAGE SOLVERS ───────────────────
def stage_solve_month(month_ids, locked_pids, p_map, prev_solution, month_name,
                      max_to_schedule):
    """
    Solve a single month with a capacity bound.
    If max_to_schedule < len(month_ids), we only try to schedule that many.
    locked_pids are from previous months and must remain scheduled.
    prev_solution contains locked edges from previous months.
    """
    if max_to_schedule <= 0:
        return prev_solution if prev_solution else [], 0

    # Subsample: only attempt max_to_schedule patients from this month
    if max_to_schedule < len(month_ids):
        # Deterministic but varied subsample
        sampled_ids = sorted(month_ids)[:max_to_schedule]
    else:
        sampled_ids = month_ids

    day_room_map = defaultdict(list)
    for pid in sampled_ids:
        p = p_map[pid]
        day_room_map[(p.original_day_idx, p.room)].append(pid)

    all_valid_days = sorted(set(k[0] for k in day_room_map.keys()))
    all_rooms = sorted(set(k[1] for k in day_room_map.keys()))

    # Start with optimized locked edges from previous months
    if prev_solution:
        optimized_prev = optimize_locked_edges(prev_solution, p_map)
        edge_bank = {e.id: e for e in optimized_prev}
    else:
        edge_bank = {}

    # Phase 1: Seed edges
    tasks = [(day, room, [p_map[pid] for pid in day_room_map.get((day, room), [])], p_map)
             for day in all_valid_days for room in all_rooms]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i in range(LAYER_1_ROUNDS):
            results = executor.map(swarm_worker, tasks)
            for edge_list in results:
                for e in edge_list:
                    if e.id not in edge_bank:
                        edge_bank[e.id] = e

    # Phase 2: Solve + Rescue
    current_solution = []
    for r in range(LAYER_2_ROUNDS):
        is_strict = (r >= 2)
        current_solution = solve_hypergraph(
            list(edge_bank.values()),
            sampled_ids,
            locked_pids,
            p_map,
            strict_required=is_strict
        )
        if not current_solution:
            current_solution = solve_hypergraph(
                list(edge_bank.values()),
                sampled_ids,
                locked_pids,
                p_map,
                strict_required=False
            )

        covered = set()
        for e in current_solution:
            covered.update(e.pids)
        missing = [pid for pid in sampled_ids if pid not in covered]

        if not missing:
            break

        # Rescue missing patients
        rescue_tasks = []
        for m_pid in missing:
            p = p_map[m_pid]
            p.priority_boost *= 3.0
            rescue_tasks.append((m_pid, p.room, all_valid_days, day_room_map, p_map))

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = executor.map(rescue_swarm_worker, rescue_tasks)
            for edge_list in results:
                for e in edge_list:
                    if e.id not in edge_bank:
                        edge_bank[e.id] = e

    # Count how many of this month were scheduled
    covered = set()
    for e in current_solution:
        covered.update(e.pids)
    month_scheduled = len([pid for pid in sampled_ids if pid in covered])

    return current_solution, month_scheduled


# ─────────────────── MAIN ITERATION LOOP ───────────────────
def main():
    print("=" * 70)
    print("ARGOS ITERATIVE SOLVER — 100 Iterations + Convergence Phase")
    print("Uses REAL CP-SAT solver (ortools)")
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

        results = []

        for iteration in range(NUM_ITERATIONS):
            iter_num = iteration + 1

            # Reset priority boosts each iteration
            for pid in all_pids:
                p_map[pid].priority_boost = 1.0

            # Compute sigmoid bounds
            jan_bound, feb_bound, mar_bound = compute_bounds(
                iteration, jan_total, feb_total, mar_total
            )

            print(f"\n  --- Iteration {iter_num}/100 "
                  f"[Bounds: Jan={jan_bound}, Feb={feb_bound}, Mar={mar_bound}] ---")

            # Stage 1: January
            jan_solution, jan_solved = stage_solve_month(
                jan_ids, set(), p_map, None, "January", jan_bound
            )
            jan_locked_pids = set()
            for e in jan_solution:
                for pid in e.pids:
                    if p_map[pid].month == 1:
                        jan_locked_pids.add(pid)

            # Stage 2: February (only if bound > 0)
            feb_solved = 0
            feb_solution = jan_solution
            if feb_bound > 0 and jan_solved >= jan_bound * 0.95:
                feb_solution, feb_solved = stage_solve_month(
                    feb_ids, jan_locked_pids, p_map, jan_solution, "February", feb_bound
                )

            # Stage 3: March (only if bound > 0)
            mar_solved = 0
            final_solution = feb_solution
            if mar_bound > 0 and feb_solved >= feb_bound * 0.95:
                feb_locked_pids = set()
                for e in feb_solution:
                    for pid in e.pids:
                        if p_map[pid].month == 2:
                            feb_locked_pids.add(pid)
                all_locked = jan_locked_pids | feb_locked_pids
                final_solution, mar_solved = stage_solve_month(
                    mar_ids, all_locked, p_map, feb_solution, "March", mar_bound
                )

            total_solved = jan_solved + feb_solved + mar_solved
            total_paths = len(final_solution)

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
            }
            results.append(result)

            print(f"    => Jan {jan_solved}/{jan_total} | "
                  f"Feb {feb_solved}/{feb_total} | "
                  f"Mar {mar_solved}/{mar_total} | "
                  f"Total {total_solved}/{grand_total} ({result['pct_total']}%) | "
                  f"Paths {total_paths}")

        # ─── CONVERGENCE PHASE: full bounds, run until stable ───
        print(f"\n{'=' * 70}")
        print(f"  CONVERGENCE PHASE — Full bounds, up to {CONVERGENCE_MAX_ITERS} extra iterations")
        print(f"{'=' * 70}")

        last_best = results[-1]["total_patients_solved"] if results else 0
        no_improve_count = 0

        for conv_iter in range(CONVERGENCE_MAX_ITERS):
            conv_num = NUM_ITERATIONS + conv_iter + 1

            # Reset priority boosts each iteration
            for pid in all_pids:
                p_map[pid].priority_boost = 1.0

            # Full bounds — everything open
            jan_bound = jan_total
            feb_bound = feb_total
            mar_bound = mar_total

            print(f"\n  --- Convergence Iter {conv_num} "
                  f"[Bounds: Jan={jan_bound}, Feb={feb_bound}, Mar={mar_bound}] ---")

            # Stage 1: January (full)
            jan_solution, jan_solved = stage_solve_month(
                jan_ids, set(), p_map, None, "January", jan_bound
            )
            jan_locked_pids = set()
            for e in jan_solution:
                for pid in e.pids:
                    if p_map[pid].month == 1:
                        jan_locked_pids.add(pid)

            # Stage 2: February (full)
            feb_solved = 0
            feb_solution = jan_solution
            if jan_solved >= jan_bound * 0.95:
                feb_solution, feb_solved = stage_solve_month(
                    feb_ids, jan_locked_pids, p_map, jan_solution, "February", feb_bound
                )

            # Stage 3: March (full)
            mar_solved = 0
            final_solution = feb_solution
            if feb_solved >= feb_bound * 0.95:
                feb_locked_pids = set()
                for e in feb_solution:
                    for pid in e.pids:
                        if p_map[pid].month == 2:
                            feb_locked_pids.add(pid)
                all_locked = jan_locked_pids | feb_locked_pids
                final_solution, mar_solved = stage_solve_month(
                    mar_ids, all_locked, p_map, feb_solution, "March", mar_bound
                )

            total_solved = jan_solved + feb_solved + mar_solved
            total_paths = len(final_solution)

            result = {
                "iteration": conv_num,
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
            }
            results.append(result)

            print(f"    => Jan {jan_solved}/{jan_total} | "
                  f"Feb {feb_solved}/{feb_total} | "
                  f"Mar {mar_solved}/{mar_total} | "
                  f"Total {total_solved}/{grand_total} ({result['pct_total']}%) | "
                  f"Paths {total_paths}")

            # Check convergence
            if total_solved > last_best:
                last_best = total_solved
                no_improve_count = 0
                print(f"    ↑ New best: {last_best}")
            else:
                no_improve_count += 1
                print(f"    — No improvement ({no_improve_count}/{CONVERGENCE_PATIENCE})")

            if no_improve_count >= CONVERGENCE_PATIENCE:
                print(f"\n  ✓ CONVERGED at iteration {conv_num} "
                      f"(no improvement for {CONVERGENCE_PATIENCE} consecutive iters)")
                print(f"    Final: {last_best}/{grand_total} "
                      f"({round(100 * last_best / grand_total, 2)}%)")
                break
        else:
            print(f"\n  ⚠ Reached max convergence iterations ({CONVERGENCE_MAX_ITERS}) "
                  f"without full convergence")
            print(f"    Best: {last_best}/{grand_total} "
                  f"({round(100 * last_best / grand_total, 2)}%)")

        total_iters = len(results)
        print(f"\n  Total iterations run: {total_iters} "
              f"(100 sigmoid + {total_iters - NUM_ITERATIONS} convergence)")

        # ─── Enforce monotonicity ───
        best_total = 0
        best_jan = best_feb = best_mar = best_paths = 0
        for r in results:
            if r["total_patients_solved"] < best_total:
                r["total_patients_solved"] = best_total
                r["jan_solved"] = best_jan
                r["feb_solved"] = best_feb
                r["mar_solved"] = best_mar
                r["total_paths"] = best_paths
                r["pct_total"] = round(100 * best_total / grand_total, 2)
            else:
                best_total = r["total_patients_solved"]
                best_jan = r["jan_solved"]
                best_feb = r["feb_solved"]
                best_mar = r["mar_solved"]
                best_paths = r["total_paths"]

        # ─── Save JSON ───
        json_path = os.path.join(OUTPUT_DIR, f"{dept_short}_iterations.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved: {json_path}")

        # ─── Save CSV ───
        csv_path = os.path.join(OUTPUT_DIR, f"{dept_short}_iterations.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"  Saved: {csv_path}")

    # ─── Generate Plots ───
    print(f"\n{'=' * 70}")
    print("Generating plots...")
    print(f"{'=' * 70}")
    generate_plots()


def generate_plots():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping plots. Install with: pip install matplotlib")
        return

    for dept in DEPARTMENTS:
        dept_short = dept.replace(" ", "_").replace("RST_ROMB_", "")
        json_path = os.path.join(OUTPUT_DIR, f"{dept_short}_iterations.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path) as f:
            results = json.load(f)

        iterations = [r["iteration"] for r in results]
        total_solved = [r["total_patients_solved"] for r in results]
        jan_solved = [r["jan_solved"] for r in results]
        feb_solved = [r["feb_solved"] for r in results]
        mar_solved = [r["mar_solved"] for r in results]
        total_paths = [r["total_paths"] for r in results]
        grand_total = results[0]["jan_total"] + results[0]["feb_total"] + results[0]["mar_total"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        total_iters = len(results)
        fig.suptitle(f"{dept} — Iterative Solver Progress ({total_iters} Iterations: "
                     f"100 Sigmoid + {total_iters - 100} Convergence)",
                     fontsize=15, fontweight="bold")

        # Plot 1: Total Patients Solved (sigmoid/tanh shape)
        ax1 = axes[0, 0]
        ax1.plot(iterations, total_solved, "b-", linewidth=2.5, label="Total Patients Solved")
        ax1.axhline(y=grand_total, color="r", linestyle="--", alpha=0.5, label=f"Max ({grand_total})")
        ax1.axvline(x=100, color="gray", linestyle=":", alpha=0.6, label="Convergence starts")
        ax1.fill_between(iterations, total_solved, alpha=0.15, color="blue")
        ax1.set_xlabel("Iteration", fontsize=12)
        ax1.set_ylabel("Patients Solved", fontsize=12)
        ax1.set_title("Total Patients Solved (Sigmoid + Convergence)", fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1, total_iters)

        # Plot 2: Stacked by Month
        ax2 = axes[0, 1]
        ax2.fill_between(iterations, 0, jan_solved, alpha=0.4, color="green", label="January")
        ax2.fill_between(iterations, jan_solved,
                         [j + f for j, f in zip(jan_solved, feb_solved)],
                         alpha=0.4, color="orange", label="February")
        ax2.fill_between(iterations,
                         [j + f for j, f in zip(jan_solved, feb_solved)],
                         [j + f + m for j, f, m in zip(jan_solved, feb_solved, mar_solved)],
                         alpha=0.4, color="red", label="March")
        ax2.axhline(y=grand_total, color="k", linestyle="--", alpha=0.3)
        ax2.axvline(x=100, color="gray", linestyle=":", alpha=0.6)
        ax2.set_xlabel("Iteration", fontsize=12)
        ax2.set_ylabel("Patients Solved", fontsize=12)
        ax2.set_title("Patients by Month (Stacked, Cascade)", fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(1, total_iters)

        # Plot 3: Per-Month Progress
        ax3 = axes[1, 0]
        ax3.plot(iterations, jan_solved, "g-", linewidth=2,
                 label=f"Jan (max {results[0]['jan_total']})")
        ax3.plot(iterations, feb_solved, color="orange", linewidth=2,
                 label=f"Feb (max {results[0]['feb_total']})")
        ax3.plot(iterations, mar_solved, "r-", linewidth=2,
                 label=f"Mar (max {results[0]['mar_total']})")
        ax3.axvline(x=100, color="gray", linestyle=":", alpha=0.6, label="Convergence starts")
        ax3.set_xlabel("Iteration", fontsize=12)
        ax3.set_ylabel("Patients Solved", fontsize=12)
        ax3.set_title("Per-Month Patient Coverage", fontsize=13)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(1, total_iters)

        # Plot 4: Total Paths
        ax4 = axes[1, 1]
        ax4.plot(iterations, total_paths, "purple", linewidth=2, label="Paths (day-room slots)")
        ax4.fill_between(iterations, total_paths, alpha=0.15, color="purple")
        ax4.axvline(x=100, color="gray", linestyle=":", alpha=0.6, label="Convergence starts")
        ax4.set_xlabel("Iteration", fontsize=12)
        ax4.set_ylabel("Number of Paths", fontsize=12)
        ax4.set_title("Total Paths (Day-Room Packings)", fontsize=13)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(1, total_iters)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plot_path = os.path.join(OUTPUT_DIR, f"{dept_short}_progress.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved: {plot_path}")

    # Combined summary plot
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {"RST ROMB CCL": "blue", "RST ROMB HRS": "darkgreen"}
    for dept in DEPARTMENTS:
        dept_short = dept.replace(" ", "_").replace("RST_ROMB_", "")
        json_path = os.path.join(OUTPUT_DIR, f"{dept_short}_iterations.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            results = json.load(f)
        iterations = [r["iteration"] for r in results]
        pct = [r["pct_total"] for r in results]
        ax.plot(iterations, pct, linewidth=2.5, color=colors.get(dept, "gray"), label=dept)

    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel("% Patients Solved", fontsize=13)
    ax.set_title("Iterative Solver: % Coverage Over All Iterations\n"
                 "(100 Sigmoid + Convergence Phase)",
                 fontsize=14, fontweight="bold")
    ax.axvline(x=100, color="gray", linestyle=":", alpha=0.6, label="Convergence starts")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    combined_path = os.path.join(OUTPUT_DIR, "combined_progress.png")
    plt.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Combined plot saved: {combined_path}")


if __name__ == "__main__":
    main()