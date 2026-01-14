import json
import csv
import random
import os
import time
import datetime
import re
import hashlib
from collections import defaultdict

# Try to import OR-Tools Linear Solver
try:
    from ortools.linear_solver import pywraplp

    HAS_LP = True
except ImportError:
    HAS_LP = False
    print(">>> CRITICAL WARNING: 'ortools' not found. LP Solver will fail.")

# --- CONFIGURATION ---
INPUT_FILE = "informative_data.json"
OUTPUT_DIR = "mayo_lp_results"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

TARGET_PATHS = 5000  # As requested

# Dates
TARGET_MONTH = 1
POOL_MONTH = 2
YEAR = 2022
DAYS_IN_JAN = 31
MINUTES_PER_DAY = 1440
TURNOVER_TIME = 15

# Weights
WEIGHT_JAN = 100_000
WEIGHT_FEB = 100

# GA Settings
GA_POPULATION = 100
GA_GENERATIONS = 50


# ==========================================
# 1. DATA LOADING & PARSING
# ==========================================
class Patient:
    __slots__ = ['id', 'duration', 'room', 'compatible_rooms', 'surgeon', 'original_date', 'type', 'weight',
                 'department', 'original_day_idx', 'original_start', 'procedure_code']

    def __init__(self, pid, duration, room, compatible_rooms, surgeon, date_str, p_type, dept, day_idx, start_time,
                 proc_code):
        self.id = pid
        self.duration = int(duration)
        self.room = room
        self.compatible_rooms = compatible_rooms
        self.surgeon = surgeon
        self.original_date = date_str
        self.type = p_type
        self.department = dept
        self.original_day_idx = day_idx
        self.original_start = start_time
        self.procedure_code = proc_code
        self.weight = WEIGHT_JAN if p_type == 'JAN' else WEIGHT_FEB


def parse_time(t_str):
    if not t_str: return None
    try:
        t = datetime.datetime.strptime(t_str, "%H:%M:%S")
        return t.hour * 60 + t.minute
    except:
        return None


def extract_proc_code(proc_str):
    if not proc_str: return "UNKNOWN"
    match = re.search(r'\[(.*?)\]', proc_str)
    return match.group(1) if match else proc_str


def load_data():
    if not os.path.exists(INPUT_FILE):
        print(f">>> ERROR: {INPUT_FILE} not found.")
        return {}

    with open(INPUT_FILE, 'r') as f:
        raw_data = json.load(f)
    print(f">>> Loaded {len(raw_data)} raw records.")

    # 1. Learn Procedure -> Room Compatibility
    proc_room_map = defaultdict(set)
    for entry in raw_data:
        if entry.get("Discharge Location") != "RST MCH Saint Marys Campus": continue
        proc_str = entry.get("Scheduled Procedure")
        room = entry.get("Room")
        if proc_str and room:
            code = extract_proc_code(proc_str)
            proc_room_map[code].add(room)

    # 2. Build Patient Objects
    departments = defaultdict(list)

    for i, entry in enumerate(raw_data):
        if entry.get("Discharge Location") != "RST MCH Saint Marys Campus": continue

        date_str = entry.get("Date")
        if not date_str: continue
        try:
            dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except:
            continue

        # Calculate relative day index (0 = Jan 1st)
        day_idx = (dt.date() - datetime.date(YEAR, TARGET_MONTH, 1)).days

        if dt.month == TARGET_MONTH:
            p_type = 'JAN'
        elif dt.month == POOL_MONTH:
            p_type = 'FEB'
        else:
            continue

        in_time = parse_time(entry.get("In Proc Room"))
        out_time = parse_time(entry.get("Out Proc Room"))
        if in_time is None or out_time is None: continue

        duration = (out_time - in_time) if out_time >= in_time else (1440 - in_time) + out_time
        if duration <= 0: duration = 30

        dept = entry.get("OR Department")
        room = entry.get("Room")
        surgeon = entry.get("Lead Surgeon/Provider")
        proc_str = entry.get("Scheduled Procedure")

        if not dept or not room or not surgeon: continue

        proc_code = extract_proc_code(proc_str)
        # Compatibility: Rooms that have hosted this procedure code
        valid_rooms = list(proc_room_map.get(proc_code, [room]))

        p = Patient(i, duration, room, valid_rooms, surgeon, date_str, p_type, dept, day_idx, in_time, proc_code)
        departments[dept].append(p)

    return departments


# ==========================================
# 2. PATH MINING (GA)
# ==========================================
class PathNode:
    __slots__ = ['id', 'day', 'room', 'ops', 'pids', 'weight', 'doc_usage', 'count']

    def __init__(self, day, room, ops, p_map):
        self.day = day
        self.room = room
        self.ops = ops
        self.ops.sort(key=lambda x: x['start'])
        self.pids = tuple(sorted([op['pid'] for op in self.ops]))
        self.weight = 0
        self.count = len(self.pids)
        self.doc_usage = []

        for op in self.ops:
            pat = p_map[op['pid']]
            w = WEIGHT_JAN if pat.type == 'JAN' else WEIGHT_FEB
            self.weight += w
            self.doc_usage.append((pat.surgeon, op['start'], op['end']))

        self.id = hashlib.md5(f"{day}_{room}_{self.pids}".encode()).hexdigest()


def decode_schedule(permutation, p_map, days_range):
    # Simplified Greedy Decoder for Mining
    sched = defaultdict(list)
    room_avail = defaultdict(int)  # (day, room) -> end_time
    surgeon_avail = defaultdict(int)  # (day, surgeon) -> end_time

    extracted_nodes = []

    for pid in permutation:
        p = p_map[pid]

        # Try a few random days + original day
        days_to_try = [p.original_day_idx]
        other_days = list(range(days_range))
        if p.original_day_idx in other_days: other_days.remove(p.original_day_idx)
        days_to_try.extend(random.sample(other_days, min(2, len(other_days))))

        placed = False
        for d in days_to_try:
            if placed: break
            if not (0 <= d < days_range): continue

            # Try rooms
            candidates = list(p.compatible_rooms)
            random.shuffle(candidates)

            best_room = None
            best_start = float('inf')

            for r in candidates:
                # Earliest start in room (finish + turnover)
                r_ready = room_avail[(d, r)]
                if r_ready > 0: r_ready += TURNOVER_TIME

                # Earliest start for surgeon
                s_ready = surgeon_avail[(d, p.surgeon)]
                # (Assume surgeon turnover handled naturally by gaps or simplified here)

                start = max(r_ready, s_ready)
                if d == p.original_day_idx: start = max(start, p.original_start)

                if start + p.duration <= MINUTES_PER_DAY:
                    if start < best_start:
                        best_start = start
                        best_room = r

            if best_room:
                end = best_start + p.duration
                room_avail[(d, best_room)] = end
                surgeon_avail[(d, p.surgeon)] = end
                sched[(d, best_room)].append({'pid': pid, 'start': best_start, 'end': end})
                placed = True

    # Convert to PathNodes
    for (d, r), ops in sched.items():
        node = PathNode(d, r, ops, p_map)
        extracted_nodes.append(node)

    return extracted_nodes


def run_mining_batch(p_ids, p_map, days_range):
    base = list(p_ids)
    # Weighted shuffle: prioritize JAN
    base.sort(key=lambda pid: p_map[pid].weight * random.uniform(0.8, 1.2), reverse=True)

    pop = []
    for _ in range(GA_POPULATION):
        ind = list(base)
        # Small shuffle
        if len(ind) > 2:
            for _ in range(10):
                i, j = random.randint(0, len(ind) - 1), random.randint(0, len(ind) - 1)
                ind[i], ind[j] = ind[j], ind[i]
        pop.append(ind)

    all_new_nodes = []

    for gen in range(GA_GENERATIONS):
        # Decode top individual
        nodes = decode_schedule(pop[0], p_map, days_range)
        all_new_nodes.extend(nodes)

        # Simple Evolution
        next_pop = []
        pop.sort(key=lambda x: random.random())  # Random mix
        while len(next_pop) < GA_POPULATION:
            p1 = pop[random.randint(0, len(pop) - 1)]
            child = list(p1)
            # Mutation
            if random.random() < 0.3 and len(child) > 2:
                i, j = random.sample(range(len(child)), 2)
                child[i], child[j] = child[j], child[i]
            next_pop.append(child)
        pop = next_pop

    return all_new_nodes


# ==========================================
# 3. LP RELAXATION SOLVER
# ==========================================
def solve_lp_relaxation_mayo(bank, p_map):
    if not HAS_LP: return [], 0
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver: return [], 0

    print("   [LP] Building Model...")
    x_vars = []
    all_nodes = []

    room_cons = defaultdict(lambda: solver.Constraint(0, 1, ''))
    patient_cons = defaultdict(lambda: solver.Constraint(0, 1, ''))

    # 1. Variables
    for key, nodes in bank.items():
        for node in nodes:
            x = solver.NumVar(0.0, 1.0, f'x_{node.id}')
            x_vars.append(x)
            all_nodes.append(node)

            # Room Constraint
            room_cons[key].SetCoefficient(x, 1)

            # Patient Constraint
            for pid in node.pids:
                patient_cons[pid].SetCoefficient(x, 1)

    # 2. Doctor Constraints (Sweep Line)
    print("   [LP] Adding Doctor Constraints...")
    doc_usage_map = defaultdict(list)
    for idx, node in enumerate(all_nodes):
        for (doc, s, e) in node.doc_usage:
            doc_usage_map[(node.day, doc)].append((s, e, idx))

    doc_cons_count = 0
    for key, intervals in doc_usage_map.items():
        doc, day = key
        # Create events
        events = []
        for s, e, idx in intervals:
            events.append((s, 1, idx))
            events.append((e, -1, idx))
        events.sort(key=lambda x: (x[0], x[1]))

        current_active = set()
        for t, type, idx in events:
            if type == 1:
                current_active.add(idx)
                if len(current_active) >= 2:
                    ct = solver.Constraint(0, 1, f'd_c_{doc_cons_count}')
                    for act_idx in current_active:
                        ct.SetCoefficient(x_vars[act_idx], 1)
                    doc_cons_count += 1
            else:
                if idx in current_active:
                    current_active.remove(idx)

    print(f"   [LP] Stats: {len(x_vars)} Vars, {len(room_cons)} Rooms, {len(patient_cons)} Pts, {doc_cons_count} Doc")

    # 3. Objective
    objective = solver.Objective()
    for i, node in enumerate(all_nodes):
        # We maximize weighted score (Jan priority)
        objective.SetCoefficient(x_vars[i], node.weight)
    objective.SetMaximization()

    print("   [LP] Solving...")
    solver.Solve()

    results = []
    for i, x in enumerate(x_vars):
        val = x.solution_value()
        if val > 1e-4:
            results.append((all_nodes[i], val))

    return results, objective.Value()


# ==========================================
# 4. MAIN
# ==========================================
def main():
    print("=== ARGOS MAYO: LP Relaxation Solver ===")

    # 1. Load
    data_by_dept = load_data()

    for dept, patients in data_by_dept.items():
        print(f"\n>>> Analyzing Department: {dept}")

        p_map = {p.id: p for p in patients}
        p_ids = [p.id for p in patients]

        # 2. Mine Paths
        bank = defaultdict(list)
        seen = set()

        print(f"   [Mining] Target: {TARGET_PATHS} paths...")
        start_mining = time.time()

        while len(seen) < TARGET_PATHS:
            new_nodes = run_mining_batch(p_ids, p_map, DAYS_IN_JAN)
            added = 0
            for n in new_nodes:
                if n.id not in seen:
                    bank[(n.day, n.room)].append(n)
                    seen.add(n.id)
                    added += 1

            if len(seen) % 500 < 50:
                print(f"   ... Mined {len(seen)}/{TARGET_PATHS}")

            if added == 0 and len(seen) > 100:
                print("   (Miner saturated early)")
                break

        print(f"   [Mining] Done. {len(seen)} paths in {time.time() - start_mining:.1f}s")

        # 3. Solve LP
        fractional_nodes, obj_val = solve_lp_relaxation_mayo(bank, p_map)

        # 4. Analysis
        total_jan_score = 0
        total_feb_score = 0
        jan_count = 0
        feb_count = 0

        # Calculate fractional coverage
        patient_coverage = defaultdict(float)

        csv_rows = []

        for node, frac in fractional_nodes:
            # Contribution to objective
            contrib = frac * node.weight

            # Count Jan vs Feb
            j_in_node = sum(1 for pid in node.pids if p_map[pid].type == 'JAN')
            f_in_node = sum(1 for pid in node.pids if p_map[pid].type == 'FEB')

            jan_count += (frac * j_in_node)
            feb_count += (frac * f_in_node)

            # CSV Data
            p_str = ";".join([f"{pid}({p_map[pid].type})" for pid in node.pids])
            csv_rows.append([dept, node.day, node.room, f"{frac:.6f}", p_str, f"{contrib:.2f}"])

        print(f"\n   >>> LP RESULTS for {dept} <<<")
        print(f"   Objective Score: {obj_val:,.2f}")
        print(f"   Jan Patients (Fractional): {jan_count:.4f}")
        print(f"   Feb Patients (Fractional): {feb_count:.4f}")

        # Save
        fname = os.path.join(OUTPUT_DIR, f"lp_relax_{dept.replace(' ', '_')}.csv")
        with open(fname, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["Department", "Day", "Room", "Fraction", "Patients", "Score_Contrib"])
            w.writerows(csv_rows)
        print(f"   Saved to {fname}")


if __name__ == "__main__":
    main()