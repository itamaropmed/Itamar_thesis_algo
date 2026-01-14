import json
import csv
import random
import os
import time
import hashlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

# Try to import OR-Tools Linear Solver
try:
    from ortools.linear_solver import pywraplp

    HAS_LP = True
except ImportError:
    HAS_LP = False
    print(">>> CRITICAL WARNING: 'ortools' not found. LP Solver will fail.")

# --- CONFIGURATION ---
INPUT_FILE = "large_setup_250.json"
OUTPUT_DIR = "../schedules_ARGOS_LP"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# Target
TARGET_PATHS = 3000

# GA Settings (From your code)
GA_POPULATION = 200
GA_GENERATIONS = 100

# Constraints
ROOMS = 6
DAYS = 5
DAY_LIMIT = 600


# ==========================================
# 1. DATA LOADING
# ==========================================
class Patient:
    __slots__ = ['id', 'duration', 'rooms', 'docs', 'type']

    def __init__(self, d):
        self.id = d["id"]
        self.duration = int(d["duration"])
        self.rooms = list(d["compatible_rooms"])
        self.docs = list(d["compatible_doctors"])
        self.type = d.get("type", "Standard")


def load_data():
    if not os.path.exists(INPUT_FILE):
        print(f">>> ERROR: Input file '{INPUT_FILE}' not found.")
        return [], {}
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    print(f">>> Data Loaded: {len(data['patients'])} patients.")
    return [Patient(p) for p in data['patients']], data['meta']


PATIENTS, META = load_data()
PATIENT_MAP = {p.id: p for p in PATIENTS}
PATIENT_IDS = [p.id for p in PATIENTS]


# ==========================================
# 2. GRAPH BANK
# ==========================================
class PathNode:
    __slots__ = ['id', 'day', 'room', 'ops', 'pids', 'count', 'doc_usage']

    def __init__(self, day, room, ops):
        self.day = day
        self.room = room
        self.ops = ops
        self.ops.sort(key=lambda x: x['start'])
        self.pids = tuple(sorted([op['pid'] for op in self.ops]))
        self.count = len(self.pids)
        self.doc_usage = []
        for op in ops:
            self.doc_usage.append((op['doc'], op['start'], op['end']))
        self.id = hashlib.md5(f"{day}_{room}_{self.pids}".encode()).hexdigest()


BANK = defaultdict(list)
SEEN_HASHES = set()


def add_to_bank(day, room, ops):
    if not ops: return False
    node = PathNode(day, room, ops)
    if node.id not in SEEN_HASHES:
        BANK[(day, room)].append(node)
        SEEN_HASHES.add(node.id)
        return True
    return False


# ==========================================
# 3. MINING LOGIC (Exact Copy + Safety)
# ==========================================
def is_free(intervals, start, duration):
    end = start + duration
    for (s, e) in intervals:
        if not (end <= s or start >= e): return False
    return True


def decode_schedule(permutation):
    room_sched = [[[] for _ in range(ROOMS)] for _ in range(DAYS)]
    doc_sched = [[[] for _ in range(15)] for _ in range(DAYS)]
    extracted_paths = defaultdict(list)
    count = 0

    for pid in permutation:
        p = PATIENT_MAP[pid]
        placed = False
        days = list(range(DAYS));
        random.shuffle(days)
        for d in days:
            if placed: break
            rooms = list(p.rooms);
            random.shuffle(rooms)
            for r in rooms:
                if placed: break
                candidates = [0]
                for (s, e) in room_sched[d][r]: candidates.append(e)
                candidates.sort()
                for start in candidates:
                    if start + p.duration > DAY_LIMIT: break
                    if not is_free(room_sched[d][r], start, p.duration): continue

                    valid_doc = -1
                    random.shuffle(p.docs)
                    for doc in p.docs:
                        if is_free(doc_sched[d][doc], start, p.duration):
                            valid_doc = doc;
                            break

                    if valid_doc != -1:
                        end = start + p.duration
                        room_sched[d][r].append((start, end))
                        doc_sched[d][valid_doc].append((start, end))
                        extracted_paths[(d, r)].append({'pid': pid, 'start': start, 'end': end, 'doc': valid_doc})
                        count += 1
                        placed = True
                        break
    return extracted_paths, count


def run_weighted_ga_batch(weights):
    if not PATIENT_IDS: return None  # Handle empty data case

    base = list(PATIENT_IDS)
    base.sort(key=lambda pid: weights.get(pid, 1.0) * random.uniform(0.8, 1.2), reverse=True)

    pop = []
    for _ in range(GA_POPULATION):
        ind = list(base)
        # Safety Check: Only swap if enough items
        if len(ind) >= 2:
            for _ in range(15):
                i, j = random.randint(0, len(ind) - 1), random.randint(0, len(ind) - 1)
                ind[i], ind[j] = ind[j], ind[i]
        pop.append(ind)

    best_sched = None
    best_score = -1

    for gen in range(GA_GENERATIONS):
        scores = []
        for chrom in pop:
            sched, count = decode_schedule(chrom)
            scores.append((count, chrom, sched))
            if count > best_score:
                best_score = count
                best_sched = sched

        scores.sort(key=lambda x: x[0], reverse=True)

        # Elitism
        cutoff = int(GA_POPULATION * 0.25)
        next_pop = [x[1] for x in scores[:cutoff]]

        while len(next_pop) < GA_POPULATION:
            p1 = random.choice(scores[:cutoff])[1]
            p2 = random.choice(scores[:cutoff])[1]

            # Crossover
            cut = random.randint(0, len(PATIENT_IDS))
            child = p1[:cut] + [x for x in p2 if x not in p1[:cut]]

            # Mutation (Safe)
            if random.random() < 0.3 and len(child) >= 2:
                # FIX: Verify length before sample
                i, j = random.sample(range(len(child)), 2)
                child[i], child[j] = child[j], child[i]
            next_pop.append(child)
        pop = next_pop

    return best_sched


# ==========================================
# 4. LP RELAXATION SOLVER (GLOP)
# ==========================================
def solve_lp_relaxation(time_limit=300):
    if not HAS_LP: return None, 0

    # Create the linear solver with the GLOP backend (Simplex)
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        print(">>> Error: GLOP solver could not be initialized.")
        return None, 0

    all_nodes = []
    x_vars = []

    # 1. Variables: 0 <= x <= 1 (Continuous)
    # 2. Room Constraints: Sum(x) <= 1 per room-day
    room_cons = defaultdict(lambda: solver.Constraint(0, 1, ''))

    # 3. Patient Constraints: Sum(x) <= 1 per patient
    patient_cons = defaultdict(lambda: solver.Constraint(0, 1, ''))

    print("   [LP] Building Variables and Basic Constraints...")
    for (d, r), nodes in BANK.items():
        for node in nodes:
            x = solver.NumVar(0.0, 1.0, f'x_{node.id}')
            x_vars.append(x)
            all_nodes.append(node)

            # Add to Room Constraint
            room_cons[(d, r)].SetCoefficient(x, 1)

            # Add to Patient Constraints
            for pid in node.pids:
                patient_cons[pid].SetCoefficient(x, 1)

    # 4. Doctor Constraints (Sweep Line for Overlaps)
    # We must ensure that Sum(x_i) <= 1 for all nodes overlapping on a doctor
    print("   [LP] Building Doctor Overlap Constraints (Sweep Line)...")

    # Group nodes by (Day, Doc)
    doc_usage_map = defaultdict(list)
    for idx, node in enumerate(all_nodes):
        for (doc, s, e) in node.doc_usage:
            doc_usage_map[(node.day, doc)].append((s, e, idx))

    doc_cons_count = 0
    for (day, doc), intervals in doc_usage_map.items():
        if not intervals: continue

        # Create events: (time, type, node_index)
        # type: #+1 for start, -1 for end
        events = []
        for s, e, idx in intervals:
            events.append((s, 1, idx))
            events.append((e, -1, idx))

        # Sort events: time asc. If time equal, process START before END (to catch single-point overlaps?
        # Usually for RCPSP [s,e) we process END before START to allow back-to-back.
        # Let's use strict: e=-1 comes before s=1 if times are equal.
        events.sort(key=lambda x: (x[0], x[1]))

        current_active = set()

        for t, type, idx in events:
            if type == 1:  # Start
                current_active.add(idx)
                # If we have >= 2 active nodes, this implies a conflict clique.
                # We add a constraint: Sum(x for x in current_active) <= 1
                if len(current_active) >= 2:
                    ct = solver.Constraint(0, 1, f'doc_{doc}_d{day}_t{t}')
                    for active_idx in current_active:
                        ct.SetCoefficient(x_vars[active_idx], 1)
                    doc_cons_count += 1
            else:  # End
                if idx in current_active:
                    current_active.remove(idx)

    print(f"   [LP] Model Stats: {len(x_vars)} Variables")
    print(f"   [LP] Constraints: {len(room_cons)} Rooms + {len(patient_cons)} Patients + {doc_cons_count} Doc Overlaps")

    # 5. Objective: Maximize Weighted Sum
    objective = solver.Objective()
    for i, node in enumerate(all_nodes):
        objective.SetCoefficient(x_vars[i], node.count)  # node.count is # of patients
    objective.SetMaximization()

    # 6. Solve
    print("   [LP] Solving with GLOP...")
    solver.SetTimeLimit(time_limit * 1000)
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        score = objective.Value()

        results = []
        for i, x in enumerate(x_vars):
            val = x.solution_value()
            if val > 1e-4:  # Filter almost-zero
                results.append((all_nodes[i], val))

        return results, score
    else:
        return [], 0


# ==========================================
# 5. MAIN
# ==========================================
def main():
    print("=== ARGOS V41: LP Relaxation Upper Bound ===")

    if len(PATIENTS) == 0:
        print(">>> ERROR: No patient data loaded. Exiting.")
        return

    # 1. Mine Paths
    print(f"   [Mining] Targeting {TARGET_PATHS} paths...")
    weights = {p.id: 1.0 for p in PATIENTS}

    start_time = time.time()

    while len(SEEN_HASHES) < TARGET_PATHS:
        sched = run_weighted_ga_batch(weights)
        added_count = 0
        if sched:
            for (d, r), ops in sched.items():
                if ops:
                    if add_to_bank(d, r, ops): added_count += 1

        if len(SEEN_HASHES) % 500 == 0:
            print(f"   ... Mined {len(SEEN_HASHES)}/{TARGET_PATHS}")

        # Break if stuck (optional safety)
        if added_count == 0 and len(SEEN_HASHES) > TARGET_PATHS * 0.99:
            break

    print(f"   [Mining] Finished in {time.time() - start_time:.2f}s. Bank Size: {len(SEEN_HASHES)}")

    # 2. Solve LP
    print("   [Solving] Calculating LP Relaxation Boundary...")
    fractional_nodes, lp_score = solve_lp_relaxation()

    print(f"\n   >>> LP RELAXATION BOUND: {lp_score:.4f} Patients")
    print(f"   >>> (Theoretical Maximum with fractional splitting)")

    # 3. Save Results
    filename_csv = os.path.join(OUTPUT_DIR, "lp_relaxation_schedule.csv")
    with open(filename_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["NodeID", "Day", "Room", "Fraction", "Patients", "Weighted_Contrib"])
        for node, frac in fractional_nodes:
            contrib = frac * node.count
            pids = ";".join(map(str, sorted(node.pids)))
            writer.writerow([node.id, node.day, node.room, f"{frac:.6f}", pids, f"{contrib:.6f}"])

    print(f"   Saved details to {filename_csv}")


if __name__ == "__main__":
    main()