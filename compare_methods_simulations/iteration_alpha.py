import json
import csv
import random
import os
import time
import hashlib
import itertools
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

# Try to import OR-Tools
try:
    from ortools.sat.python import cp_model

    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False
    print(">>> CRITICAL WARNING: 'ortools' not found. Solver will fail.")

# --- CONFIGURATION ---
INPUT_FILE = "large_setup_250.json"
OUTPUT_DIR = "schedules_ARGO_hard"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# Exact Milestones
LAYER_TARGETS = [120, 240, 510, 1020, 2010, 3000, 4000, 5000]

# --- UPGRADE 1: HEAVIER GA INVESTMENT ---
GA_POPULATION = 200  # Was 40
GA_GENERATIONS = 100  # Was 15

# --- UPGRADE 2: HIGHER LIMIT ---
MAX_DENOMINATOR_LIMIT = 10 ** 7

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
        return [], {}
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    return [Patient(p) for p in data['patients']], data['meta']


PATIENTS, META = load_data()
PATIENT_MAP = {p.id: p for p in PATIENTS}
PATIENT_IDS = [p.id for p in PATIENTS]


# ==========================================
# 2. PARALLEL TOTAL SEARCH SPACE (Unlimited Width)
# ==========================================
def count_paths_for_room(room_idx, patients_data, day_limit, limit):
    """
    Worker function to count paths for a single room via DFS.
    REMOVED TIME LIMIT. REMOVED BRANCHING LIMIT.
    Runs until 'limit' is hit or tree is fully explored.
    """
    candidates = [p for p in patients_data if room_idx in p['compatible_rooms'] and p['duration'] <= day_limit]

    count = 0
    # Stack: (current_duration, used_ids_set)
    stack = [(0, set())]

    while stack:
        # Check hard limit only
        if count >= limit: return limit

        cur_dur, used = stack.pop()

        # Valid path found
        if cur_dur > 0: count += 1

        remaining = day_limit - cur_dur

        # Find valid next steps (Optimized list comp)
        potential = [p for p in candidates if p['id'] not in used and p['duration'] <= remaining]

        # --- UPGRADE 3: UNLIMITED BRANCHING ---
        # We check all potential next steps, not just top 40.
        # This ensures we truly see "all possibilities".
        for p in potential:
            new_used = used.copy()
            new_used.add(p['id'])
            stack.append((cur_dur + p['duration'], new_used))

    return count


def calculate_total_parallel():
    print(f"\n>>> CALCULATING TOTAL SEARCH SPACE (Parallel, Limit {MAX_DENOMINATOR_LIMIT})...")
    print("    (This may take a while as it is doing a deep exhaustive count...)")

    p_data = [{'id': p.id, 'duration': p.duration, 'compatible_rooms': p.rooms} for p in PATIENTS]
    total_count = 0

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(count_paths_for_room, r, p_data, DAY_LIMIT, MAX_DENOMINATOR_LIMIT): r for r in
                   range(ROOMS)}

        for future in futures:
            r = futures[future]
            try:
                c = future.result()
                print(f"   -> Room {r} Valid Combinations Found: {c}")
                total_count += c
            except Exception as e:
                print(f"   -> Room {r} Error: {e}")

    grand_total = total_count * DAYS
    print(f"   >>> Total Hypergraph Nodes (Est): {grand_total}")
    return grand_total


# ==========================================
# 3. GRAPH BANK
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
# 4. WEIGHTED GA MINER (Aggressive)
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
    base = list(PATIENT_IDS)
    base.sort(key=lambda pid: weights.get(pid, 1.0) * random.uniform(0.8, 1.2), reverse=True)

    pop = []
    for _ in range(GA_POPULATION):
        ind = list(base)
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

        # Elitism: Keep top 25%
        cutoff = int(GA_POPULATION * 0.25)
        next_pop = [x[1] for x in scores[:cutoff]]

        while len(next_pop) < GA_POPULATION:
            p1 = random.choice(scores[:cutoff])[1]
            p2 = random.choice(scores[:cutoff])[1]
            cut = random.randint(0, len(PATIENT_IDS))
            child = p1[:cut] + [x for x in p2 if x not in p1[:cut]]

            # Mutation (Higher rate for diversity)
            if random.random() < 0.3:
                i, j = random.sample(range(len(child)), 2)
                child[i], child[j] = child[j], child[i]
            next_pop.append(child)
        pop = next_pop

    return best_sched


# ==========================================
# 5. MASTER SOLVER (Hard Optimization)
# ==========================================
def solve_master_problem(time_limit=300):  # 5 Minutes
    """
    Hypergraph Solver with Aggressive Optimization Settings.
    """
    if not HAS_ORTOOLS: return [], 0
    model = cp_model.CpModel()
    all_nodes = []
    x_vars = []
    slot_map = defaultdict(list)
    patient_map = defaultdict(list)
    doc_map = defaultdict(list)

    # 1. Build Hypergraph
    for (d, r), nodes in BANK.items():
        for node in nodes:
            x = model.NewBoolVar(f'x_{node.id}')
            x_vars.append(x)
            all_nodes.append(node)

            slot_map[(d, r)].append(x)
            for pid in node.pids: patient_map[pid].append(x)
            for (doc, s, e) in node.doc_usage:
                iv = model.NewOptionalFixedSizeIntervalVar(s, e - s, x, f'd_{doc}')
                doc_map[(d, doc)].append(iv)

    # 2. Add Constraints
    for vars in slot_map.values(): model.Add(sum(vars) <= 1)
    for vars in patient_map.values(): model.Add(sum(vars) <= 1)
    for intervals in doc_map.values(): model.AddNoOverlap(intervals)

    # 3. Objective: Maximize Patients
    coeffs = [n.count for n in all_nodes]
    model.Maximize(cp_model.LinearExpr.WeightedSum(x_vars, coeffs))

    # 4. Configure Solver for "Hard" Optimization
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 16  # Use all cores
    solver.parameters.linearization_level = 2  # Aggressive Linearization
    solver.parameters.symmetry_level = 2  # Aggressive Symmetry Breaking

    # Optional: Print search progress
    # solver.parameters.log_search_progress = True

    print(f"   [Solver] Optimizing {len(x_vars)} variables with {len(patient_map) + len(doc_map)} constraints...")

    status = solver.Solve(model)
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        score = int(solver.ObjectiveValue())
        selected = [all_nodes[i] for i, x in enumerate(x_vars) if solver.Value(x)]
        return selected, score
    return [], 0


# ==========================================
# 6. MAIN EXECUTION
# ==========================================
def save_iteration_schedule(nodes, score, bank_size, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    nodes.sort(key=lambda n: (n.day, n.room))
    with open(filepath, 'w', newline='') as f:
        f.write(f"Metric,Value\nTotal Assigned,{score}\nBank Size,{bank_size}\n,\n")
        f.write("--- OPTIMIZED SCHEDULE ---\n")
        writer = csv.writer(f)
        writer.writerow(['Room ID', 'Patient IDs'])
        for node in nodes:
            pids_str = ";".join(map(str, sorted(node.pids)))
            writer.writerow([node.room, pids_str])


def main():
    print("=== ARGO: Hard Mode (Deep Mining + Heavy Optimization) ===")

    # 1. Denominator (Parallel + Unlimited)
    total_space = calculate_total_parallel()

    global_best = 0
    weights = {p.id: 1.0 for p in PATIENTS}
    results = []

    for target in LAYER_TARGETS:
        print(f"\n[TARGET: {target} PATHS]")

        # 2. Mine Paths (Heavier GA)
        while len(SEEN_HASHES) < target:
            sched = run_weighted_ga_batch(weights)
            added_count = 0
            if sched:
                for (d, r), ops in sched.items():
                    if ops:
                        if add_to_bank(d, r, ops): added_count += 1

            # Saturated Check
            if added_count == 0 and len(SEEN_HASHES) > target * 0.98:
                print("   (Miner Saturated)")
                break

        # 3. Solve (Heavier Solver)
        # Increased time limit to 300s to "look harder"
        selected, score = solve_master_problem(time_limit=300)

        frac = len(SEEN_HASHES) / total_space if total_space > 0 else 0
        print(f"   [ARGO-{target}] Succeeded: {score} | Frac: {frac:.10f}")

        # Save
        filename = f"schedule_{target}_paths.csv"
        save_iteration_schedule(selected, score, len(SEEN_HASHES), filename)

        if score > global_best: global_best = score

        results.append({
            "Method": f"ARGO-{target}",
            "Paths": len(SEEN_HASHES),
            "Fraction": f"{frac:.10f}",
            "Succeeded": score,
            "Total": 250
        })

        # Feedback
        covered = set()
        for node in selected: covered.update(node.pids)
        missing = [p for p in PATIENT_IDS if p not in covered]
        print(f"   [Feedback] Missing {len(missing)} patients. Boosting weights.")
        weights = {p: 1.0 for p in PATIENT_IDS}
        for m in missing: weights[m] = 100.0

    # Summary
    res_file = os.path.join(OUTPUT_DIR, "algorithm_results_ARGO.csv")
    with open(res_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "Paths", "Fraction", "Succeeded", "Total"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nFinal Best Score: {global_best}")
    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()