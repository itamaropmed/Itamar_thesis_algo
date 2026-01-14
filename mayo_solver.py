import hashlib
import json
import csv
import random
import os
import time
import copy
import datetime
import re
from collections import defaultdict

# Try to import OR-Tools
try:
    from ortools.sat.python import cp_model

    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False
    print(">>> CRITICAL WARNING: 'ortools' not found. Solver will fail.")

# Try to import TQDM
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print(">>> WARNING: 'tqdm' not found. Progress bars will be disabled.")


    # Dummy wrapper if tqdm missing
    def tqdm(iterable, **kwargs):
        return iterable

# --- CONFIGURATION ---
INPUT_FILE = "informative_data.json"
OUTPUT_CSV = "schedule_final.csv"
OUTPUT_JSON = "schedule_final_soft_push.json"
SUMMARY_FILE = "schedule_summary.csv"

# Iterations (Number of Full Schedule Runs)
SCHEDULE_BATCHES = [5, 10, 26]

# Constraints
TARGET_MONTH = 1
POOL_MONTH = 2
YEAR = 2022
DAYS_IN_JAN = 31
MINUTES_PER_DAY = 1440
TURNOVER_TIME = 15

# GA Settings
POPULATION_SIZE = 100
GENERATIONS = 30  # As requested
ELITISM_RATE = 0.1
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1

# Weights
WEIGHT_JAN = 10_000
WEIGHT_FEB = 100


# ==========================================
# 1. DATA LOADING & MAPPING
# ==========================================
class Patient:
    __slots__ = ['id', 'duration', 'compatible_rooms', 'surgeon', 'original_date', 'type', 'weight', 'department',
                 'original_day_idx', 'original_start', 'procedure_code']

    def __init__(self, pid, duration, compatible_rooms, surgeon, date_str, p_type, dept, day_idx, start_time,
                 proc_code):
        self.id = pid
        self.duration = int(duration)
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
    if not os.path.exists(INPUT_FILE): return {}
    with open(INPUT_FILE, 'r') as f:
        raw_data = json.load(f)
    print(f">>> Loaded {len(raw_data)} raw records.")

    # Map Procedures -> Rooms
    proc_room_map = defaultdict(set)
    for entry in raw_data:
        if entry.get("Discharge Location") != "RST MCH Saint Marys Campus": continue
        proc_str = entry.get("Scheduled Procedure")
        room = entry.get("Room")
        if proc_str and room:
            code = extract_proc_code(proc_str)
            proc_room_map[code].add(room)

    print(f"    [Mapping] Mapped {len(proc_room_map)} procedures.")

    departments = defaultdict(list)
    for i, entry in enumerate(raw_data):
        if entry.get("Discharge Location") != "RST MCH Saint Marys Campus": continue
        date_str = entry.get("Date")
        if not date_str: continue
        try:
            dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except:
            continue

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
        valid_rooms = list(proc_room_map.get(proc_code, [room]))

        p = Patient(i, duration, valid_rooms, surgeon, date_str, p_type, dept, day_idx, in_time, proc_code)
        departments[dept].append(p)
    return departments


def learn_constraints(patients):
    jan_p = [p for p in patients if p.type == 'JAN' and 0 <= p.original_day_idx < DAYS_IN_JAN]
    jan_p.sort(key=lambda x: x.original_start)

    room_turnover = defaultdict(lambda: TURNOVER_TIME)
    surgeon_turnover = defaultdict(lambda: 0)

    s_history = defaultdict(list)
    for p in jan_p: s_history[(p.original_day_idx, p.surgeon)].append(p)

    for _, p_list in s_history.items():
        for i in range(len(p_list) - 1):
            gap = p_list[i + 1].original_start - (p_list[i].original_start + p_list[i].duration)
            if gap < 0:
                surgeon_turnover[p_list[i].surgeon] = min(surgeon_turnover[p_list[i].surgeon], gap)

    print(f"    [Learner] {len([k for k, v in surgeon_turnover.items() if v < 0])} surgeons with historical overlaps.")
    return room_turnover, surgeon_turnover


# ==========================================
# 3. HARTMANN GA ENGINE
# ==========================================
class Individual:
    def __init__(self, gene, p_map):
        self.gene = gene
        self.p_map = p_map
        self.score = 0
        self.jan_count = 0
        self.feb_count = 0
        self.schedule = {}


def decode_sgs_flexible(ind, room_turnover, surgeon_turnover):
    room_avail = defaultdict(int)
    surgeon_avail = defaultdict(int)
    schedule_out = defaultdict(list)
    score = 0
    jan = 0
    feb = 0

    for pid in ind.gene:
        p = ind.p_map[pid]

        days_to_try = []
        if p.type == 'JAN' and 0 <= p.original_day_idx < DAYS_IN_JAN:
            days_to_try.append(p.original_day_idx)

        other_days = list(range(DAYS_IN_JAN))
        if days_to_try: other_days.remove(days_to_try[0])
        days_to_try.extend(other_days)

        s_gap = surgeon_turnover[p.surgeon]

        placed = False
        for d in days_to_try:
            if placed: break

            best_room = None
            best_start = float('inf')

            candidates = list(p.compatible_rooms)
            random.shuffle(candidates)

            for r in candidates:
                r_gap = room_turnover[r]
                r_ready = room_avail[(d, r)]
                s_ready = surgeon_avail[(d, p.surgeon)]

                start_r = r_ready + r_gap if r_ready > 0 else 0
                start_s = s_ready + s_gap if s_ready > 0 else 0
                start = max(start_r, start_s)

                if start < best_start:
                    best_start = start
                    best_room = r

            if best_room and best_start + p.duration <= MINUTES_PER_DAY:
                end = best_start + p.duration
                room_avail[(d, best_room)] = end
                surgeon_avail[(d, p.surgeon)] = end
                schedule_out[(d, best_room)].append({'pid': pid, 'start': best_start, 'end': end})
                score += p.weight
                if p.type == 'JAN':
                    jan += 1
                else:
                    feb += 1
                placed = True

    ind.score = score
    ind.jan_count = jan
    ind.feb_count = feb
    ind.schedule = schedule_out


def run_hartmann_ga(jan_p, feb_p, p_map, room_gap, s_gap, run_idx):
    jan_p.sort(key=lambda x: (x.original_day_idx, x.original_start))
    seed_gene = [p.id for p in jan_p]

    population = []
    for _ in range(POPULATION_SIZE):
        gene = list(seed_gene)
        feb_sample = random.sample(feb_p, min(len(feb_p), 50))
        gene.extend([p.id for p in feb_sample])

        # Mutation: Slight shuffle for all except the first pure seed
        if len(population) > 0:
            random.shuffle(gene)

        ind = Individual(gene, p_map)
        decode_sgs_flexible(ind, room_gap, s_gap)
        population.append(ind)

    # TQDM Progress Bar
    iterator = tqdm(range(GENERATIONS), desc=f"GA Run #{run_idx}", leave=False) if HAS_TQDM else range(GENERATIONS)

    for gen in iterator:
        population.sort(key=lambda x: x.score, reverse=True)
        next_pop = population[:int(POPULATION_SIZE * ELITISM_RATE)]

        while len(next_pop) < POPULATION_SIZE:
            p1 = random.choice(population[:POPULATION_SIZE // 2])
            p2 = random.choice(population[:POPULATION_SIZE // 2])

            cut = random.randint(0, len(p1.gene))
            child_gene = p1.gene[:cut] + [x for x in p2.gene if x not in p1.gene[:cut]]

            if random.random() < MUTATION_RATE:
                if len(child_gene) > 2:
                    i, j = random.sample(range(len(child_gene)), 2)
                    child_gene[i], child_gene[j] = child_gene[j], child_gene[i]

            child = Individual(child_gene, p_map)
            decode_sgs_flexible(child, room_gap, s_gap)
            next_pop.append(child)
        population = next_pop

    return population[0]


# ==========================================
# 4. HYPERGRAPH SOLVER
# ==========================================
class PathNode:
    __slots__ = ['id', 'day', 'room', 'ops', 'pids', 'weight', 'surgeon_usage', 'is_historical']

    def __init__(self, day, room, ops, patients_map, is_historical=False):
        self.day = day
        self.room = room
        self.ops = ops
        self.ops.sort(key=lambda x: x['start'])
        self.pids = tuple(sorted([op['pid'] for op in self.ops]))
        self.weight = sum(patients_map[pid].weight for pid in self.pids)
        self.is_historical = is_historical
        self.surgeon_usage = []
        for op in self.ops:
            p = patients_map[op['pid']]
            self.surgeon_usage.append((p.surgeon, op['start'], op['end']))
        self.id = hashlib.md5(f"{day}_{room}_{self.pids}".encode()).hexdigest()


def solve_hypergraph(bank, p_map, surgeon_turnover, time_limit=120):
    if not HAS_ORTOOLS: return []

    model = cp_model.CpModel()
    x_vars = []
    nodes = []

    day_room_cons = defaultdict(list)
    patient_cons = defaultdict(list)
    surgeon_intervals = defaultdict(list)

    for key, n_list in bank.items():
        for node in n_list:
            x = model.NewBoolVar(f'x_{node.id}')
            if node.is_historical: model.AddHint(x, 1)
            x_vars.append(x)
            nodes.append(node)

            day_room_cons[key].append(x)
            for pid in node.pids: patient_cons[pid].append(x)

            day_offset = node.day * MINUTES_PER_DAY
            for (s_id, s, e) in node.surgeon_usage:
                iv = model.NewOptionalFixedSizeIntervalVar(day_offset + s, e - s, x, f's_{s_id}')
                surgeon_intervals[s_id].append(iv)

    for v in day_room_cons.values(): model.Add(sum(v) <= 1)
    for v in patient_cons.values(): model.Add(sum(v) <= 1)

    for s_id, intervals in surgeon_intervals.items():
        gap = surgeon_turnover[s_id]
        if gap < 0:
            demands = [1] * len(intervals)
            model.AddCumulative(intervals, demands, 2)
        else:
            model.AddNoOverlap(intervals)

    coeffs = [int(n.weight) for n in nodes]
    model.Maximize(cp_model.LinearExpr.WeightedSum(x_vars, coeffs))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 16

    status = solver.Solve(model)
    selected = []
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print(f"    [Hypergraph] Obj: {solver.ObjectiveValue()}")
        for i, x in enumerate(x_vars):
            if solver.Value(x): selected.append(nodes[i])

    return selected


# ==========================================
# 5. MAIN
# ==========================================
def main():
    print("=== ARGO V23.1: GA + TQDM + Hypergraph ===")

    data_by_dept = load_data()
    full_csv = []
    final_json = {}
    summary = []

    for dept, patients in data_by_dept.items():
        print(f"\n>>> Optimizing: {dept}")

        p_map = {p.id: p for p in patients}
        jan_p = [p for p in patients if p.type == 'JAN']
        feb_p = [p for p in patients if p.type == 'FEB']

        r_gaps, s_gaps = learn_constraints(patients)

        bank = defaultdict(list)
        seen = set()
        best_nodes = []

        total_runs = 0

        for target_runs in SCHEDULE_BATCHES:
            runs_to_do = target_runs - total_runs
            print(f"    [Mining] Running {runs_to_do} new GA schedules (Total Target: {target_runs})...")

            for i in range(runs_to_do):
                sched = run_hartmann_ga(jan_p, feb_p, p_map, r_gaps, s_gaps, total_runs + i + 1)

                # Atomize
                for (d, r), ops in sched.schedule.items():
                    node = PathNode(d, r, ops, p_map)
                    if node.id not in seen:
                        bank[(d, r)].append(node)
                        seen.add(node.id)

            total_runs = target_runs

            print(f"    [Solving] Bank Size: {len(seen)} paths...")
            selected = solve_hypergraph(bank, p_map, s_gaps, time_limit=120)

            s_ids = set()
            for n in selected: s_ids.update(n.pids)
            j_cnt = sum(1 for pid in s_ids if p_map[pid].type == 'JAN')
            f_cnt = sum(1 for pid in s_ids if p_map[pid].type == 'FEB')
            print(f"    -> Result: Jan {j_cnt}/{len(jan_p)} | Feb Moved {f_cnt}")

            if not best_nodes:
                best_nodes = selected
            else:
                curr = j_cnt * WEIGHT_JAN + f_cnt * WEIGHT_FEB
                best = sum(1 for n in best_nodes for pid in n.pids if p_map[pid].type == 'JAN') * WEIGHT_JAN + \
                       sum(1 for n in best_nodes for pid in n.pids if p_map[pid].type == 'FEB') * WEIGHT_FEB
                if curr >= best: best_nodes = selected

        # Export
        summary.append({
            "Department": dept, "Jan_Total": len(jan_p), "Jan_Scheduled": j_cnt, "Feb_Pushed": f_cnt
        })

        dept_list = []
        for n in best_nodes:
            date_obj = datetime.date(YEAR, TARGET_MONTH, 1) + datetime.timedelta(days=n.day)
            date_str = date_obj.strftime("%Y-%m-%d")
            for op in n.ops:
                pat = p_map[op['pid']]
                s_h, s_m = divmod(op['start'], 60)
                e_h, e_m = divmod(op['end'], 60)
                time_range = f"{s_h:02d}:{s_m:02d}-{e_h:02d}:{e_m:02d}"
                row = {
                    "Department": dept, "Room": n.room, "Date": date_str, "Time": time_range,
                    "Start": op['start'], "End": op['end'], "PatientID": pat.id, "Type": pat.type,
                    "Surgeon": pat.surgeon, "Duration": pat.duration
                }
                dept_list.append(row)
                full_csv.append(list(row.values()))
        final_json[dept] = dept_list

    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Department", "Room", "Date", "Time", "Start", "End", "PID", "Type", "Surgeon", "Dur"])
        w.writerows(full_csv)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_json, f, indent=4)

    print("\n=== FINAL SUMMARY ===")
    for s in summary:
        print(f"{s['Department']:<20} | {s['Jan_Scheduled']}/{s['Jan_Total']} | {s['Feb_Pushed']}")


if __name__ == "__main__":
    main()