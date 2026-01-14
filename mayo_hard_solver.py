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

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


    def tqdm(iterable, **kwargs):
        return iterable

# --- CONFIGURATION ---
INPUT_FILE = "informative_data.json"
OUTPUT_CSV = "schedule_final_optimized.csv"
OUTPUT_JSON = "schedule_final_optimized.json"

# Constraints
TARGET_MONTH = 1
POOL_MONTH = 2
YEAR = 2022
DAYS_IN_JAN = 31
MINUTES_PER_DAY = 1440
TURNOVER_TIME = 15

# GA Settings
GA_POP_SIZE = 60
GA_GENERATIONS = 40
ELITISM_RATE = 0.1
MUTATION_RATE = 0.3

# Limits
MAX_SIEGE_ATTEMPTS = 50

# Weights
WEIGHT_JAN = 100_000
WEIGHT_FEB = 100


# ==========================================
# 1. DATA LOADING
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
    if not os.path.exists(INPUT_FILE): return {}
    with open(INPUT_FILE, 'r') as f:
        raw_data = json.load(f)
    print(f">>> Loaded {len(raw_data)} raw records.")

    proc_room_map = defaultdict(set)
    for entry in raw_data:
        if entry.get("Discharge Location") != "RST MCH Saint Marys Campus": continue
        proc_str = entry.get("Scheduled Procedure")
        room = entry.get("Room")
        if proc_str and room:
            code = extract_proc_code(proc_str)
            proc_room_map[code].add(room)

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

        p = Patient(i, duration, room, valid_rooms, surgeon, date_str, p_type, dept, day_idx, in_time, proc_code)
        departments[dept].append(p)
    return departments


def learn_constraints(patients):
    patients.sort(key=lambda x: x.original_start)
    # We define strict room turnover here, though mainly used in decoder
    room_turnover = defaultdict(lambda: TURNOVER_TIME)
    surgeon_turnover = defaultdict(lambda: 0)

    s_history = defaultdict(list)
    for p in patients: s_history[(p.original_day_idx, p.surgeon)].append(p)

    for _, p_list in s_history.items():
        for i in range(len(p_list) - 1):
            gap = p_list[i + 1].original_start - (p_list[i].original_start + p_list[i].duration)
            if gap < 0:
                surgeon_turnover[p_list[i].surgeon] = min(surgeon_turnover[p_list[i].surgeon], gap)
    print(f"    [Learner] Learned overlaps for {len([k for k, v in surgeon_turnover.items() if v < 0])} surgeons.")
    return surgeon_turnover


def sanitize_history(patients):
    patients.sort(key=lambda x: (x.original_day_idx, x.original_start))
    surgeon_timeline = defaultdict(list)
    room_timeline = defaultdict(list)
    shifts = 0
    for p in patients:
        start = p.original_day_idx * MINUTES_PER_DAY + p.original_start
        end = start + p.duration

        # Check conflicts
        s_conflicts = [c for c in surgeon_timeline[p.surgeon] if not (end <= c[0] or start >= c[1])]
        # FIX: Room timeline check must include turnover
        r_conflicts = [c for c in room_timeline[(p.original_day_idx, p.room)] if
                       not (end + TURNOVER_TIME <= c[0] or start >= c[1] + TURNOVER_TIME)]

        if s_conflicts or r_conflicts:
            max_s = max([c[1] for c in s_conflicts]) if s_conflicts else 0
            max_r = max([c[1] + TURNOVER_TIME for c in r_conflicts]) if r_conflicts else 0
            new_start = max(start, max_s, max_r)
            shift = new_start - start
            p.original_start += shift
            start += shift
            end += shift
            shifts += 1

        surgeon_timeline[p.surgeon].append((start, end))
        room_timeline[(p.original_day_idx, p.room)].append((start, end))
    print(f"    [Sanitizer] Resolved {shifts} historical conflicts.")
    return patients


# ==========================================
# 3. GA ENGINE (Siege Mode)
# ==========================================
class Individual:
    def __init__(self, gene, p_map):
        self.gene = gene
        self.p_map = p_map
        self.score = 0
        self.schedule = {}


def decode_sgs_flexible(ind, surgeon_turnover):
    room_avail = defaultdict(int)
    surgeon_avail = defaultdict(int)
    sched = defaultdict(list)
    score = 0

    for pid in ind.gene:
        p = ind.p_map[pid]
        days_to_try = [p.original_day_idx]
        other = list(range(DAYS_IN_JAN))
        if days_to_try[0] in other: other.remove(days_to_try[0])
        days_to_try.extend(random.sample(other, 2))

        placed = False
        s_gap = surgeon_turnover[p.surgeon]

        for d in days_to_try:
            if placed: break
            candidates = list(p.compatible_rooms)
            random.shuffle(candidates)
            best_room = None
            best_start = float('inf')

            for r in candidates:
                # FIX: Strict Turnover Calculation
                last_end = room_avail[(d, r)]
                # If room was used, next start must be at least last_end + turnover
                start_r = last_end + TURNOVER_TIME if last_end > 0 else 0

                s_ready = surgeon_avail[(d, p.surgeon)]
                start_s = s_ready + s_gap

                start = max(start_r, start_s)

                if d == p.original_day_idx: start = max(start, p.original_start)

                if start < best_start:
                    best_start = start
                    best_room = r

            if best_room and best_start + p.duration <= MINUTES_PER_DAY:
                end = best_start + p.duration
                sched[(d, best_room)].append({'pid': pid, 'start': best_start, 'end': end})
                room_avail[(d, best_room)] = end
                surgeon_avail[(d, p.surgeon)] = end
                score += p.weight
                placed = True
    ind.schedule = sched
    ind.score = score


def run_targeted_ga(target_pids, p_map, s_gap, boost_factor=1):
    for pid in target_pids: p_map[pid].weight = 1000 * boost_factor
    pop = []
    for _ in range(GA_POP_SIZE):
        g = list(target_pids)
        if len(pop) > 0: random.shuffle(g)
        ind = Individual(g, p_map)
        decode_sgs_flexible(ind, s_gap)
        pop.append(ind)
    for _ in range(GA_GENERATIONS):
        pop.sort(key=lambda x: x.score, reverse=True)
        next_pop = pop[:int(GA_POP_SIZE * ELITISM_RATE)]
        while len(next_pop) < GA_POP_SIZE:
            p1 = random.choice(pop[:GA_POP_SIZE // 2])
            child_g = list(p1.gene)
            if random.random() < MUTATION_RATE:
                i, j = random.sample(range(len(child_g)), 2)
                child_g[i], child_g[j] = child_g[j], child_g[i]
            child = Individual(child_g, p_map)
            decode_sgs_flexible(child, s_gap)
            next_pop.append(child)
        pop = next_pop
    return pop[0]


# --- BULLDOZER LOGIC ---
def run_bulldozer_ga(missing_ids, covered_ids, p_map, s_gap):
    pop = []
    for _ in range(GA_POP_SIZE):
        priority = list(missing_ids)
        random.shuffle(priority)
        filler = list(covered_ids)
        random.shuffle(filler)
        g = priority + filler
        ind = Individual(g, p_map)
        decode_sgs_flexible(ind, s_gap)
        pop.append(ind)

    for _ in range(GA_GENERATIONS):
        pop.sort(key=lambda x: x.score, reverse=True)
        next_pop = pop[:int(GA_POP_SIZE * ELITISM_RATE)]
        while len(next_pop) < GA_POP_SIZE:
            p1 = random.choice(pop[:GA_POP_SIZE // 2])
            child_g = list(p1.gene)
            if random.random() < MUTATION_RATE:
                i, j = random.sample(range(len(child_g)), 2)
                child_g[i], child_g[j] = child_g[j], child_g[i]
            child = Individual(child_g, p_map)
            decode_sgs_flexible(child, s_gap)
            next_pop.append(child)
        pop = next_pop
    return pop[0]


# ==========================================
# 4. PATH INJECTOR (Phase 2) - STRICT
# ==========================================
def create_augmented_paths(valid_jan_path_node, feb_pool, p_map):
    base_ops = sorted(valid_jan_path_node.ops, key=lambda x: x['start'])

    # 1. Identify Gaps WITH Turnover
    gaps = []
    # If first op starts at 100, gap is 0 to 100. Turnover is applied when *entering* a used room.
    # Case A: Morning start. Room is clean. Gap is 0 to start.
    current_time = 0
    for op in base_ops:
        if op['start'] > current_time:
            # FIX: Only valid gap if duration > 0.
            # Note: We need turnover BEFORE the op starts if there was a previous op.
            # But here 'op' is the *next* pre-scheduled Jan patient.
            # We must finish cleaning the Feb patient BEFORE 'op' starts.
            available_end = op['start'] - TURNOVER_TIME
            if available_end > current_time:
                gaps.append({'start': current_time, 'end': available_end})

        # When does the room become free? After Jan patient ends + Cleaning.
        current_time = op['end'] + TURNOVER_TIME

    # Check end of day
    if current_time < MINUTES_PER_DAY:
        gaps.append({'start': current_time, 'end': MINUTES_PER_DAY})

    if not gaps: return []

    candidates = [p for p in feb_pool if valid_jan_path_node.room in p.compatible_rooms]
    random.shuffle(candidates)
    new_nodes = []

    for _ in range(10):
        new_ops = copy.deepcopy(base_ops)
        added = False
        local_gaps = copy.deepcopy(gaps)

        for cand in candidates:
            for g in local_gaps:
                if (g['end'] - g['start']) >= cand.duration:
                    # FIX: Start at gap start. No need to add turnover before, as 'g['start']' already accounts for prev op turnover
                    new_ops.append({'pid': cand.id, 'start': g['start'], 'end': g['start'] + cand.duration})

                    # Update gap: The new patient finishes, then we need to clean.
                    g['start'] += cand.duration + TURNOVER_TIME
                    added = True
                    break

        if added:
            new_node = PathNode(valid_jan_path_node.day, valid_jan_path_node.room, new_ops, p_map)
            new_nodes.append(new_node)
    return new_nodes


# ==========================================
# 5. HYPERGRAPH SOLVER
# ==========================================
class PathNode:
    __slots__ = ['id', 'day', 'room', 'ops', 'pids', 'weight', 'surgeon_usage', 'is_historical']

    def __init__(self, day, room, ops, patients_map, is_historical=False):
        self.day = day
        self.room = room
        self.ops = ops
        self.ops.sort(key=lambda x: x['start'])
        self.pids = tuple(sorted([op['pid'] for op in self.ops]))
        self.weight = 0
        for pid in self.pids:
            w = WEIGHT_JAN if patients_map[pid].type == 'JAN' else WEIGHT_FEB
            self.weight += w
        self.is_historical = is_historical
        self.surgeon_usage = []
        for op in self.ops:
            p = patients_map[op['pid']]
            self.surgeon_usage.append((p.surgeon, op['start'], op['end']))
        import hashlib
        self.id = hashlib.md5(f"{day}_{room}_{self.pids}".encode()).hexdigest()


def solve_hypergraph(bank, p_map, surgeon_turnover, hints=None, min_jan_count=0, time_limit=60):
    if not HAS_ORTOOLS: return [], [], -1
    model = cp_model.CpModel()
    x_vars = []
    nodes = []

    day_room_cons = defaultdict(list)
    patient_cons = defaultdict(list)
    surgeon_intervals = defaultdict(list)

    hint_map = {n.id: True for n in hints} if hints else {}

    for key, n_list in bank.items():
        for node in n_list:
            x = model.NewBoolVar(f'x_{node.id}')
            if node.id in hint_map: model.AddHint(x, 1)
            x_vars.append(x)
            nodes.append(node)
            day_room_cons[key].append(x)
            for pid in node.pids: patient_cons[pid].append(x)
            offset = node.day * 1440
            for (s_id, s, e) in node.surgeon_usage:
                gap = surgeon_turnover[s_id]
                effective_end = e + gap if gap < 0 else e
                duration = max(1, effective_end - s)
                iv = model.NewOptionalFixedSizeIntervalVar(offset + s, duration, x, f's_{s_id}')
                surgeon_intervals[s_id].append(iv)

    for v in day_room_cons.values(): model.Add(sum(v) <= 1)
    for v in patient_cons.values(): model.Add(sum(v) <= 1)
    for s_id, intervals in surgeon_intervals.items(): model.AddNoOverlap(intervals)

    if min_jan_count > 0:
        total_jan = []
        for pid, vars_list in patient_cons.items():
            if p_map[pid].type == 'JAN':
                total_jan.append(sum(vars_list))
        model.Add(sum(total_jan) >= min_jan_count)

    coeffs = [int(n.weight) for n in nodes]
    model.Maximize(cp_model.LinearExpr.WeightedSum(x_vars, coeffs))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 16
    status = solver.Solve(model)

    selected = []
    covered = set()
    obj_val = 0
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        obj_val = solver.ObjectiveValue()
        for i, x in enumerate(x_vars):
            if solver.Value(x):
                selected.append(nodes[i])
                covered.update(nodes[i].pids)
        return selected, covered, obj_val
    else:
        return None, None, -1


# ==========================================
# 6. MAIN
# ==========================================
def main():
    print("=== ARGO V40: Bulldozer + Strict Turnover Fix ===")

    data_by_dept = load_data()
    final_json = {}
    full_csv = []
    summary = []

    for dept, patients in data_by_dept.items():
        print(f"\n>>> Optimizing: {dept}")

        patients = sanitize_history(patients)
        p_map = {p.id: p for p in patients}
        s_gaps = learn_constraints(patients)
        jan_p = [p for p in patients if p.type == 'JAN']
        feb_p = [p for p in patients if p.type == 'FEB']
        all_jan_ids = [p.id for p in jan_p]

        bank = defaultdict(list)
        seen = set()

        # Seed
        hist_sched = defaultdict(list)
        for p in patients:
            if 0 <= p.original_day_idx < DAYS_IN_JAN:
                hist_sched[(p.original_day_idx, p.room)].append(p)
        for (d, r), plist in hist_sched.items():
            ops = [{'pid': p.id, 'start': p.original_start, 'end': p.original_start + p.duration} for p in plist]
            node = PathNode(d, r, ops, p_map, is_historical=True)
            if node.id not in seen:
                bank[(d, r)].append(node)
                seen.add(node.id)

        best_nodes = []
        best_jan_count = 0

        # --- PHASE 1: BULLDOZER SIEGE ---
        print(f"    [Phase 1] Locking {len(all_jan_ids)} Jan patients...")

        # Initial Mining
        for _ in range(3):
            sched = run_targeted_ga(all_jan_ids, p_map, s_gaps, boost_factor=1)
            for (d, r), ops in sched.schedule.items():
                node = PathNode(d, r, ops, p_map)
                if node.id not in seen:
                    bank[(d, r)].append(node)
                    seen.add(node.id)

        attempts = 0
        while attempts < MAX_SIEGE_ATTEMPTS:
            time_limit = 60 + (attempts * 5)
            selected, covered, score = solve_hypergraph(bank, p_map, s_gaps, hints=best_nodes,
                                                        min_jan_count=best_jan_count, time_limit=time_limit)

            if selected is None:
                print("    [Warning] Infeasible. Relaxing constraint...")
                best_jan_count -= 1
                continue

            current_jan = sum(1 for pid in covered if p_map[pid].type == 'JAN')
            print(f"    [Iter {attempts + 1}] Jan: {current_jan}/{len(all_jan_ids)}")

            if current_jan >= best_jan_count:
                best_nodes = selected
                best_jan_count = current_jan

            if current_jan == len(all_jan_ids):
                print("    -> SUCCESS: 100% January Coverage.")
                break

            covered_pids = set()
            for n in best_nodes: covered_pids.update(n.pids)
            missing = [pid for pid in all_jan_ids if pid not in covered_pids]

            if not missing: break

            print(f"    -> Bulldozing {len(missing)} missing patients...")
            fillers = list(covered_pids)
            sched = run_bulldozer_ga(missing, fillers, p_map, s_gaps)

            added = 0
            for (d, r), ops in sched.schedule.items():
                node = PathNode(d, r, ops, p_map)
                if node.id not in seen:
                    bank[(d, r)].append(node)
                    seen.add(node.id)
                    added += 1
            print(f"    -> Added {added} bulldozer paths.")
            attempts += 1

        # --- PHASE 2: INJECTION ---
        print(f"    [Phase 2] Injecting February into restricted neighborhood...")

        phase2_bank = defaultdict(list)
        phase2_seen = set()

        for n in best_nodes:
            phase2_bank[(n.day, n.room)].append(n)
            phase2_seen.add(n.id)

        injection_count = 0
        for jan_node in best_nodes:
            new_variants = create_augmented_paths(jan_node, feb_p, p_map)
            for v in new_variants:
                if v.id not in phase2_seen:
                    phase2_bank[(v.day, v.room)].append(v)
                    phase2_seen.add(v.id)
                    injection_count += 1

        print(f"    -> Added {injection_count} augmented paths. Solving final...")

        final_nodes, final_covered, f_score = solve_hypergraph(phase2_bank, p_map, s_gaps, hints=best_nodes,
                                                               min_jan_count=len(all_jan_ids), time_limit=180)

        if final_nodes is None:
            print("    [Warning] Could not improve Feb without breaking Jan. Keeping Phase 1 result.")
            final_nodes = best_nodes
            final_covered = set()
            for n in final_nodes: final_covered.update(n.pids)

        j_cnt = sum(1 for pid in final_covered if p_map[pid].type == 'JAN')
        f_cnt = sum(1 for pid in final_covered if p_map[pid].type == 'FEB')

        print(f"    [Final] Jan: {j_cnt} | Feb: {f_cnt}")
        summary.append({'Department': dept, 'Jan_Scheduled': j_cnt, 'Feb_Pushed': f_cnt})

        dept_list = []
        for n in final_nodes:
            date_str = (datetime.date(YEAR, TARGET_MONTH, 1) + datetime.timedelta(days=n.day)).strftime("%Y-%m-%d")
            for op in n.ops:
                pat = p_map[op['pid']]
                s_h, s_m = divmod(op['start'], 60)
                e_h, e_m = divmod(op['end'], 60)
                row = {
                    "Department": dept, "Room": n.room, "Date": date_str,
                    "Time": f"{s_h:02d}:{s_m:02d}-{e_h:02d}:{e_m:02d}",
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

    print("\n=== OPTIMIZATION COMPLETE ===")
    for s in summary: print(s)


if __name__ == "__main__":
    main()