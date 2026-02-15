import json
import csv
import random
import os
import time
import copy
import datetime
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

try:
    from ortools.sat.python import cp_model

    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False
    print(">>> CRITICAL WARNING: 'ortools' not found.")

INPUT_FILE = "informative_data.json"
OUTPUT_CSV = "schedule_2024_cascade.csv"
OUTPUT_JSON = "schedule_2024_cascade.json"

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

LAYER_1_ROUNDS = 8
LAYER_2_ROUNDS = 5
SOLVER_TIME_LIMIT = 60


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


class Patient:
    __slots__ = ['id', 'duration', 'room', 'surgeon', 'original_date',
                 'year', 'type', 'weight', 'department', 'original_day_idx', 'priority_boost', 'month',
                 'is_fixed', 'eligible_rooms']

    def __init__(self, pid, duration, room, surgeon, date_str, year, p_type, dept, day_idx, month, fixed_rooms,
                 all_rooms):
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

        import hashlib
        content = f"{day}_{room}_{sorted(self.pids)}_{[o['start'] for o in ops]}"
        self.id = hashlib.md5(content.encode()).hexdigest()


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

        p = Patient(i, duration, room, surgeon, date_str, dt.year, p_type, dept, day_idx, month,
                    FIXED_ROOMS[dept], departments[dept]["rooms"])
        departments[dept]["patients"].append(p)

    result = {}
    for dept in DEPARTMENTS:
        if departments[dept]["patients"]:
            result[dept] = departments[dept]["patients"]

    return result


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


def solve_hypergraph(all_edges, required_ids, locked_ids, p_map, strict_required=False):
    print(
        f"    [Solver] Proc {len(all_edges)} edges. Req: {len(required_ids)}. Locked: {len(locked_ids)}. Strict: {strict_required}")
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


def stage_solve_january(month_ids, all_patient_ids, p_map):
    """Solve January to 100% completion"""
    print(f"\n>>> January Anchor: {len(month_ids)} patients")

    day_room_map = defaultdict(list)
    for pid in month_ids:
        p = p_map[pid]
        day_room_map[(p.original_day_idx, p.room)].append(pid)

    all_valid_days = sorted(list(set(k[0] for k in day_room_map.keys())))
    all_rooms = sorted(list(set(k[1] for k in day_room_map.keys())))

    print(f"    Days: {len(all_valid_days)}, Rooms: {len(all_rooms)}")

    edge_bank = {}

    print(f"\n    [PHASE 1] Seeding Foundation")
    tasks = [(day, room, [p_map[pid] for pid in day_room_map.get((day, room), [])], p_map)
             for day in all_valid_days for room in all_rooms]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i in range(LAYER_1_ROUNDS):
            results = executor.map(swarm_worker, tasks)
            count = 0
            for edge_list in results:
                for e in edge_list:
                    if e.id not in edge_bank:
                        edge_bank[e.id] = e
                        count += 1
            print(f"        L1.{i + 1}: Added {count} edges")

    print(f"\n    [PHASE 2] Rescue & Lock")

    for r in range(LAYER_2_ROUNDS):
        is_strict = (r >= 2)
        current_solution = solve_hypergraph(
            list(edge_bank.values()),
            month_ids,
            set(),
            p_map,
            strict_required=is_strict
        )

        if not current_solution:
            current_solution = solve_hypergraph(
                list(edge_bank.values()),
                month_ids,
                set(),
                p_map,
                strict_required=False
            )

        covered = set()
        for e in current_solution:
            covered.update(e.pids)
        missing = [pid for pid in month_ids if pid not in covered]

        print(f"        R{r + 1}: Coverage {len(month_ids) - len(missing)}/{len(month_ids)}")

        if not missing:
            print(f"        ✓ SUCCESS: All {len(month_ids)} January locked!")
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
                    if e.id not in edge_bank:
                        edge_bank[e.id] = e
                        new_count += 1
            print(f"        Rescue: Added {new_count} edges")

    return current_solution


def stage_solve_february_with_locked(feb_ids, jan_locked_pids, all_patient_ids, p_map, jan_solution):
    """Solve February with January locked and repositioned"""
    print(f"\n>>> February Fill: {len(feb_ids)} patients (Jan locked: {len(jan_locked_pids)})")

    day_room_map = defaultdict(list)
    for pid in feb_ids:
        p = p_map[pid]
        day_room_map[(p.original_day_idx, p.room)].append(pid)

    all_valid_days = sorted(list(set(k[0] for k in day_room_map.keys())))
    all_rooms = sorted(list(set(k[1] for k in day_room_map.keys())))

    print(f"    Days: {len(all_valid_days)}, Rooms: {len(all_rooms)}")

    print(f"\n    [REPOSITION JANUARY] Tightening schedule")
    optimized_jan_edges = optimize_locked_edges(jan_solution, p_map)
    edge_bank = {e.id: e for e in optimized_jan_edges}

    print(f"\n    [PHASE 1] Seeding February")
    tasks = [(day, room, [p_map[pid] for pid in day_room_map.get((day, room), [])], p_map)
             for day in all_valid_days for room in all_rooms]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i in range(LAYER_1_ROUNDS):
            results = executor.map(swarm_worker, tasks)
            count = 0
            for edge_list in results:
                for e in edge_list:
                    if e.id not in edge_bank:
                        edge_bank[e.id] = e
                        count += 1
            print(f"        L1.{i + 1}: Added {count} edges")

    print(f"\n    [PHASE 2] Rescue & Lock February")

    for r in range(LAYER_2_ROUNDS):
        is_strict = (r >= 2)
        current_solution = solve_hypergraph(
            list(edge_bank.values()),
            feb_ids,
            jan_locked_pids,
            p_map,
            strict_required=is_strict
        )

        if not current_solution:
            current_solution = solve_hypergraph(
                list(edge_bank.values()),
                feb_ids,
                jan_locked_pids,
                p_map,
                strict_required=False
            )

        covered = set()
        for e in current_solution:
            covered.update(e.pids)
        missing = [pid for pid in feb_ids if pid not in covered]
        jan_check = len([pid for pid in jan_locked_pids if pid in covered])

        print(
            f"        R{r + 1}: Feb {len(feb_ids) - len(missing)}/{len(feb_ids)} | Jan {jan_check}/{len(jan_locked_pids)}")

        if not missing and jan_check == len(jan_locked_pids):
            print(f"        ✓ SUCCESS: All {len(feb_ids)} Feb + {len(jan_locked_pids)} Jan locked!")
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
                    if e.id not in edge_bank:
                        edge_bank[e.id] = e
                        new_count += 1
            print(f"        Rescue: Added {new_count} edges")

    return current_solution


def stage_solve_march(mar_ids, jan_locked_pids, feb_locked_pids, all_patient_ids, p_map, feb_solution):
    """Solve March best effort with Jan+Feb locked"""
    print(
        f"\n>>> March Extension: {len(mar_ids)} patients (Jan+Feb locked: {len(jan_locked_pids) + len(feb_locked_pids)})")

    day_room_map = defaultdict(list)
    for pid in mar_ids:
        p = p_map[pid]
        day_room_map[(p.original_day_idx, p.room)].append(pid)

    all_valid_days = sorted(list(set(k[0] for k in day_room_map.keys())))
    all_rooms = sorted(list(set(k[1] for k in day_room_map.keys())))

    print(f"    Days: {len(all_valid_days)}, Rooms: {len(all_rooms)}")

    print(f"\n    [REPOSITION JAN+FEB] Tightening schedule")
    optimized_janfeb = optimize_locked_edges(feb_solution, p_map)
    edge_bank = {e.id: e for e in optimized_janfeb}

    print(f"\n    [PHASE 1] Seeding March")
    tasks = [(day, room, [p_map[pid] for pid in day_room_map.get((day, room), [])], p_map)
             for day in all_valid_days for room in all_rooms]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i in range(LAYER_1_ROUNDS):
            results = executor.map(swarm_worker, tasks)
            count = 0
            for edge_list in results:
                for e in edge_list:
                    if e.id not in edge_bank:
                        edge_bank[e.id] = e
                        count += 1
            print(f"        L1.{i + 1}: Added {count} edges")

    print(f"\n    [PHASE 2] Best-Effort March")

    locked_pids = jan_locked_pids | feb_locked_pids
    current_solution = solve_hypergraph(
        list(edge_bank.values()),
        mar_ids,
        locked_pids,
        p_map,
        strict_required=False
    )

    covered = set()
    for e in current_solution:
        covered.update(e.pids)
    mar_covered = [pid for pid in mar_ids if pid in covered]
    janfeb_covered = [pid for pid in locked_pids if pid in covered]

    print(f"        Coverage: Mar {len(mar_covered)}/{len(mar_ids)} | Jan+Feb {len(janfeb_covered)}/{len(locked_pids)}")

    return current_solution


def main():
    print("=== ARGO CASCADE V3.1: Multi-Department Independent Cascade ===\n")
    data_by_dept = load_data()
    final_json = {}
    full_csv = []

    for dept in DEPARTMENTS:
        if dept not in data_by_dept:
            print(f"\n{'=' * 70}")
            print(f">>> {dept}: No data")
            print(f"{'=' * 70}")
            continue

        patients = data_by_dept[dept]
        print(f"\n{'=' * 70}")
        print(f">>> {dept}: {len(patients)} patients")
        print(f"{'=' * 70}")

        p_map = {p.id: p for p in patients}
        all_pids = list(p_map.keys())

        jan_ids = [pid for pid in all_pids if p_map[pid].month == 1]
        feb_ids = [pid for pid in all_pids if p_map[pid].month == 2]
        mar_ids = [pid for pid in all_pids if p_map[pid].month == 3]

        print(f"    Jan: {len(jan_ids)}, Feb: {len(feb_ids)}, Mar: {len(mar_ids)}")

        print(f"\n{'─' * 70}\nSTAGE 1: JANUARY\n{'─' * 70}")
        jan_solution = stage_solve_january(jan_ids, all_pids, p_map)

        jan_locked_pids = set()
        for e in jan_solution:
            jan_locked_pids.update(e.pids)

        if len(jan_locked_pids) < len(jan_ids):
            print(f"\n✗ January incomplete: {len(jan_locked_pids)}/{len(jan_ids)}")
            final_solution = jan_solution
        else:
            print(f"✓ January 100% locked!")

            print(f"\n{'─' * 70}\nSTAGE 2: FEBRUARY\n{'─' * 70}")
            feb_solution = stage_solve_february_with_locked(feb_ids, jan_locked_pids, all_pids, p_map, jan_solution)

            feb_scheduled = len([pid for pid in feb_ids if any(pid in e.pids for e in feb_solution)])

            if feb_scheduled < len(feb_ids):
                print(f"\n✗ February incomplete: {feb_scheduled}/{len(feb_ids)}")
                final_solution = feb_solution
            else:
                print(f"✓ February 100% locked!")

                print(f"\n{'─' * 70}\nSTAGE 3: MARCH\n{'─' * 70}")
                mar_solution = stage_solve_march(mar_ids, jan_locked_pids, set(feb_ids), all_pids, p_map, feb_solution)
                final_solution = mar_solution

        print(f"\n{'─' * 70}\nEXPORT\n{'─' * 70}")

        dept_list = []
        for edge in final_solution:
            date_str = (ANCHOR_DATE + datetime.timedelta(days=edge.day)).strftime("%Y-%m-%d")
            for op in edge.ops:
                pat = p_map[op['pid']]
                row = {
                    "Department": dept, "Room": edge.room, "Date": date_str,
                    "Start": op['start'], "End": op['end'], "PatientID": pat.id,
                    "Type": pat.type, "Surgeon": pat.surgeon,
                    "Time": f"{minutes_to_hhmm(op['start'])}-{minutes_to_hhmm(op['end'])}",
                    "Year": pat.year
                }
                dept_list.append(row)
                full_csv.append(list(row.values()))
        final_json[dept] = dept_list

        print(f"Exported: {len(dept_list)} rows")

    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Department", "Room", "Date", "Start", "End", "PID", "Type", "Surgeon", "Time", "Year"])
        w.writerows(full_csv)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_json, f, indent=4)

    print(f"\n{'=' * 70}")
    print(f"✓ Saved: {OUTPUT_CSV} and {OUTPUT_JSON}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()