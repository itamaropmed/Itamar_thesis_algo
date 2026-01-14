import json
import csv
import random
import os
import time
import copy
import datetime
import re
from collections import defaultdict

# --- CONFIGURATION ---
INPUT_FILE = "informative_data.json"
OUTPUT_CSV = "schedule_optimized.csv"
OUTPUT_JSON = "schedule_optimized_soft_push.json"
SUMMARY_FILE = "schedule_summary.csv"

# Time Config
TARGET_MONTH = 1
POOL_MONTH = 2
YEAR = 2022
DAYS_IN_JAN = 31
MINUTES_PER_DAY = 1440
TURNOVER_TIME = 15

# GA Tuning (Hartmann-esque)
POPULATION_SIZE = 100
GENERATIONS = 100
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
        self.compatible_rooms = compatible_rooms  # List of all valid rooms
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

    # Pass 1: Map Procedures to Rooms
    proc_room_map = defaultdict(set)
    for entry in raw_data:
        if entry.get("Discharge Location") != "RST MCH Saint Marys Campus": continue
        proc_str = entry.get("Scheduled Procedure")
        room = entry.get("Room")
        if proc_str and room:
            code = extract_proc_code(proc_str)
            proc_room_map[code].add(room)

    print(f"    [Mapping] Mapped {len(proc_room_map)} procedures to compatible rooms.")

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
        # Use all rooms that perform this procedure as candidates, defaulting to historical room
        valid_rooms = list(proc_room_map.get(proc_code, [room]))

        p = Patient(i, duration, valid_rooms, surgeon, date_str, p_type, dept, day_idx, in_time, proc_code)
        departments[dept].append(p)
    return departments


# ==========================================
# 2. CONSTRAINT LEARNER
# ==========================================
def learn_constraints(patients):
    """
    Learns the 'Effective Turnover' for surgeons from history.
    Room turnover is standard (15) but can be learned if needed.
    """
    jan_p = [p for p in patients if p.type == 'JAN' and 0 <= p.original_day_idx < DAYS_IN_JAN]
    jan_p.sort(key=lambda x: x.original_start)

    # Defaults
    room_turnover = defaultdict(lambda: TURNOVER_TIME)
    surgeon_turnover = defaultdict(lambda: 0)

    # 1. Analyze Surgeon Gaps
    s_history = defaultdict(list)
    for p in jan_p: s_history[(p.original_day_idx, p.surgeon)].append(p)

    for _, p_list in s_history.items():
        for i in range(len(p_list) - 1):
            gap = p_list[i + 1].original_start - (p_list[i].original_start + p_list[i].duration)
            if gap < 0:
                # Valid overlap found in history
                surgeon_turnover[p_list[i].surgeon] = min(surgeon_turnover[p_list[i].surgeon], gap)

    print(
        f"    [Learner] Found {len([k for k, v in surgeon_turnover.items() if v < 0])} surgeons with valid historical overlaps.")
    return room_turnover, surgeon_turnover


# ==========================================
# 3. HARTMANN GA ENGINE (FLEXIBLE)
# ==========================================
class Individual:
    def __init__(self, gene, p_map):
        self.gene = gene  # List of patient IDs (Permutation)
        self.p_map = p_map
        self.score = 0
        self.jan_count = 0
        self.feb_count = 0
        self.schedule = {}  # Result of decoding


def decode_sgs_flexible(ind, room_turnover, surgeon_turnover):
    """
    Serial Schedule Generation Scheme with Flexible Room Assignment.
    Places patients one by one into the FIRST AVAILABLE valid slot in ANY COMPATIBLE ROOM.
    """
    # State
    # Room Avail: (Day, Room) -> Available Time
    room_avail = defaultdict(int)
    # Surgeon Avail: (Day, Surgeon) -> Available Time
    surgeon_avail = defaultdict(int)

    schedule_out = defaultdict(list)
    score = 0
    jan = 0
    feb = 0

    for pid in ind.gene:
        p = ind.p_map[pid]

        placed = False

        # Determine Day order to try
        # If Jan, try Original Day first
        days_to_try = []
        if p.type == 'JAN' and 0 <= p.original_day_idx < DAYS_IN_JAN:
            days_to_try.append(p.original_day_idx)

        # Then try other days (randomized for diversity in search)
        other_days = list(range(DAYS_IN_JAN))
        if days_to_try: other_days.remove(days_to_try[0])
        # random.shuffle(other_days) # Optimization: Don't shuffle to keep it deterministic SGS? No, shuffle helps.
        # Actually Hartmann uses determinism. Let's sort by load? Simple: 0..30
        days_to_try.extend(other_days)

        s_gap = surgeon_turnover[p.surgeon]

        for d in days_to_try:
            # Check all compatible rooms
            best_room = None
            best_start = float('inf')

            # Heuristic: Try to find the earliest valid start time across all compatible rooms
            # Shuffle compatible rooms to avoid bias if times are equal
            candidates = list(p.compatible_rooms)
            random.shuffle(candidates)

            for r in candidates:
                r_gap = room_turnover[r]

                # Earliest start based on Resource Availability
                r_ready = room_avail[(d, r)]
                s_ready = surgeon_avail[(d, p.surgeon)]

                # Logic Correction: If Room_Ready == 0 (First op), gap is 0.
                start_r = r_ready + r_gap if r_ready > 0 else 0
                start_s = s_ready + s_gap if s_ready > 0 else 0

                start = max(start_r, start_s)

                if start < best_start:
                    best_start = start
                    best_room = r

            # Can we fit in the best room?
            if best_room and best_start + p.duration <= MINUTES_PER_DAY:
                end = best_start + p.duration

                # Book it
                room_avail[(d, best_room)] = end
                surgeon_avail[(d, p.surgeon)] = end

                schedule_out[(d, best_room)].append({'pid': pid, 'start': best_start, 'end': end})

                score += p.weight
                if p.type == 'JAN':
                    jan += 1
                else:
                    feb += 1
                placed = True
                break

    ind.score = score
    ind.jan_count = jan
    ind.feb_count = feb
    ind.schedule = schedule_out


def run_hartmann_ga(dept, jan_p, feb_p, p_map, room_gap, s_gap):
    # 1. Population Init
    # Seed 1: Historical Order (Jan only)
    jan_p.sort(key=lambda x: (x.original_day_idx, x.original_start))
    seed_gene = [p.id for p in jan_p]

    # Population
    population = []

    # Inject Historical Gene + Random Febs
    for _ in range(POPULATION_SIZE):
        gene = list(seed_gene)
        # Add random Feb candidates
        feb_sample = random.sample(feb_p, min(len(feb_p), 50))  # Try to fit 50 Febs
        gene.extend([p.id for p in feb_sample])

        # Mutation: Slight shuffle
        if len(population) > 0:  # Keep first pure history
            # Shuffle slightly to allow packing optimization
            # But keep Jan relatively sorted by date to help SGS
            pass

        ind = Individual(gene, p_map)
        decode_sgs_flexible(ind, room_gap, s_gap)
        population.append(ind)

    print(f"    [Init] Best Historical Score: {max(i.score for i in population)}")

    # 2. Evolution
    for gen in range(GENERATIONS):
        # Sort by Fitness
        population.sort(key=lambda x: x.score, reverse=True)

        # Elitism
        cutoff = int(POPULATION_SIZE * ELITISM_RATE)
        next_pop = population[:cutoff]

        while len(next_pop) < POPULATION_SIZE:
            # Tournament
            p1 = random.choice(population[:POPULATION_SIZE // 2])
            p2 = random.choice(population[:POPULATION_SIZE // 2])

            # Crossover (OX1 - Order Crossover)
            # Create child gene from p1, p2
            # Here simplified: Single point crossover on lists
            cut = random.randint(0, len(p1.gene))
            child_gene = p1.gene[:cut] + [x for x in p2.gene if x not in p1.gene[:cut]]

            # Mutation (Swap)
            if random.random() < MUTATION_RATE:
                if len(child_gene) > 2:
                    i, j = random.sample(range(len(child_gene)), 2)
                    child_gene[i], child_gene[j] = child_gene[j], child_gene[i]

            # Evaluate
            child = Individual(child_gene, p_map)
            decode_sgs_flexible(child, room_gap, s_gap)
            next_pop.append(child)

        population = next_pop

        best = population[0]
        if gen % 10 == 0:
            print(f"    [Gen {gen}] Best: {best.jan_count} Jan | {best.feb_count} Feb")

    return population[0]


# ==========================================
# 4. MAIN
# ==========================================
def main():
    print("=== ARGO V13: Hartmann GA + Learned Constraints (Flexible Rooms) ===")

    data_by_dept = load_data()
    final_json = {}
    full_csv_rows = []
    summary = []

    for dept, patients in data_by_dept.items():
        print(f"\n>>> Optimizing: {dept}")

        p_map = {p.id: p for p in patients}
        jan_p = [p for p in patients if p.type == 'JAN']
        feb_p = [p for p in patients if p.type == 'FEB']

        # 1. Learn Constraints from History
        r_gaps, s_gaps = learn_constraints(patients)

        # 2. Run Hartmann GA
        best_sched = run_hartmann_ga(dept, jan_p, feb_p, p_map, r_gaps, s_gaps)

        print(f"    -> DONE. Jan: {best_sched.jan_count}/{len(jan_p)} | Feb Moved: {best_sched.feb_count}")

        summary.append({
            "Department": dept,
            "Jan_Total": len(jan_p),
            "Jan_Scheduled": best_sched.jan_count,
            "Feb_Pushed": best_sched.feb_count
        })

        dept_list = []
        for (day, room), ops in best_sched.schedule.items():
            date_obj = datetime.date(YEAR, TARGET_MONTH, 1) + datetime.timedelta(days=day)
            date_str = date_obj.strftime("%Y-%m-%d")

            for op in ops:
                pid = op['pid']
                pat = p_map[pid]
                start, end = op['start'], op['end']

                s_h, s_m = divmod(start, 60)
                e_h, e_m = divmod(end, 60)
                time_range = f"{s_h:02d}:{s_m:02d}-{e_h:02d}:{e_m:02d}"

                row = {
                    "Department": dept, "Room": room, "Date": date_str, "Time": time_range,
                    "Start": start, "End": end, "PatientID": pid, "Type": pat.type,
                    "Surgeon": pat.surgeon, "Duration": pat.duration
                }
                dept_list.append(row)
                full_csv_rows.append(list(row.values()))

        final_json[dept] = dept_list

    # Save
    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Department", "Room", "Date", "Time", "Start", "End", "PID", "Type", "Surgeon", "Dur"])
        w.writerows(full_csv_rows)

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_json, f, indent=4)

    print("\n=== FINAL SUMMARY ===")
    for s in summary:
        print(f"{s['Department']:<20} | {s['Jan_Scheduled']}/{s['Jan_Total']} | {s['Feb_Pushed']}")


if __name__ == "__main__":
    main()
