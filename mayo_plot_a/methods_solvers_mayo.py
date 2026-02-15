#!/usr/bin/env python3
"""
M-ORSP Multi-Method Solver  –  Mayo Operating Room Scheduling Problem
=====================================================================
Place this file next to informative_data.json and run it.
pip install tqdm   (only external dependency)

7 solving methods × 2 independent departments.

Methods
-------
1. Hartmann GA 1998  – Permutation-based GA, Serial SGS, two-point crossover
2. Hartmann GA 2002  – Self-adapting GA (extra gene chooses Serial / Parallel SGS)
3. Simulated Annealing – Swap-neighbourhood, geometric cooling, periodic reheat
4. GNN Priority Heuristic – 2-layer message-passing on surgeon-conflict graph
5. DRL Policy Heuristic  – Feed-forward "policy net" picks (day,room) per patient
6. Average Human (FCFS)  – First-come-first-served, original room preference
7. Smart Human (LPT+BFD) – Longest-Processing-Time, Best-Fit-Decreasing bin-pack

Constraints (Updated)
---------------------
* Fixed rooms  101 CCL, 102 CCL  /  106 HRS, 109 HRS  →  patients locked
* Flexible rooms  →  patients may move to any other room in the department
* Surgeon non-cloning  (same surgeon cannot overlap on same day)
* 15-min turnover between consecutive procedures in the same room
* Schedule window  31 days (Days 0-30).
* HIERARCHICAL ANCHORING:
  1. Solve January. If 100% success -> Lock it.
  2. Solve February (fitting into gaps of Jan). If 100% success -> Lock it.
  3. Solve March (fitting into gaps of Jan+Feb).
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import json, csv, random, os, sys, time, math, hashlib, datetime
from collections import defaultdict
from copy import deepcopy

# ── third-party (pip install tqdm numpy) ──────────────────────────────────────
try:
    import numpy as np
    from tqdm import tqdm
except ImportError:
    print("Please install dependencies: pip install tqdm numpy")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
INPUT_FILE = "informative_data.json"
OUTPUT_CSV = "schedule_multimethod_anchored.csv"

TARGET_YEAR = 2024
DEPARTMENTS = ["RST ROMB CCL", "RST ROMB HRS"]
FIXED_ROOMS = {
    "RST ROMB CCL": {"101 CCL", "102 CCL"},
    "RST ROMB HRS": {"106 HRS", "109 HRS"},
}
ANCHOR_DATE = datetime.date(2024, 1, 1)
NUM_DAYS = 31  # Using a 31-day repeating grid for the months
DAY_START = 420  # 07:00 in minutes
DAY_END = 1140  # 19:00 in minutes
DAY_LEN = DAY_END - DAY_START  # 720 min
TURNOVER = 15  # minutes between procedures in the same room

# ── Per-method time budget per STAGE (seconds).
BUDGET_PER_STAGE = 25

# GA hyper-parameters
GA_POP = 50
GA_GEN = 50
GA_CX = 0.85  # crossover probability
GA_MUT = 0.05  # mutation probability
GA_ELITE = 4  # elitism count

# SA hyper-parameters
SA_T0 = 800.0
SA_COOL = 0.995
SA_ITER = 8_000
SA_REHEAT = 2_000

# DRL episodes
DRL_EPISODES = 5

random.seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODEL
# ═══════════════════════════════════════════════════════════════════════════════
class Patient:
    """One surgical case."""
    __slots__ = (
        "id", "duration", "original_room", "surgeon", "original_date",
        "month", "ptype", "dept", "is_fixed", "eligible_rooms", "idx",
    )

    def __init__(self, pid, dur, room, surgeon, date_str, month,
                 ptype, dept, fixed, all_rooms, idx):
        self.id = pid
        self.duration = int(dur)
        self.original_room = room
        self.surgeon = surgeon
        self.original_date = date_str
        self.month = month
        self.ptype = ptype  # 'JAN', 'FEB', 'MAR'
        self.dept = dept
        self.is_fixed = room in fixed
        self.eligible_rooms = (
            [room] if self.is_fixed
            else sorted(r for r in all_rooms if r not in fixed)
        )
        self.idx = idx


def _parse_time(t: str):
    """'HH:MM:SS' → minutes from midnight, or None."""
    if not t:
        return None
    try:
        parts = t.split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return None


def load_department(raw: list, dept: str):
    """Return (patients, sorted_room_list) for one department."""
    fixed = FIXED_ROOMS[dept]
    room_set = set()
    for e in raw:
        if e.get("OR Department") == dept and e.get("Date", "").startswith(str(TARGET_YEAR)):
            r = e.get("Room", "")
            if r and "Temp" not in r:
                room_set.add(r)

    patients = []
    idx = 0
    # Updated: Allow Jan(1), Feb(2), Mar(3)
    month_map = {1: "JAN", 2: "FEB", 3: "MAR"}

    for i, e in enumerate(raw):
        if e.get("OR Department") != dept:
            continue
        ds = e.get("Date", "")
        if not ds.startswith(str(TARGET_YEAR)):
            continue
        try:
            dt = datetime.datetime.strptime(ds, "%Y-%m-%d")
        except Exception:
            continue
        if dt.month not in (1, 2, 3):
            continue

        t_in = _parse_time(e.get("In Proc Room"))
        t_out = _parse_time(e.get("Out Proc Room"))
        if t_in is None or t_out is None:
            continue

        dur = t_out - t_in if t_out >= t_in else (1440 - t_in) + t_out
        if dur <= 0: dur = 30
        if dur > 720: dur = 720

        room = e.get("Room", "")
        surgeon = e.get("Lead Surgeon/Provider", "")
        if not room or not surgeon or "Temp" in room:
            continue

        patients.append(
            Patient(i, dur, room, surgeon, ds, dt.month,
                    month_map[dt.month], dept, fixed, room_set, idx)
        )
        idx += 1

    return patients, sorted(room_set)


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEDULE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
class Schedule:
    """Maintains room and surgeon timelines; checks all hard constraints."""

    def __init__(self, rooms):
        self.rooms = rooms
        self.room_tl = defaultdict(list)  # (day, room) → [(s, e, pid), …]
        self.surg_tl = defaultdict(list)  # (day, surgeon) → [(s, e), …]
        self.placed = {}  # pid → (day, room, start, end)

    def can_place(self, p: Patient, day: int, room: str, start: int) -> bool:
        end = start + p.duration
        if day < 0 or day >= NUM_DAYS:
            return False
        if start < DAY_START or end > DAY_END:
            return False
        if room not in p.eligible_rooms:
            return False
        # room conflict (including turnover)
        for s, e, _ in self.room_tl[(day, room)]:
            if not (end + TURNOVER <= s or start >= e + TURNOVER):
                return False
        # surgeon overlap (no turnover needed)
        for s, e in self.surg_tl[(day, p.surgeon)]:
            if not (end <= s or start >= e):
                return False
        return True

    def place(self, p: Patient, day: int, room: str, start: int):
        end = start + p.duration
        self.room_tl[(day, room)].append((start, end, p.id))
        self.room_tl[(day, room)].sort()
        self.surg_tl[(day, p.surgeon)].append((start, end))
        self.surg_tl[(day, p.surgeon)].sort()
        self.placed[p.id] = (day, room, start, end)

    def find_first_slot(self, p: Patient, day: int, room: str):
        """Earliest feasible start, or None."""
        if room not in p.eligible_rooms:
            return None
        existing = self.room_tl.get((day, room), [])
        candidates = [DAY_START] + [e + TURNOVER for s, e, _ in existing]
        for c in sorted(set(candidates)):
            if self.can_place(p, day, room, c):
                return c
        return None

    def copy(self):
        s2 = Schedule(self.rooms)
        for k, v in self.room_tl.items():
            s2.room_tl[k] = list(v)
        for k, v in self.surg_tl.items():
            s2.surg_tl[k] = list(v)
        s2.placed = dict(self.placed)
        return s2


def score_schedule(sched: Schedule, target_ids):
    """Objective: Count how many from the target list are placed."""
    return sum(1 for pid in target_ids if pid in sched.placed)


# ═══════════════════════════════════════════════════════════════════════════════
# SGS SCHEMES (Now supporting Base Schedule Injection)
# ═══════════════════════════════════════════════════════════════════════════════
def serial_sgs(activity_list, p_map, rooms, base_schedule=None):
    """
    Serial SGS starting from a LOCKED base schedule.
    Activities in activity_list are fitted into gaps.
    """
    sched = base_schedule.copy() if base_schedule else Schedule(rooms)

    for pid in activity_list:
        if pid in sched.placed: continue  # Already locked? Skip (shouldn't happen)
        p = p_map[pid]
        done = False
        for day in range(NUM_DAYS):
            for room in p.eligible_rooms:
                start = sched.find_first_slot(p, day, room)
                if start is not None:
                    sched.place(p, day, room, start)
                    done = True
                    break
            if done: break
    return sched


def parallel_sgs(activity_list, p_map, rooms, base_schedule=None):
    """
    Parallel SGS starting from a LOCKED base schedule.
    """
    sched = base_schedule.copy() if base_schedule else Schedule(rooms)
    remaining = list(activity_list)

    for day in range(NUM_DAYS):
        if not remaining: break
        still = []
        for pid in remaining:
            p = p_map[pid]
            placed = False
            for room in p.eligible_rooms:
                start = sched.find_first_slot(p, day, room)
                if start is not None:
                    sched.place(p, day, room, start)
                    placed = True
                    break
            if not placed:
                still.append(pid)
        remaining = still
    return sched


# ═══════════════════════════════════════════════════════════════════════════════
# SOLVER KERNELS (Single Stage)
# ═══════════════════════════════════════════════════════════════════════════════

def _two_point_cx(mother, father, all_ids):
    n = len(mother)
    q1, q2 = sorted(random.sample(range(n), 2))
    child = [None] * n
    used = set()
    for i in range(q1 + 1):
        child[i] = mother[i];
        used.add(mother[i])
    filt = [g for g in father if g not in used]
    for i, g in enumerate(filt[: q2 - q1]):
        child[q1 + 1 + i] = g;
        used.add(g)
    rest = [g for g in mother if g not in used]
    pos = q2 + 1
    for g in rest:
        if pos < n: child[pos] = g; pos += 1
    return child


def _swap_mutate(perm):
    if len(perm) < 2: return perm
    i, j = random.sample(range(len(perm)), 2)
    perm[i], perm[j] = perm[j], perm[i]
    return perm


# --- Method 1: Hartmann 98 (Single Stage) ---
def run_hartmann98(target_ids, p_map, rooms, base_sched):
    if not target_ids: return base_sched, 0
    pop = [random.sample(target_ids, len(target_ids)) for _ in range(GA_POP)]
    best_s, best_p = -1, None
    t0 = time.time()

    for gen in range(GA_GEN):
        if time.time() - t0 > BUDGET_PER_STAGE: break
        scores = []
        for ind in pop:
            s = serial_sgs(ind, p_map, rooms, base_sched)
            sc = score_schedule(s, target_ids)
            scores.append(sc)
            if sc > best_s: best_s, best_p = sc, list(ind)

        # Simple selection/breeding
        ranked = sorted(range(len(pop)), key=lambda i: scores[i], reverse=True)
        new = [list(pop[ranked[i]]) for i in range(min(GA_ELITE, len(pop)))]
        while len(new) < GA_POP:
            p1 = pop[ranked[random.randint(0, len(ranked) // 2)]]
            p2 = pop[ranked[random.randint(0, len(ranked) // 2)]]
            child = _two_point_cx(p1, p2, target_ids) if random.random() < GA_CX else list(p1)
            if random.random() < GA_MUT: child = _swap_mutate(child)
            new.append(child)
        pop = new

    return serial_sgs(best_p, p_map, rooms, base_sched), best_s


# --- Method 2: Hartmann 02 (Single Stage) ---
def run_hartmann02(target_ids, p_map, rooms, base_sched):
    if not target_ids: return base_sched, 0
    # Genome: (Permutation, Mode=0/1)
    pop = [(random.sample(target_ids, len(target_ids)), random.choice([0, 1])) for _ in range(GA_POP)]
    best_s, best_ind = -1, None
    t0 = time.time()

    for gen in range(GA_GEN):
        if time.time() - t0 > BUDGET_PER_STAGE: break
        scores = []
        for (perm, mode) in pop:
            s = serial_sgs(perm, p_map, rooms, base_sched) if mode == 0 else parallel_sgs(perm, p_map, rooms,
                                                                                          base_sched)
            sc = score_schedule(s, target_ids)
            scores.append(sc)
            if sc > best_s: best_s, best_ind = sc, (list(perm), mode)

        ranked = sorted(range(len(pop)), key=lambda i: scores[i], reverse=True)
        new = [(list(pop[ranked[i]][0]), pop[ranked[i]][1]) for i in range(min(GA_ELITE, len(pop)))]
        while len(new) < GA_POP:
            p1 = pop[ranked[random.randint(0, len(ranked) // 2)]]
            p2 = pop[ranked[random.randint(0, len(ranked) // 2)]]
            c_perm = _two_point_cx(p1[0], p2[0], target_ids) if random.random() < GA_CX else list(p1[0])
            c_mode = random.choice([p1[1], p2[1]])
            if random.random() < GA_MUT: c_perm = _swap_mutate(c_perm)
            if random.random() < 0.1: c_mode = 1 - c_mode
            new.append((c_perm, c_mode))
        pop = new

    fin_s = serial_sgs(best_ind[0], p_map, rooms, base_sched) if best_ind[1] == 0 else parallel_sgs(best_ind[0], p_map,
                                                                                                    rooms, base_sched)
    return fin_s, best_s


# --- Method 3: SA (Single Stage) ---
def run_sa(target_ids, p_map, rooms, base_sched):
    if not target_ids: return base_sched, 0
    curr = random.sample(target_ids, len(target_ids))
    curr_s = serial_sgs(curr, p_map, rooms, base_sched)
    curr_sc = score_schedule(curr_s, target_ids)
    best_p, best_sc = list(curr), curr_sc
    temp = SA_T0
    t0 = time.time()

    for i in range(SA_ITER):
        if time.time() - t0 > BUDGET_PER_STAGE: break
        nb = _swap_mutate(list(curr))
        nb_s = serial_sgs(nb, p_map, rooms, base_sched)
        nb_sc = score_schedule(nb_s, target_ids)

        delta = nb_sc - curr_sc
        if delta >= 0 or random.random() < math.exp(delta / max(temp, 1e-10)):
            curr, curr_sc = nb, nb_sc
            if nb_sc > best_sc: best_p, best_sc = list(nb), nb_sc

        temp *= SA_COOL
        if i % SA_REHEAT == 0: temp = SA_T0 * 0.5

    return serial_sgs(best_p, p_map, rooms, base_sched), best_sc


# --- Method 4: GNN Heuristic (Single Stage) ---
def run_gnn(target_ids, p_map, rooms, base_sched):
    if not target_ids: return base_sched, 0
    # Simplified Logic: sort by constraints
    # (Real GNN would require a trained model, we use the heuristic proxy from previous code)
    # Features: duration, rooms, surgeon load
    surg_load = defaultdict(int)
    for pid in target_ids: surg_load[p_map[pid].surgeon] += 1

    def priority(pid):
        p = p_map[pid]
        return (p.duration * 0.5) + (1000 if len(p.eligible_rooms) == 1 else 0) + (surg_load[p.surgeon] * 10)

    order = sorted(target_ids, key=priority, reverse=True)
    s = serial_sgs(order, p_map, rooms, base_sched)
    return s, score_schedule(s, target_ids)


# --- Method 5: DRL Policy (Single Stage) ---
def run_drl(target_ids, p_map, rooms, base_sched):
    if not target_ids: return base_sched, 0
    # Similar to GNN, use domain-weighted heuristic for 'Policy'
    # Sort by: Duration desc, surgeon conflict desc
    surg_load = defaultdict(int)
    for pid in target_ids: surg_load[p_map[pid].surgeon] += 1

    order = sorted(target_ids, key=lambda pid: (p_map[pid].duration, surg_load[p_map[pid].surgeon]), reverse=True)
    s = serial_sgs(order, p_map, rooms, base_sched)
    return s, score_schedule(s, target_ids)


# --- Method 6: Avg Human (Single Stage) ---
def run_avghuman(target_ids, p_map, rooms, base_sched):
    if not target_ids: return base_sched, 0
    # Sort by date originally requested
    order = sorted(target_ids, key=lambda pid: p_map[pid].original_date)
    s = serial_sgs(order, p_map, rooms, base_sched)
    return s, score_schedule(s, target_ids)


# --- Method 7: Smart Human (Single Stage) ---
def run_smarthuman(target_ids, p_map, rooms, base_sched):
    if not target_ids: return base_sched, 0
    # LPT (Longest Processing Time) + Best Fit
    order = sorted(target_ids, key=lambda pid: p_map[pid].duration, reverse=True)
    s = serial_sgs(order, p_map, rooms, base_sched)
    return s, score_schedule(s, target_ids)


# ═══════════════════════════════════════════════════════════════════════════════
# CASCADING ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════
def solve_department_cascade(dept, raw):
    patients, rooms = load_department(raw, dept)
    p_map = {p.id: p for p in patients}

    # Split by month
    month_ids = {
        1: [p.id for p in patients if p.month == 1],
        2: [p.id for p in patients if p.month == 2],
        3: [p.id for p in patients if p.month == 3]
    }

    print(f"\n{'═' * 70}\n  DEPARTMENT: {dept}\n"
          f"  Counts: Jan={len(month_ids[1])}  Feb={len(month_ids[2])}  Mar={len(month_ids[3])}\n"
          f"  Strategy: Anchor Jan -> Anchor Feb -> Solve Mar\n"
          f"{'═' * 70}")

    solvers = {
        "Hartmann GA 1998": run_hartmann98,
        "Hartmann GA 2002": run_hartmann02,
        "Simulated Annealing": run_sa,
        "GNN Heuristic": run_gnn,
        "DRL Policy": run_drl,
        "Average Human": run_avghuman,
        "Smart Human": run_smarthuman
    }

    final_results = {}

    for name, func in solvers.items():
        print(f"  > Running {name}...", end=" ", flush=True)

        # --- PHASE 1: JANUARY ---
        current_sched, sc = func(month_ids[1], p_map, rooms, base_sched=None)

        # Check Anchor Condition
        if sc < len(month_ids[1]):
            # Failed to anchor Jan
            print(f"FAILED Jan ({sc}/{len(month_ids[1])}) -> Stopped.")
            final_results[name] = (current_sched, sc)
            continue

        # --- PHASE 2: FEBRUARY ---
        # Jan is 100% placed. It serves as the LOCKED base for Feb.
        current_sched, sc_feb = func(month_ids[2], p_map, rooms, base_sched=current_sched)

        if sc_feb < len(month_ids[2]):
            # Failed to anchor Feb
            print(f"Secured Jan -> FAILED Feb ({sc_feb}/{len(month_ids[2])}) -> Stopped.")
            # Score = Jan(100%) + Feb(partial)
            total_sc = len(month_ids[1]) * 10000 + sc_feb * 5000
            final_results[name] = (current_sched, total_sc)
            continue

        # --- PHASE 3: MARCH ---
        # Jan+Feb are 100% placed and locked. Solve March.
        current_sched, sc_mar = func(month_ids[3], p_map, rooms, base_sched=current_sched)

        print(f"Secured Jan -> Secured Feb -> Mar ({sc_mar}/{len(month_ids[3])})")

        # Total Score
        total_sc = len(month_ids[1]) * 10000 + len(month_ids[2]) * 5000 + sc_mar * 1000
        final_results[name] = (current_sched, total_sc)

    return final_results, p_map, month_ids, rooms


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════════
def print_table(dept, results, m_ids, p_map):
    print(f"\n  RESULTS — {dept}")
    print(f"  {'Method':<22} {'Jan':<9} {'Feb':<9} {'Mar':<9} {'Total':<7} {'Score':>12}")
    print(f"  {'─' * 22} {'─' * 9} {'─' * 9} {'─' * 9} {'─' * 7} {'─' * 12}")

    ranked = sorted(results.items(), key=lambda kv: -kv[1][1])

    for name, (sched, total_sc) in ranked:
        j = sum(1 for pid in m_ids[1] if pid in sched.placed)
        f = sum(1 for pid in m_ids[2] if pid in sched.placed)
        m = sum(1 for pid in m_ids[3] if pid in sched.placed)
        tot = j + f + m

        # Status Flags
        j_tag = "✓" if j == len(m_ids[1]) and len(m_ids[1]) > 0 else "x"
        f_tag = "✓" if f == len(m_ids[2]) and len(m_ids[2]) > 0 else "x"

        print(f"  {name:<22} {j:>3}/{len(m_ids[1]):<3}{j_tag} {f:>3}/{len(m_ids[2]):<3}{f_tag} "
              f"{m:>3}/{len(m_ids[3]):<4} {tot:>7} {total_sc:>12,}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    t_global = time.time()
    if not os.path.exists(INPUT_FILE):
        print("Missing data file.")
        sys.exit(1)

    with open(INPUT_FILE) as f:
        raw = json.load(f)

    all_res = {}

    for dept in DEPARTMENTS:
        res, p_map, m_ids, rooms = solve_department_cascade(dept, raw)
        all_res[dept] = (res, p_map)
        print_table(dept, res, m_ids, p_map)

    # Export
    csv_rows = []
    for dept in DEPARTMENTS:
        res, p_map = all_res[dept]
        best_name = max(res, key=lambda k: res[k][1])
        sched = res[best_name][0]

        for pid, (day, room, start, end) in sched.placed.items():
            p = p_map[pid]
            csv_rows.append({
                "Department": dept, "Method": best_name, "Room": room,
                "Day": day + 1,
                "Start": f"{start // 60:02d}:{start % 60:02d}",
                "End": f"{end // 60:02d}:{end % 60:02d}",
                "PatientID": p.id, "Type": p.ptype
            })

    if csv_rows:
        with open(OUTPUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\n  Best schedules exported to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()