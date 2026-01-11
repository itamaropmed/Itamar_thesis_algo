import json
import csv
import random
import os
import time
import sys
from collections import defaultdict
import optuna

# Suppress Optuna logging for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Configuration ---
INPUT_FILE = "impossible_100.json"
OUTPUT_FILE = "ga_tuned_solution.csv"
DAYS_PER_WEEK = 5
MINUTES_PER_DAY = 600


class Patient:
    def __init__(self, d):
        self.id = d["id"]
        self.type = d.get("type", "Standard")
        self.duration = int(d["duration"])
        self.compat_rooms = list(d["compatible_rooms"])
        self.compat_docs = list(d["compatible_doctors"])
        self.dur_mask = (1 << self.duration) - 1


def load_data():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return [], {}, {}
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    return [Patient(p) for p in data["patients"]], data["meta"], data.get("constraints", {})


# =========================================================
# 1. EVALUATOR (Bitwise SGS)
# =========================================================
def serial_sgs_hyperfast(activity_list, meta, constraints):
    day_limit = meta.get("day_limit", 600)
    num_days = DAYS_PER_WEEK
    num_rooms = meta["num_rooms"]
    num_docs = meta["num_doctors"]
    day_mask = (1 << day_limit) - 1

    r_bits = [[0] * num_days for _ in range(num_rooms)]
    d_bits = [[0] * num_days for _ in range(num_docs)]
    r_counts = [[defaultdict(int) for _ in range(num_days)] for _ in range(num_rooms)]

    schedule_pats = {r: [] for r in range(num_rooms)}
    unassigned = 0

    for p in activity_list:
        assigned = False
        p_dur = p.duration
        p_mask = getattr(p, 'dur_mask', (1 << p_dur) - 1)

        for day in range(num_days):
            if assigned: break

            valid_rooms = []
            for r in p.compat_rooms:
                if p.type in constraints and r_counts[r][day][p.type] >= constraints[p.type]:
                    continue
                valid_rooms.append(r)

            # --- FIX: Python < 3.10 Compatibility ---
            valid_rooms.sort(key=lambda r: bin(r_bits[r][day]).count('1'), reverse=True)

            for r in valid_rooms:
                if assigned: break

                docs = p.compat_docs
                if len(docs) > 1:
                    docs = list(docs)
                    random.shuffle(docs)

                for d in docs:
                    busy = r_bits[r][day] | d_bits[d][day]
                    avail = (~busy) & day_mask

                    run_map = avail
                    for _ in range(p_dur - 1):
                        run_map &= (run_map >> 1)
                        if run_map == 0: break

                    if run_map != 0:
                        t = (run_map & -run_map).bit_length() - 1
                        shifted = p_mask << t
                        r_bits[r][day] |= shifted
                        d_bits[d][day] |= shifted
                        r_counts[r][day][p.type] += 1
                        schedule_pats[r].append(p)
                        assigned = True
                        break
        if not assigned:
            unassigned += 1

    return schedule_pats, unassigned


# =========================================================
# 2. GA ENGINE (Parametrized)
# =========================================================
def run_ga_engine(patients, meta, constraints, params):
    # Unpack Tuned Parameters
    pop_size = params['pop_size']
    generations = params['generations']
    mutation_rate = params['mutation_rate']
    elitism_rate = params['elitism_rate']

    # Initialization Strategies
    seq_lpt = sorted(patients, key=lambda x: x.duration, reverse=True)
    seq_scarcity = sorted(patients, key=lambda x: x.duration / ((len(x.compat_rooms) * len(x.compat_docs)) + 0.1),
                          reverse=True)

    population = []
    for _ in range(pop_size):
        if random.random() < 0.5:
            ind = list(seq_scarcity)
        else:
            ind = list(seq_lpt)
        # Shuffle for initial diversity
        for _ in range(5):
            i, j = random.sample(range(len(ind)), 2)
            ind[i], ind[j] = ind[j], ind[i]
        population.append(ind)

    best_sched = None
    best_unass = len(patients) + 1

    for gen in range(generations):
        scored_pop = []
        for ind in population:
            sched, unass = serial_sgs_hyperfast(ind, meta, constraints)
            scored_pop.append((unass, ind))
            if unass < best_unass:
                best_unass = unass
                best_sched = sched

        # Elitism
        scored_pop.sort(key=lambda x: x[0])
        elite_count = max(2, int(pop_size * elitism_rate))
        new_pop = [x[1] for x in scored_pop[:elite_count]]

        # Breeding
        while len(new_pop) < pop_size:
            # Tournament Selection
            p1 = min(random.sample(scored_pop, 5), key=lambda x: x[0])[1]
            p2 = min(random.sample(scored_pop, 5), key=lambda x: x[0])[1]

            # OX1 Crossover
            size = len(p1)
            c1, c2 = sorted(random.sample(range(size), 2))
            child = [None] * size
            child[c1:c2] = p1[c1:c2]
            cur = 0
            for i in range(size):
                if not (c1 <= i < c2):
                    while p2[cur] in child[c1:c2]: cur += 1
                    child[i] = p2[cur]
                    cur += 1

            # Mutation
            if random.random() < mutation_rate:
                a, b = random.sample(range(size), 2)
                child[a], child[b] = child[b], child[a]

            new_pop.append(child)
        population = new_pop

    return best_sched, best_unass


# =========================================================
# 3. OPTUNA OBJECTIVE
# =========================================================
def objective(trial):
    patients, meta, constraints = load_data()

    # Hyperparameters to Tune
    params = {
        'pop_size': trial.suggest_int('pop_size', 50, 200),
        'generations': 50,  # Keep low for fast tuning
        'mutation_rate': trial.suggest_float('mutation_rate', 0.1, 0.5),
        'elitism_rate': trial.suggest_float('elitism_rate', 0.05, 0.2)
    }

    _, unass = run_ga_engine(patients, meta, constraints, params)

    # Return Score (Patients Scheduled)
    return len(patients) - unass


# =========================================================
# 4. MAIN EXECUTION
# =========================================================
def main():
    patients, meta, constraints = load_data()
    if not patients: return

    print(f"--- 1. TUNING GA on {INPUT_FILE} ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)  # 10 Trials to find best params

    best_params = study.best_params
    best_params['generations'] = 500  # Increase gens for final run

    print(f"\n--- Best Parameters Found ---")
    print(best_params)

    print(f"\n--- 2. RUNNING FINAL GA (Gens: {best_params['generations']}) ---")
    start_time = time.time()
    best_sched, unassigned_count = run_ga_engine(patients, meta, constraints, best_params)
    runtime = time.time() - start_time

    scheduled_count = len(patients) - unassigned_count
    print(f"\n>>> Final GA Result: {scheduled_count}/{len(patients)}")
    print(f">>> Runtime: {runtime:.2f}s")

    # Save to CSV
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Assigned", scheduled_count])
        writer.writerow(["Total", len(patients)])
        writer.writerow(["Parameters", str(best_params)])
        writer.writerow([])
        writer.writerow(["Room", "Patient IDs"])

        if best_sched:
            for r in sorted(best_sched.keys()):
                p_list = best_sched[r]
                if p_list:
                    pids = sorted([p.id for p in p_list])
                    writer.writerow([r, ";".join(map(str, pids))])

    print(f"Schedule saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()