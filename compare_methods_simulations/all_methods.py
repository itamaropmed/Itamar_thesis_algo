import json
import random
import math
import sys
import os
import time
import optuna
import pandas as pd
from collections import deque

# Suppress optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ------------------------------------------------------------------------------
# 1. DATA LOADING & RESOURCE MODEL
# ------------------------------------------------------------------------------

def load_data(filename='large_setup_250.json'):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


class ProjectInstance:
    def __init__(self, data):
        self.meta = data['meta']
        self.patients = data['patients']
        self.num_patients = len(self.patients)
        self.minutes_per_day = self.meta['day_limit']
        self.num_days = 5
        self.num_rooms = self.meta['num_rooms']
        self.num_doctors = self.meta['num_doctors']
        self.patient_map = {p['id']: p for p in self.patients}

    def evaluate_schedule(self, activity_list, time_step=1):
        """
        Standard Serial SGS evaluator.
        Returns: succeeded_count
        """
        room_usage = [[[False] * self.minutes_per_day for _ in range(self.num_rooms)] for _ in range(self.num_days)]
        doc_usage = [[[False] * self.minutes_per_day for _ in range(self.num_doctors)] for _ in range(self.num_days)]

        succeeded_count = 0

        for pid in activity_list:
            p = self.patient_map[pid]
            dur = p['duration']
            scheduled = False

            for d in range(self.num_days):
                comp_rooms = p['compatible_rooms']
                comp_docs = p['compatible_doctors']

                # --- THE HUMAN FACTOR ---
                for t in range(0, self.minutes_per_day - dur + 1, time_step):

                    # 1. Check Room
                    valid_room = -1
                    for r in comp_rooms:
                        if not any(room_usage[d][r][t:t + dur]):
                            valid_room = r
                            break
                    if valid_room == -1: continue

                    # 2. Check Doc
                    valid_doc = -1
                    for doc in comp_docs:
                        if not any(doc_usage[d][doc][t:t + dur]):
                            valid_doc = doc
                            break
                    if valid_doc == -1: continue

                    # 3. Book Resources
                    for k in range(t, t + dur):
                        room_usage[d][valid_room][k] = True
                        doc_usage[d][valid_doc][k] = True

                    succeeded_count += 1
                    scheduled = True
                    break
                if scheduled: break

        return succeeded_count

    def get_schedule_stats(self, activity_list, time_step=1):
        """
        Runs the SGS exactly like evaluate_schedule but returns detailed metrics
        (Scheduled, Makespan, Utilization, Valid) for the CSV report.
        """
        room_usage = [[[False] * self.minutes_per_day for _ in range(self.num_rooms)] for _ in range(self.num_days)]
        doc_usage = [[[False] * self.minutes_per_day for _ in range(self.num_doctors)] for _ in range(self.num_days)]

        scheduled_patients = []

        for pid in activity_list:
            p = self.patient_map[pid]
            dur = p['duration']
            scheduled = False

            for d in range(self.num_days):
                comp_rooms = p['compatible_rooms']
                comp_docs = p['compatible_doctors']

                for t in range(0, self.minutes_per_day - dur + 1, time_step):
                    valid_room = -1
                    for r in comp_rooms:
                        if not any(room_usage[d][r][t:t + dur]):
                            valid_room = r
                            break
                    if valid_room == -1: continue

                    valid_doc = -1
                    for doc in comp_docs:
                        if not any(doc_usage[d][doc][t:t + dur]):
                            valid_doc = doc
                            break
                    if valid_doc == -1: continue

                    for k in range(t, t + dur):
                        room_usage[d][valid_room][k] = True
                        doc_usage[d][valid_doc][k] = True

                    scheduled_patients.append({
                        'day': d,
                        'end_time': t + dur,
                        'duration': dur
                    })
                    scheduled = True
                    break
                if scheduled: break

        # --- Calculate Metrics ---
        count = len(scheduled_patients)

        if count > 0:
            # Makespan is the last minute anyone is busy in the entire week
            makespan = max(p['day'] * self.minutes_per_day + p['end_time'] for p in scheduled_patients)
        else:
            makespan = 0

        total_duration_scheduled = sum(p['duration'] for p in scheduled_patients)
        # Total Capacity = Rooms * Days * Minutes
        total_capacity = self.num_rooms * self.num_days * self.minutes_per_day
        utilization = total_duration_scheduled / total_capacity if total_capacity > 0 else 0.0

        return {
            'Scheduled': count,
            'Makespan': makespan,
            'Utilization': f"{utilization:.2%}",
            'Valid': True
        }


# ------------------------------------------------------------------------------
# 2. METAHEURISTICS (GA & SA)
# ------------------------------------------------------------------------------

class HartmannGA:
    def __init__(self, instance, pop_size=30, generations=15, mutation_rate=0.05):
        self.inst = instance
        self.pop_size = pop_size
        self.generations = generations
        self.mut_rate = mutation_rate
        self.population = []

    def run(self):
        base_list = [p['id'] for p in self.inst.patients]
        # Init
        for _ in range(self.pop_size):
            perm = base_list[:]
            random.shuffle(perm)
            score = self.inst.evaluate_schedule(perm, time_step=1)
            self.population.append({'genes': perm, 'score': score})

        for gen in range(self.generations):
            self.population.sort(key=lambda x: x['score'], reverse=True)
            self.population = self.population[:self.pop_size]

            offspring = []
            while len(offspring) < self.pop_size:
                p1, p2 = random.sample(self.population, 2)
                s1, s2 = p1['genes'], p2['genes']
                size = len(s1)
                cx1, cx2 = sorted([random.randint(0, size - 1) for _ in range(2)])

                child_seq = [None] * size
                child_seq[cx1:cx2] = s1[cx1:cx2]
                current = 0
                for i in range(size):
                    if child_seq[i] is None:
                        while s2[current] in child_seq[cx1:cx2]: current += 1
                        child_seq[i] = s2[current]
                        current += 1

                if random.random() < self.mut_rate:
                    i, j = random.sample(range(size), 2)
                    child_seq[i], child_seq[j] = child_seq[j], child_seq[i]

                score = self.inst.evaluate_schedule(child_seq, time_step=1)
                offspring.append({'genes': child_seq, 'score': score})

            self.population.extend(offspring)

        # Return best score AND the sequence that produced it
        best_ind = max(self.population, key=lambda x: x['score'])
        return best_ind['score'], best_ind['genes']


class BouleimenSA:
    def __init__(self, instance, T0=50, h=0.1, chains=2, steps=5):
        self.inst = instance
        self.T0 = T0
        self.h = h
        self.chains = chains
        self.steps = steps

    def run(self):
        best_score = 0
        best_list = []  # Track the best solution found

        for _ in range(self.chains):
            curr_list = [p['id'] for p in self.inst.patients]
            random.shuffle(curr_list)
            curr_score = self.inst.evaluate_schedule(curr_list, time_step=1)

            if curr_score > best_score:
                best_score = curr_score
                best_list = curr_list[:]

            T = self.T0
            Ns = 5
            for s in range(self.steps):
                limit = min(int(Ns * (1 + self.h * s)), 10)
                for _ in range(limit):
                    neighbor = curr_list[:]
                    i = random.randint(0, len(neighbor) - 1)
                    val = neighbor.pop(i)
                    j = random.randint(0, len(neighbor))
                    neighbor.insert(j, val)

                    new_score = self.inst.evaluate_schedule(neighbor, time_step=1)
                    delta = -new_score - (-curr_score)

                    if delta < 0 or random.random() < math.exp(-delta / T):
                        curr_list = neighbor
                        curr_score = new_score
                        if curr_score > best_score:
                            best_score = curr_score
                            best_list = curr_list[:]
                T *= 0.9

        return best_score, best_list


# ------------------------------------------------------------------------------
# 3. HUMAN HEURISTICS
# ------------------------------------------------------------------------------

class HumanHeuristics:
    def __init__(self, instance):
        self.inst = instance

    def run_average(self):
        print("   -> Simulating Average Human (FIFO + 30min Blocks)...")
        p_sorted = sorted(self.inst.patients, key=lambda x: x['id'])
        lst = [p['id'] for p in p_sorted]
        # Return both the list and the score for detailed analysis
        return self.inst.evaluate_schedule(lst, time_step=30), lst

    def run_smart(self):
        print("   -> Simulating Smart Human (Smallest First + Precision)...")
        list_spt = sorted(self.inst.patients, key=lambda x: x['duration'])
        lst = [p['id'] for p in list_spt]
        return self.inst.evaluate_schedule(lst, time_step=1), lst


# ------------------------------------------------------------------------------
# 4. MAIN
# ------------------------------------------------------------------------------

def objective_ga(trial, inst):
    pop = trial.suggest_int('pop', 30, 60)
    mut = trial.suggest_float('mut', 0.05, 0.2)
    ga = HartmannGA(inst, pop_size=pop, mutation_rate=mut, generations=5)
    return ga.run()[0]  # Return only score for Optuna


def objective_sa(trial, inst):
    t0 = trial.suggest_float('t0', 50, 150)
    h = trial.suggest_float('h', 0.1, 0.3)
    sa = BouleimenSA(inst, T0=t0, h=h, chains=1, steps=3)
    return sa.run()[0]  # Return only score for Optuna


def main():
    try:
        data = load_data('large_setup_250.json')
    except Exception as e:
        print(f"Error: {e}")
        return

    inst = ProjectInstance(data)
    print(f"Setup: 250 Patients, 5 Days, 600 Mins/Day")
    print("-" * 40)

    # Tuning
    print("Tuning Hartmann GA...")
    study_ga = optuna.create_study(direction='maximize')
    study_ga.optimize(lambda t: objective_ga(t, inst), n_trials=3)

    print("Tuning Bouleimen SA...")
    study_sa = optuna.create_study(direction='maximize')
    study_sa.optimize(lambda t: objective_sa(t, inst), n_trials=3)

    # Final Run & Data Collection
    print("-" * 40)
    print("Running Final Simulations...")
    final_stats = []

    # 1. GA
    start_time = time.time()
    p_ga = study_ga.best_params
    ga = HartmannGA(inst, pop_size=p_ga['pop'], mutation_rate=p_ga['mut'], generations=30)
    _, best_list_ga = ga.run()
    stats_ga = inst.get_schedule_stats(best_list_ga, time_step=1)
    stats_ga['Method'] = 'Hartmann 2002 (GA)'
    stats_ga['Runtime'] = time.time() - start_time
    final_stats.append(stats_ga)

    # 2. SA
    start_time = time.time()
    p_sa = study_sa.best_params
    sa = BouleimenSA(inst, T0=p_sa['t0'], h=p_sa['h'], chains=3, steps=10)
    _, best_list_sa = sa.run()
    stats_sa = inst.get_schedule_stats(best_list_sa, time_step=1)
    stats_sa['Method'] = 'Bouleimen 2003 (SA)'
    stats_sa['Runtime'] = time.time() - start_time
    final_stats.append(stats_sa)

    # 3. Humans
    hh = HumanHeuristics(inst)

    # Average Human
    start_time = time.time()
    _, list_avg = hh.run_average()
    stats_avg = inst.get_schedule_stats(list_avg, time_step=30)  # 30 min step
    stats_avg['Method'] = 'Average Human'
    stats_avg['Runtime'] = time.time() - start_time
    final_stats.append(stats_avg)

    # Smart Human
    start_time = time.time()
    _, list_smart = hh.run_smart()
    stats_smart = inst.get_schedule_stats(list_smart, time_step=1)  # 1 min step
    stats_smart['Method'] = 'Smart Human'
    stats_smart['Runtime'] = time.time() - start_time
    final_stats.append(stats_smart)

    # Output CSV
    df = pd.DataFrame(final_stats)
    # Reorder columns to match the target format
    df = df[['Method', 'Scheduled', 'Makespan', 'Utilization', 'Runtime', 'Valid']]

    print("\n" + df.to_string(index=False))
    df.to_csv('algorithm_results_5day.csv', index=False)
    print("\nResults saved to 'algorithm_results_5day.csv'")


if __name__ == "__main__":
    main()