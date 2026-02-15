import json
import csv
import time
import os
import sys
from datetime import datetime

# ---------------------------------------------------------
# 1. IMPORT THE SOLVER
# ---------------------------------------------------------
try:
    import methods_solvers_mayo as engine

    print(">>> Successfully imported 'methods_solvers_mayo.py'")
except ImportError:
    print("!!! CRITICAL ERROR !!!")
    print("Could not find 'methods_solvers_mayo.py'.")
    print("Please save your solver code as 'methods_solvers_mayo.py' in this folder.")
    sys.exit(1)

# ---------------------------------------------------------
# 2. CONFIGURATION
# ---------------------------------------------------------
ITERATIONS = 30
INPUT_FILE = "informative_data.json"
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
CSV_FILENAME = f"experiment_results_{TIMESTAMP}.csv"
JSON_FILENAME = f"experiment_results_{TIMESTAMP}.json"

DEPARTMENTS = ["RST ROMB CCL", "RST ROMB HRS"]


def run_experiment():
    print(f"\n=== M-ORSP 30-ITERATION EXPERIMENT ===")
    print(f"Goal: {ITERATIONS} Iterations per Department")
    print(f"Target Departments: {DEPARTMENTS}")

    # Load Data Once
    if not os.path.exists(INPUT_FILE):
        print("Error: Input file 'informative_data.json' not found.")
        return

    with open(INPUT_FILE, 'r') as f:
        raw_data = json.load(f)

    # Storage for JSON output
    all_records = []

    # Initialize CSV with Headers
    headers = [
        "Run_ID",
        "Department",
        "Method",
        "Jan_Count",
        "Feb_Count",
        "Mar_Count",
        "Total_Placed",
        "Jan_Coverage_Pct",
        "Score",
        "Runtime_Sec"
    ]

    # Create File and Write Headers
    with open(CSV_FILENAME, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    start_time_global = time.time()

    # ---------------------------------------------------------
    # 3. EXPERIMENT LOOP
    # ---------------------------------------------------------
    for run_id in range(1, ITERATIONS + 1):
        print(f"\n>>> EXPERIMENT ITERATION {run_id}/{ITERATIONS}")

        for dept in DEPARTMENTS:
            t0 = time.time()

            # --- CALL THE SOLVER ---
            # This function runs all 7 methods in the cascade (Jan->Feb->Mar)
            try:
                # Returns: (results_dict, p_map, month_ids_dict, rooms_list)
                results, p_map, m_ids, rooms = engine.solve_department_cascade(dept, raw_data)
            except AttributeError:
                print("Error: 'methods_solvers_mayo.py' appears to be missing the 'solve_department_cascade' function.")
                sys.exit(1)
            except Exception as e:
                print(f"Error running solver for {dept}: {e}")
                continue

            t_dept = time.time() - t0

            # --- PROCESS RESULTS FOR THIS DEPARTMENT ---
            for method_name, (sched, total_score) in results.items():
                # 1. Calculate Granular Counts
                # m_ids structure is {1: [ids], 2: [ids], 3: [ids]}
                # We check if each patient ID is in the 'sched.placed' dictionary

                j_sched = sum(1 for pid in m_ids[1] if pid in sched.placed)
                f_sched = sum(1 for pid in m_ids[2] if pid in sched.placed)
                m_sched = sum(1 for pid in m_ids[3] if pid in sched.placed) if 3 in m_ids else 0

                total_placed = j_sched + f_sched + m_sched

                # Calculate Coverage Percentage for January (The Anchor)
                jan_total_demand = len(m_ids[1])
                jan_pct = (j_sched / jan_total_demand * 100) if jan_total_demand > 0 else 0.0

                # Estimate runtime per method (Total Dept Time / 7 methods)
                est_runtime = t_dept / len(results)

                # 2. Prepare Record
                record = {
                    "Run_ID": run_id,
                    "Department": dept,
                    "Method": method_name,
                    "Jan_Count": j_sched,
                    "Feb_Count": f_sched,
                    "Mar_Count": m_sched,
                    "Total_Placed": total_placed,
                    "Jan_Coverage_Pct": round(jan_pct, 1),
                    "Score": total_score,
                    "Runtime_Sec": round(est_runtime, 2)
                }

                all_records.append(record)

                # 3. Save to CSV immediately (Append mode)
                with open(CSV_FILENAME, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row = [
                        record["Run_ID"],
                        record["Department"],
                        record["Method"],
                        record["Jan_Count"],
                        record["Feb_Count"],
                        record["Mar_Count"],
                        record["Total_Placed"],
                        record["Jan_Coverage_Pct"],
                        record["Score"],
                        record["Runtime_Sec"]
                    ]
                    writer.writerow(row)

    # ---------------------------------------------------------
    # 4. FINAL JSON DUMP
    # ---------------------------------------------------------
    with open(JSON_FILENAME, 'w') as f:
        json.dump(all_records, f, indent=4)

    total_time = time.time() - start_time_global
    print(f"\n=== EXPERIMENT COMPLETE ===")
    print(f"Total Experiment Time: {total_time / 60:.1f} minutes")
    print(f"Detailed CSV saved to: {CSV_FILENAME}")
    print(f"Full JSON saved to:    {JSON_FILENAME}")


if __name__ == "__main__":
    run_experiment()