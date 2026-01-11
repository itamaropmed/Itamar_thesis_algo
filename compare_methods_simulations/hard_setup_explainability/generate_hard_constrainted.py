import json
import random
import os


def generate_impossible_setup():
    filename = "impossible_100.json"

    # Configuration
    num_patients = 100
    num_rooms = 5  # Reduced rooms to limit capacity
    num_doctors = 15
    day_limit = 600

    # Calculate Limits
    total_capacity = num_rooms * 5 * day_limit  # 15,000 mins
    target_demand = int(total_capacity * 1.4)  # Aim for 140% demand (21,000 mins)
    avg_duration = target_demand // num_patients  # ~210 mins per patient

    patients = []

    for i in range(num_patients):
        # Durations around 210 mins (some small, some large)
        dur = random.randint(180, 240)

        # Randomly assign compatible rooms/docs
        # We ensure they aren't TOO flexible to keep it hard
        n_rooms = random.randint(1, 2)
        rooms = random.sample(range(num_rooms), n_rooms)

        n_docs = random.randint(1, 3)
        docs = random.sample(range(num_doctors), n_docs)

        patients.append({
            "id": i,
            "type": "Standard",
            "duration": dur,
            "compatible_rooms": rooms,
            "compatible_doctors": docs
        })

    data = {
        "meta": {
            "num_patients": num_patients,
            "num_rooms": num_rooms,
            "num_doctors": num_doctors,
            "day_limit": day_limit,
            "difficulty": "Impossible - Capacity Overload (140%)"
        },
        "patients": patients
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

    total_dur = sum(p['duration'] for p in patients)
    print(f"Generated {filename}")
    print(f"Total Capacity: {total_capacity} mins")
    print(f"Total Demand:   {total_dur} mins")
    print(f"Overload:       {total_dur / total_capacity:.1%}")
    print("It is mathematically impossible to schedule everyone.")


if __name__ == "__main__":
    generate_impossible_setup()