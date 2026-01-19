import json


def validate_data():
    try:
        with open('global_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: global_data.json not found.")
        return

    patients = data.get('patients', {})
    treatments = data.get('treatments', {})
    providers = data.get('providers', {})

    # Create a map of treatment_code -> list of provider_ids
    treatment_to_providers = {}
    for pid, p_data in providers.items():
        disciplines = p_data.get('disciplines', [])
        for code in disciplines:
            if code not in treatment_to_providers:
                treatment_to_providers[code] = []
            treatment_to_providers[code].append(pid)

    missing_treatments_in_db = set()
    treatments_missing_details = set()
    treatments_with_no_provider = set()

    # Iterate through all patients and their rehab plans
    for pat_id, pat_data in patients.items():
        rehab_plan = pat_data.get('rehab_plan', [])

        for t_code in rehab_plan:
            # 1. Validate existence in treatments object
            if t_code not in treatments:
                missing_treatments_in_db.add(t_code)
            else:
                # 2. Validate details (type and duration)
                t_details = treatments[t_code]
                if 'type' not in t_details or 'duration' not in t_details:
                    treatments_missing_details.add(t_code)

            # 3. Validate provider existence
            if t_code not in treatment_to_providers:
                treatments_with_no_provider.add(t_code)

    # Print Report
    print("Validation Report:")
    print("------------------")

    if missing_treatments_in_db:
        print(
            f"FAILED: Found {len(missing_treatments_in_db)} treatments in rehab plans that are missing from the 'treatments' list:")
        print(list(missing_treatments_in_db))
    else:
        print("PASS: All rehab_plan treatments exist in 'treatments'.")

    if treatments_missing_details:
        print(f"FAILED: Found {len(treatments_missing_details)} treatments missing 'type' or 'duration':")
        print(list(treatments_missing_details))
    else:
        print("PASS: All found treatments have type and duration.")

    if treatments_with_no_provider:
        print(f"FAILED: Found {len(treatments_with_no_provider)} treatments in rehab plans with NO capable provider:")
        print(list(treatments_with_no_provider))
    else:
        print("PASS: All rehab_plan treatments have at least one provider.")


if __name__ == "__main__":
    validate_data()