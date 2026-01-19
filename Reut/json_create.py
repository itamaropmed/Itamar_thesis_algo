import pandas as pd
import json
import os
import datetime


def generate_global_json():
    # ---------------------------------------------------------
    # 1. Initialize Global Data Structure
    # ---------------------------------------------------------
    data = {
        "patients": {},
        "providers": {},
        "treatments": {}
    }

    # Helper to convert time/date objects to string for JSON
    def clean_val(val):
        if pd.isna(val):
            return None
        if isinstance(val, (datetime.time, datetime.datetime, pd.Timestamp)):
            return str(val)
        return val

    # Helper function to try reading a file (handles csv/xlsx variations)
    def load_file(base_name):
        candidates = [
            base_name,
            base_name.replace('.csv', '.xlsx'),
            base_name.replace('.xlsx', '.csv')
        ]
        for fname in candidates:
            if os.path.exists(fname):
                print(f"Loading: {fname}")
                if fname.endswith('.csv'):
                    df = pd.read_csv(fname)
                else:
                    df = pd.read_excel(fname)
                # DEDUPLICATE: Remove exact duplicate rows immediately
                return df.drop_duplicates()

        print(f"Warning: Could not find '{base_name}' (or variations) in {os.getcwd()}")
        return None

    # ---------------------------------------------------------
    # 2. Process Patients Care Plan (Rehab Plan Only)
    # ---------------------------------------------------------
    df_care_plan = load_file('patients_care_plan_expanded.xlsx')
    if df_care_plan is not None:
        df_care_plan['idclient'] = df_care_plan['idclient'].astype(str)

        grouped_care = df_care_plan.groupby('idclient')['treatment_code'].apply(list).to_dict()

        for pid, codes in grouped_care.items():
            clean_codes = [c for c in codes if pd.notna(c)]

            if pid not in data['patients']:
                data['patients'][pid] = {}

            data['patients'][pid]['rehab_plan'] = clean_codes

    # ---------------------------------------------------------
    # 3. Process Main Care Givers
    # ---------------------------------------------------------
    df_main_care = load_file('main_care_giver_expanded.xlsx')
    if df_main_care is not None:
        df_main_care['idclient'] = df_main_care['idclient'].astype(str)
        care_cols_map = {
            'idoved_main_phisio': 'main_phisio',
            'idoved_second_phisio': 'second_phisio',
            'idoved_main_ot': 'main_ot',
            'idoved_second_ot': 'second_ot',
            'idoved_doctor': 'doctor',
            'idoved_nurse': 'nurse',
            'idoved_psycho': 'psycho',
            'idoved_sw': 'sw'
        }

        for _, row in df_main_care.iterrows():
            pid = row['idclient']
            if pid not in data['patients']:
                data['patients'][pid] = {}

            for col, feature_name in care_cols_map.items():
                if col in df_main_care.columns:
                    val = row[col]
                    if pd.isna(val):
                        data['patients'][pid][feature_name] = []
                    else:
                        data['patients'][pid][feature_name] = str(int(val)) if isinstance(val, (int, float)) else str(
                            val)

    # ---------------------------------------------------------
    # 4. Process Treatments List (Type & Duration)
    # ---------------------------------------------------------
    df_treatments = load_file('treatments_list_expanded.xlsx')
    if df_treatments is not None:
        for _, row in df_treatments.iterrows():
            t_code = str(row['treatment_code'])
            t_type_raw = str(row['type'])
            t_duration = row['duration']

            # Type logic
            t_type = "personal" if "פרטני" in t_type_raw else "group"

            if t_code not in data['treatments']:
                data['treatments'][t_code] = {}

            data['treatments'][t_code]['type'] = t_type
            data['treatments'][t_code]['duration'] = t_duration

    # ---------------------------------------------------------
    # 5. Process Provider Disciplines
    # ---------------------------------------------------------
    df_valid_treatments = load_file('valid_treatments_for_care_giver_expanded.xlsx')
    if df_valid_treatments is not None:
        df_valid_treatments['idoved'] = df_valid_treatments['idoved'].astype(str)
        grouped_disciplines = df_valid_treatments.groupby('idoved')['treatment_code'].apply(list).to_dict()

        for prov_id, codes in grouped_disciplines.items():
            if prov_id not in data['providers']:
                data['providers'][prov_id] = {}

            clean_codes = [c for c in codes if pd.notna(c)]
            data['providers'][prov_id]['disciplines'] = clean_codes

    # ---------------------------------------------------------
    # 6. Process Patient Constraints (Convert Hebrew Days)
    # ---------------------------------------------------------
    df_constraints = load_file('patients_constraints_expanded.xlsx')
    if df_constraints is not None:
        df_constraints['idclient'] = df_constraints['idclient'].astype(str)

        constraint_cols = [
            'not_before', 'not_after', 'td1', 'td2', 'td3', 'td4',
            'td1_not_before', 'td1_not_after', 'td2_not_before', 'td2_not_after',
            'td3_not_before', 'td3_not_after', 'td4_not_before', 'td4_not_after'
        ]

        # Day Mapping for td columns
        day_map_constraints = {'א': 1, 'ב': 2, 'ג': 3, 'ד': 4, 'ה': 5, 'ו': 6, 'ש': 7}

        for _, row in df_constraints.iterrows():
            pid = row['idclient']
            if pid in data['patients']:
                for col in constraint_cols:
                    if col in df_constraints.columns:
                        val = row[col]

                        if pd.isna(val):
                            data['patients'][pid][col] = []
                        else:
                            if col.startswith('td') and 'not' not in col:
                                val_str = str(val).strip()
                                mapped_val = day_map_constraints.get(val_str, val)
                                data['patients'][pid][col] = mapped_val
                            else:
                                data['patients'][pid][col] = clean_val(val)

    # ---------------------------------------------------------
    # 7. Process Active Days (Work Days)
    # ---------------------------------------------------------
    df_active = load_file('active_days_of_care_givers_expanded.xlsx')
    if df_active is not None:
        df_active['idoved'] = df_active['idoved'].astype(str)
        for _, row in df_active.iterrows():
            prov_id = row['idoved']
            day = row['day']
            start = clean_val(row['start'])
            end = clean_val(row['end'])

            if prov_id not in data['providers']:
                data['providers'][prov_id] = {}

            if 'work_days' not in data['providers'][prov_id]:
                data['providers'][prov_id]['work_days'] = []

            data['providers'][prov_id]['work_days'].append((day, start, end))

    # ---------------------------------------------------------
    # 8. Process Administrative Constraints (Breaks)
    # ---------------------------------------------------------
    df_admin = load_file('administrative_constrains_of_care_givers_expanded.xlsx')
    if df_admin is not None:
        df_admin['idoved'] = df_admin['idoved'].astype(str)
        for _, row in df_admin.iterrows():
            prov_id = row['idoved']
            date_str = str(row['date'])
            start = clean_val(row['start'])
            duration = row['duration']

            try:
                dt = pd.to_datetime(date_str)
                day_num = (dt.isoweekday() % 7) + 1
            except:
                day_num = 1

            if prov_id not in data['providers']:
                data['providers'][prov_id] = {}

            if 'breaks' not in data['providers'][prov_id]:
                data['providers'][prov_id]['breaks'] = []

            data['providers'][prov_id]['breaks'].append((day_num, start, duration))

    # ---------------------------------------------------------
    # 9. Process Groups List (Flattened Format & Unique Providers)
    # ---------------------------------------------------------
    df_groups = load_file('groups_list_expanded.xlsx')
    day_map_groups = {'א': 1, 'ב': 2, 'ג': 3, 'ד': 4, 'ה': 5, 'ו': 6, 'ש': 7}

    if df_groups is not None:
        for _, row in df_groups.iterrows():
            t_code = str(row['treatment_code'])

            # Map Day
            day_char = str(row['day']).strip()
            day_val = day_map_groups.get(day_char, day_char)

            start = clean_val(row.get('start'))
            end = clean_val(row.get('end'))
            provider = row.get('provider')
            min_p = row.get('min')
            max_p = row.get('max')

            if t_code not in data['treatments']:
                data['treatments'][t_code] = {}

            # Ensure lists exist
            if 'day' not in data['treatments'][t_code]: data['treatments'][t_code]['day'] = []
            if 'start' not in data['treatments'][t_code]: data['treatments'][t_code]['start'] = []
            if 'end' not in data['treatments'][t_code]: data['treatments'][t_code]['end'] = []

            # Use a Set for providers temporarily to handle uniqueness
            if 'provider_set' not in data['treatments'][t_code]: data['treatments'][t_code]['provider_set'] = set()

            # Append Schedule
            data['treatments'][t_code]['day'].append(day_val)
            data['treatments'][t_code]['start'].append(start)
            data['treatments'][t_code]['end'].append(end)

            # Add Provider to Set (Unique)
            if pd.notna(provider):
                # If provider is float/int, convert to clean int string
                p_str = str(int(provider)) if isinstance(provider, (int, float)) else str(provider)
                data['treatments'][t_code]['provider_set'].add(p_str)

            # Set Min/Max
            if pd.notna(min_p): data['treatments'][t_code]['min'] = int(min_p)
            if pd.notna(max_p): data['treatments'][t_code]['max'] = int(max_p)

    # ---------------------------------------------------------
    # 10. Post-Validation and Cleanup for Groups
    # ---------------------------------------------------------
    for t_code, t_data in data['treatments'].items():
        if t_data.get('type') == 'group':
            # Initialize lists if they don't exist
            if 'day' not in t_data: t_data['day'] = []
            if 'start' not in t_data: t_data['start'] = []
            if 'end' not in t_data: t_data['end'] = []

            # Convert provider_set to list 'provider'
            if 'provider_set' in t_data:
                t_data['provider'] = list(t_data['provider_set'])
                del t_data['provider_set']
            else:
                t_data['provider'] = []

            # Set defaults
            if 'min' not in t_data: t_data['min'] = 2
            if 'max' not in t_data: t_data['max'] = 8

    # ---------------------------------------------------------
    # 11. Save to JSON
    # ---------------------------------------------------------
    with open('global_reut_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        print("Successfully created 'global_data.json'")


if __name__ == "__main__":
    generate_global_json()