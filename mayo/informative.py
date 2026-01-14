import pandas as pd
import os


def create_informative_data():
    # 1. Define the filename (ensure this matches your file exactly)
    excel_filename = 'HRS and CCL Cases Jan 2022 - April 2025 from Sue.xlsx'

    # Debug: Print current directory to ensure we know where we are looking
    print(f"Current Working Directory: {os.getcwd()}")

    # 2. Check if file exists before trying to read
    if not os.path.exists(excel_filename):
        print(f"\nERROR: The file '{excel_filename}' was not found.")
        print("Please ensure the Excel file is inside the same folder as this script.")
        print(f"Looking in: {os.getcwd()}")
        # Check if it might be in the parent directory
        if os.path.exists(os.path.join('..', excel_filename)):
            print(f"Found it in the parent directory! Using '../{excel_filename}'")
            excel_filename = os.path.join('..', excel_filename)
        else:
            return

    # 3. Read the Excel file
    print(f"Reading '{excel_filename}'... (this might take a moment)")
    try:
        df = pd.read_excel(excel_filename, engine='openpyxl')
    except ImportError:
        print("\nERROR: 'openpyxl' library is missing.")
        print("Please run this command in your terminal: pip install openpyxl")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    # 4. Select the specific columns you requested
    target_columns = [
        'Date',
        'Day of Week',
        'Discharge Location',
        'Patient Class',
        'OR Department',
        'Room',
        'Lead Surgeon/Provider',
        'Scheduled Procedure',
        'Actual Procedures',
        'In Proc Room',
        'Proc Start',
        'Proc Comp',
        'Out Proc Room'
    ]

    # Verify columns exist
    missing_cols = [c for c in target_columns if c not in df.columns]
    if missing_cols:
        print(f"Warning: The following columns were not found: {missing_cols}")

    # 5. Create and Save the DataFrame
    # Filter only existing columns to prevent crash
    existing_cols = [c for c in target_columns if c in df.columns]
    df_informative = df[existing_cols].copy()

    output_filename = 'informative_data.csv'
    df_informative.to_csv(output_filename, index=False)

    print(f"\nSuccess! '{output_filename}' has been created.")
    print(f"Rows: {len(df_informative)}")
    print(df_informative.head())


if __name__ == "__main__":
    create_informative_data()
