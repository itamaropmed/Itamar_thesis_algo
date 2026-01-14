import pandas as pd
import os


def convert_csv_to_json(csv_filename):
    # 1. Check if the CSV file exists
    if not os.path.exists(csv_filename):
        print(f"Error: The file '{csv_filename}' was not found in the current directory.")
        return

    # 2. Read the CSV file
    try:
        df = pd.read_csv(csv_filename)
        print(f"Successfully read {csv_filename}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 3. Save as JSON
    # orient='records' creates a list of dictionaries (one per row)
    # indent=4 makes it human-readable
    json_filename = csv_filename.replace('.csv', '.json')
    df.to_json(json_filename, orient='records', indent=4)

    print(f"Successfully saved JSON file to: {json_filename}")


if __name__ == "__main__":
    file_name = 'informative_data.csv'
    convert_csv_to_json(file_name)