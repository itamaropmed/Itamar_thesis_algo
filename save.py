import pandas as pd
import os
import matplotlib.pyplot as plt


def analyze_and_save_data(file_name):
    # 1. Create directory 'data'
    output_folder = 'data'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    # 2. Read the file
    # The code checks for the uploaded CSV name first, then tries the XLSX name if needed
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_name)
        else:
            df = pd.read_excel(file_name)
        print(f"Successfully read {file_name}")
    except FileNotFoundError:
        # Fallback logic if the exact name differs locally
        alt_name = file_name.replace('.csv', '').replace(' - Sheet1', '')
        if os.path.exists(alt_name):
            print(f"File {file_name} not found, reading {alt_name} instead...")
            df = pd.read_excel(alt_name)
        else:
            print(f"Error: Could not find file {file_name} or {alt_name} in the current folder.")
            return

    # 3. Save the DataFrame to the data folder
    output_csv_path = os.path.join(output_folder, 'saved_dataframe.csv')
    df.to_csv(output_csv_path, index=False)
    print(f"DataFrame saved to: {output_csv_path}")

    # 4 & 5. Statistics and Plots
    stats_file_path = os.path.join(output_folder, 'statistics.txt')

    print("\n--- Generating Statistics and Plots ---")

    with open(stats_file_path, 'w', encoding='utf-8') as f:
        for col in df.columns:
            # Calculate Statistics
            unique_values = df[col].unique()
            num_unique = len(unique_values)

            # Prepare statistic string
            stat_str = f"Feature Column: {col}\n"
            stat_str += f"Count of Unique Values: {num_unique}\n"

            if num_unique < 50:
                stat_str += f"Unique Values: {list(unique_values)}\n"
            else:
                stat_str += f"First 10 Unique Values: {list(unique_values[:10])} ... (truncated)\n"
            stat_str += "-" * 40 + "\n"

            # Print to console and write to file
            print(stat_str)
            f.write(stat_str)

            # Generate Plots
            # Sanitize column name for filename
            safe_col_name = "".join([c if c.isalnum() else "_" for c in col])
            plot_path = os.path.join(output_folder, f"dist_{safe_col_name}.png")

            plt.figure(figsize=(10, 6))
            try:
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Histogram for numeric data
                    df[col].plot(kind='hist', bins=20, color='skyblue', edgecolor='black')
                    plt.title(f'Distribution of {col}')
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    plt.tight_layout()
                    plt.savefig(plot_path)
                    plt.close()
                elif num_unique <= 20:
                    # Bar chart for categorical data with few unique values
                    df[col].value_counts().plot(kind='bar', color='lightgreen', edgecolor='black')
                    plt.title(f'Count of {col}')
                    plt.xlabel(col)
                    plt.ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(plot_path)
                    plt.close()
                else:
                    # Skip plotting for high cardinality non-numeric data (e.g., specific dates, unique IDs)
                    plt.close()
            except Exception as e:
                plt.close()
                print(f"Could not create plot for {col}: {e}")

    print(f"\nStatistics saved to: {stats_file_path}")
    print(f"Plots saved in folder: {output_folder}")


if __name__ == "__main__":
    # Use the filename as uploaded
    filename = 'HRS and CCL Cases Jan 2022 - April 2025 from Sue.xlsx'
    analyze_and_save_data(filename)