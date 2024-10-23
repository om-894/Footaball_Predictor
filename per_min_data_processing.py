
import pandas as pd

# Load all matchday data files
file_paths = [
    'data/data_raw/sunderland_md1_player_stats.csv',
    'data/data_raw/sunderland_md2_player_stats.csv',
    'data/data_raw/sunderland_md3_player_stats.csv',
    'data/data_raw/sunderland_md4_player_stats.csv',
    'data/data_raw/sunderland_md5_player_stats.csv',
    'data/data_raw/sunderland_md6_player_stats.csv',
    'data/data_raw/sunderland_md7_player_stats.csv',
    'data/data_raw/sunderland_md8_player_stats.csv',
    'data/data_raw/sunderland_md9_player_stats.csv'
]

# Read the data from each file into pandas DataFrames
dfs = [pd.read_csv(file) for file in file_paths]

# Define a function to divide columns from 'Min' onwards by the 'Min' column
def divide_stats_by_minutes(df):
    if 'Min' in df.columns:
        for col in df.columns[df.columns.get_loc('Min')+1:]:
            df[col] = df[col] / df['Min']
    return df

# Apply the function to each DataFrame
dfs = [divide_stats_by_minutes(df) for df in dfs]

# Save each DataFrame to a CSV file
for i, df in enumerate(dfs):
    output_file_path = f'data/normalized_per_min/normalized_sunderland_md{i+1}_player_stats.csv'
    df.to_csv(output_file_path, index=False)

print("DataFrames have been saved to CSV files.")


