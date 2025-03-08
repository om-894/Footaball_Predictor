
import pandas as pd
import os

# Load all matchday data files from the data_raw directory
file_paths = [
    'data/data_raw/sunderland_md1_player_stats.csv',
    'data/data_raw/sunderland_md2_player_stats.csv',
    'data/data_raw/sunderland_md3_player_stats.csv',
    'data/data_raw/sunderland_md4_player_stats.csv',
    'data/data_raw/sunderland_md5_player_stats.csv',
    'data/data_raw/sunderland_md6_player_stats.csv',
    'data/data_raw/sunderland_md7_player_stats.csv',
    'data/data_raw/sunderland_md8_player_stats.csv',
    'data/data_raw/sunderland_md9_player_stats.csv',
    'data/data_raw/sunderland_md10_player_stats.csv',
    'data/data_raw/sunderland_md11_player_stats.csv',
    'data/data_raw/sunderland_md12_player_stats.csv',
    'data/data_raw/sunderland_md13_player_stats.csv',
    'data/data_raw/sunderland_md14_player_stats.csv',
    'data/data_raw/sunderland_md15_player_stats.csv',
    'data/data_raw/sunderland_md16_player_stats.csv',
    'data/data_raw/sunderland_md17_player_stats.csv'
]

# Read the data from each file into pandas DataFrames
dfs = [pd.read_csv(file) for file in file_paths]

# Function to upscale stats to 90 minutes for players who played more than 20 minutes and less than or equal to 90 minutes
def upscale_stats_to_90_minutes(df):
    if 'Min' in df.columns:       
        # Create a boolean mask to filter players who played more than 20 minutes but at most 90 minutes
        mask = (df['Min'] > 20) & (df['Min'] <= 90)
        # 'Min' [+1:] allows for all columns after min.
        for col in df.columns[df.columns.get_loc('Min')+1:]:    
            # Scale the values in the specified column to a per 90 minutes basis for the rows where the mask is True
            df.loc[mask, col] = df.loc[mask, col] / df.loc[mask, 'Min'] * 90
    return df

# Apply the function to each DataFrame
dfs = [upscale_stats_to_90_minutes(df) for df in dfs]

# Function to remove the 'Min' column
def remove_minutes_column(df):
    if 'Min' in df.columns:
        df = df.drop(columns=['Min'])
    return df

# Apply the function to each DataFrame
dfs = [remove_minutes_column(df) for df in dfs]

# Function to add a 'Matchday' column
def add_matchday_column(df, matchday):
    df['Matchday'] = matchday
    return df

# Apply the function to each DataFrame
dfs = [add_matchday_column(df, i+1) for i, df in enumerate(dfs)]

# Ensure the output directory exists
output_dir = 'data/normalized_per_90_min'
os.makedirs(output_dir, exist_ok=True)

# Save each DataFrame to a CSV file
for i, df in enumerate(dfs):
    output_file_path = f'{output_dir}/normalized_sunderland_md{i+1}_stats.csv'
    df.to_csv(output_file_path, index=False)

print("Data normalization and file saving completed.")
