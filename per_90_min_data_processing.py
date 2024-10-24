
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

# for any player that has played less than 90 minutes, upscale their data to 90 minutes
def upscale_stats_to_90_minutes(df):
    if 'Min' in df.columns:
        for col in df.columns[df.columns.get_loc('Min')+1:]:
            df[col] = df[col] / df['Min'] * 90
    return df

# Apply the function to each DataFrame
dfs = [upscale_stats_to_90_minutes(df) for df in dfs]

# remove the minutes column
def remove_minutes_column(df):
    if 'Min' in df.columns:
        df = df.drop(columns=['Min'])
    return df

# Apply the function to each DataFrame
dfs = [remove_minutes_column(df) for df in dfs]

# add a matchday column
def add_matchday_column(df, matchday):
    df['Matchday'] = matchday
    return df

# Apply the function to each DataFrame
dfs = [add_matchday_column(df, i+1) for i, df in enumerate(dfs)]

# Save each DataFrame to a CSV file
for i, df in enumerate(dfs):
    output_file_path = f'data/normalized_per_90_min/normalized_sunderland_md{i+1}_stats.csv'
    df.to_csv(output_file_path, index=False)