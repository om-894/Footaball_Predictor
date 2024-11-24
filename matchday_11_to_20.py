
import pandas as pd
import requests
import os
import re
import urllib3

# Create the "data" folder if it doesn't exist
data_folder = "data/data_raw"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

####################################################
#         Match 11: Sunderland vs Luton town       #
####################################################

md11_url = "https://fbref.com/en/matches/b2475a0d/Luton-Town-Sunderland-October-23-2024-Championship"
response = requests.get(md11_url, verify=False)
md11_df = pd.read_html(response.text, attrs={"id":"stats_8ef52968_summary"})[0]

# Join multiindex columns with an underscore and remove extra spaces
md11_df.columns = ['_'.join(col).strip() for col in md11_df.columns.values]

# Clean column names using regex
md11_df.columns = [re.sub(r'Unnamed:.*?_level_0_', '', col).strip() for col in md11_df.columns]

# remove bottom row so that we only have individual player data.
md11_df = md11_df.iloc[:-1]

# Assuming your dataframe is named 'df'
md11_df['Saves'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]
md11_df['Fouls_Committed'] = [1, 0, 3, 0, 2, 0, 0, 0, 3, 2, 0, 0, 1, 0]
md11_df['Fouls_Won'] = [0, 1, 1, 0, 1, 0, 3, 0, 3, 2, 1, 0, 1, 1]

print(md11_df) # to display the dataframe with the new column names

# Export to CSV
md11_df.to_csv(os.path.join(data_folder, "sunderland_md11_player_stats.csv"), index=False)


####################################################
#      Match 12: Sunderland vs Oxford united       #
####################################################

md12_url = "https://fbref.com/en/matches/da73045d/Sunderland-Oxford-United-October-26-2024-Championship"
response = requests.get(md12_url, verify=False)
md12_df = pd.read_html(response.text, attrs={"id":"stats_8ef52968_summary"})[0]

# Join multiindex columns with an underscore and remove extra spaces
md12_df.columns = ['_'.join(col).strip() for col in md12_df.columns.values]

# Clean column names using regex
md12_df.columns = [re.sub(r'Unnamed:.*?_level_0_', '', col).strip() for col in md12_df.columns]

# remove bottom row so that we only have individual player data.
md12_df = md12_df.iloc[:-1]

# Assuming your dataframe is named 'df'
md12_df['Saves'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
md12_df['Fouls_Committed'] = [0, 0, 1, 0, 1, 2, 0, 0, 1, 2, 0, 2, 0, 0, 0]
md12_df['Fouls_Won'] = [0, 1, 1, 0, 2, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0]

print(md12_df) # to display the dataframe with the new column names

# Export to CSV
md12_df.to_csv(os.path.join(data_folder, "sunderland_md12_player_stats.csv"), index=False)


####################################################
#         Match 13: Sunderland vs QPR              #
####################################################

md13_url = "https://fbref.com/en/matches/ed5ce326/Queens-Park-Rangers-Sunderland-November-2-2024-Championship"
response = requests.get(md13_url, verify=False)
md13_df = pd.read_html(response.text, attrs={"id":"stats_8ef52968_summary"})[0]

# Join multiindex columns with an underscore and remove extra spaces
md13_df.columns = ['_'.join(col).strip() for col in md13_df.columns.values]

# Clean column names using regex
md13_df.columns = [re.sub(r'Unnamed:.*?_level_0_', '', col).strip() for col in md13_df.columns]

# remove bottom row so that we only have individual player data.
md13_df = md13_df.iloc[:-1]

# Assuming your dataframe is named 'df'
md13_df['Saves'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]
md13_df['Fouls_Committed'] = [0, 0, 3, 1, 2, 0, 2, 0, 2, 0, 0, 0, 0]
md13_df['Fouls_Won'] = [0, 0, 0, 2, 1, 1, 0, 3, 2, 1, 2, 0, 0]

print(md13_df) # to display the dataframe with the new column names

# Export to CSV
md13_df.to_csv(os.path.join(data_folder, "sunderland_md13_player_stats.csv"), index=False)


####################################################
#         Match 14: Sunderland vs Preston          #
####################################################

md14_url = "https://fbref.com/en/matches/d71db0bd/Preston-North-End-Sunderland-November-6-2024-Championship"
response = requests.get(md14_url, verify=False)
md14_df = pd.read_html(response.text, attrs={"id":"stats_8ef52968_summary"})[0]

# Join multiindex columns with an underscore and remove extra spaces
md14_df.columns = ['_'.join(col).strip() for col in md14_df.columns.values]

# Clean column names using regex
md14_df.columns = [re.sub(r'Unnamed:.*?_level_0_', '', col).strip() for col in md14_df.columns]

# remove bottom row so that we only have individual player data.
md14_df = md14_df.iloc[:-1]

# Assuming your dataframe is named 'df'
md14_df['Saves'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
md14_df['Fouls_Committed'] = [0, 0, 2, 1, 0, 0, 0, 0, 1, 3, 1, 3, 0, 0]
md14_df['Fouls_Won'] = [2, 1, 2, 0, 2, 3, 3, 3, 1, 2, 0, 0, 1, 0]

print(md14_df) # to display the dataframe with the new column names

# Export to CSV
md14_df.to_csv(os.path.join(data_folder, "sunderland_md14_player_stats.csv"), index=False)


####################################################
#         Match 15: Sunderland vs Coventry         #
####################################################


