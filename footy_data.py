
# Scrape data from websites or use an API that offers football data. 
# The purpose of this script is to put together a solid football dataset.

import pandas as pd
import requests
import os
import re

# Create the "data" folder if it doesn't exist
data_folder = "data"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)


# We want to bypass the SSL certificate verification so we set it to False
main_url = "https://fbref.com/en/squads/8ef52968/Sunderland-Stats"

# Get the HTML content of the URL
response = requests.get(main_url, verify=False)  # Disable SSL verification
full_df = pd.read_html(response.text, attrs={"id": "stats_standard_10"})[0]
print(full_df.head())


# Save the DataFrame as a CSV file to the "data" folder
csv_file_path = os.path.join(data_folder, "sunderland_player_season_stats_ovr.csv")
full_df.to_csv(csv_file_path, index=False)


# Need to get weekly match data, then add in two columns for fouls committed and fouls drawn
# Need to get team overall stats, including fouls committed and fouls drawn

#############################################
#     Match 1: Cardiff City vs Sunderland   #
#############################################

md1_url = "https://fbref.com/en/matches/9e3914bb/Cardiff-City-Sunderland-August-10-2024-Championship"
response = requests.get(md1_url, verify=False)
md1_df = pd.read_html(response.text, attrs={"id":"stats_8ef52968_summary"})[0]

# '_'.join(col).strip(): Joins the two levels of the MultiIndex with an underscore and removes any extra spaces.
md1_df.columns = ['_'.join(col).strip() for col in md1_df.columns.values]

# Clean column names using regex
md1_df.columns = [re.sub(r'Unnamed:.*?_level_0_', '', col).strip() for col in md1_df.columns]

# remove bottom ro so that we only have individual player data.
md1_df = md1_df.iloc[:-1]

# Assuming your dataframe is named 'df'
md1_df['Saves'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4]
md1_df['Fouls_Committed'] = [1, 1, 0, 0, 0, 0, 1, 2, 0, 0, 1, 1, 1, 0]  # Initialize with 0
md1_df['Fouls_Won'] = [0, 0, 2, 0, 0, 0, 2, 2, 1, 1, 0, 1, 1, 0]  # Initialize with 0

print(md1_df) # to display the dataframe with the new column names

# Looks good. Export to CSV
md1_df.to_csv(os.path.join(data_folder, "sunderland_md1_player_stats.csv"), index=False)


####################################################
#     Match 2: Sunderland vs Sheffield Wednesday   #
####################################################

md2_url = "https://fbref.com/en/matches/ab1574d3/Sunderland-Sheffield-Wednesday-August-18-2024-Championship"
response = requests.get(md2_url, verify=False)
md2_df = pd.read_html(response.text, attrs={"id":"stats_8ef52968_summary"})[0]

# '_'.join(col).strip(): Joins the two levels of the MultiIndex with an underscore and removes any extra spaces.
md2_df.columns = ['_'.join(col).strip() for col in md2_df.columns.values]

# Clean column names using regex
md2_df.columns = [re.sub(r'Unnamed:.*?_level_0_', '', col).strip() for col in md2_df.columns]

# remove bottom ro so that we only have individual player data.
md2_df = md2_df.iloc[:-1]

# Assuming your dataframe is named 'df'
md2_df['Saves'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]
md2_df['Fouls_Committed'] = [2, 1, 2, 0, 1, 0, 0, 1, 2, 0, 0, 0, 0, 0]  
md2_df['Fouls_Won'] = [0, 1, 5, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]

print(md2_df) # to display the dataframe with the new column names

# Looks good. Export to CSV
md2_df.to_csv(os.path.join(data_folder, "sunderland_md2_player_stats.csv"), index=False)


####################################################
#         Match 3: Sunderland vs Burnley           #
####################################################

md3_url = "https://fbref.com/en/matches/2375d0ce/Sunderland-Burnley-August-24-2024-Championship"
response = requests.get(md3_url, verify=False)
md3_df = pd.read_html(response.text, attrs={"id":"stats_8ef52968_summary"})[0]

# '_'.join(col).strip(): Joins the two levels of the MultiIndex with an underscore and removes any extra spaces.
md3_df.columns = ['_'.join(col).strip() for col in md3_df.columns.values]

# Clean column names using regex
md3_df.columns = [re.sub(r'Unnamed:.*?_level_0_', '', col).strip() for col in md3_df.columns]

# remove bottom ro so that we only have individual player data.
md3_df = md3_df.iloc[:-1]

# Assuming your dataframe is named 'df'
md3_df['Saves'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
md3_df['Fouls_Committed'] = [0, 2, 5, 0, 0, 0, 1, 0, 2, 1, 1, 0, 0, 1, 0]
md3_df['Fouls_Won'] = [0, 0, 4, 0, 5, 0, 0, 1, 2, 2, 3, 0, 1, 1, 0]

print(md3_df) # to display the dataframe with the new column names

# Looks good. Export to CSV
md3_df.to_csv(os.path.join(data_folder, "sunderland_md3_player_stats.csv"), index=False)


####################################################
#         Match 4: Sunderland vs Portsmouth        #
####################################################

md4_url = "https://fbref.com/en/matches/26ebbffe/Portsmouth-Sunderland-August-31-2024-Championship"
response = requests.get(md4_url, verify=False)
md4_df = pd.read_html(response.text, attrs={"id":"stats_8ef52968_summary"})[0]

# Join multiindex columns with an underscore and remove extra spaces
md4_df.columns = ['_'.join(col).strip() for col in md4_df.columns.values]

# Clean column names using regex
md4_df.columns = [re.sub(r'Unnamed:.*?_level_0_', '', col).strip() for col in md4_df.columns]

# remove bottom row so that we only have individual player data.
md4_df = md4_df.iloc[:-1]

# Assuming your dataframe is named 'df'
md4_df['Saves'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
md4_df['Fouls_Committed'] = [1, 1, 4, 1, 1, 1, 0, 1, 1, 2, 0, 1, 0, 0]
md4_df['Fouls_Won'] = [0, 0, 2, 0, 1, 2, 0, 0, 2, 2, 2, 2, 1, 0]

print(md4_df) # to display the dataframe with the new column names

# Looks good. Export to CSV
md4_df.to_csv(os.path.join(data_folder, "sunderland_md4_player_stats.csv"), index=False)


####################################################
#         Match 5: Sunderland vs Plymouth          #
####################################################

md5_url = "https://fbref.com/en/matches/b872dddc/Plymouth-Argyle-Sunderland-September-14-2024-Championship"
response = requests.get(md5_url, verify=False)
md5_df = pd.read_html(response.text, attrs={"id":"stats_8ef52968_summary"})[0]

# Join multiindex columns with an underscore and remove extra spaces
md5_df.columns = ['_'.join(col).strip() for col in md5_df.columns.values]

# Clean column names using regex
md5_df.columns = [re.sub(r'Unnamed:.*?_level_0_', '', col).strip() for col in md5_df.columns]

# remove bottom row so that we only have individual player data.
md5_df = md5_df.iloc[:-1]

# Assuming your dataframe is named 'df'
md5_df['Saves'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4]
md5_df['Fouls_Committed'] = [1, 0, 2, 2, 1, 0, 1, 0, 1, 0, 2, 2, 0]
md5_df['Fouls_Won'] = [1, 0, 5, 0, 0, 2, 1, 1, 0, 0, 1, 1, 0]

print(md5_df) # to display the dataframe with the new column names

# Looks good. Export to CSV
md5_df.to_csv(os.path.join(data_folder, "sunderland_md5_player_stats.csv"), index=False)