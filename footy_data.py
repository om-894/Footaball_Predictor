
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

# MD1: Cardiff City vs Sunderland
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