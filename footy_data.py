
# Scrape data from websites or use an API that offers football data. 
# The purpose of this script is to put together a solid football dataset.

import pandas as pd
import requests
import os
import re
import urllib3

# Create the "data" folder if it doesn't exist
data_folder = "data/data_raw"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

##################################################
#         Overall player and squad stats         #
##################################################

# We want to bypass the SSL certificate verification so we set it to False
main_url = "https://fbref.com/en/squads/8ef52968/Sunderland-Stats"

# Get the HTML content of the URL
response = requests.get(main_url, verify=False)  # Disable SSL verification
full_df = pd.read_html(response.text, attrs={"id": "stats_standard_10"})[0]

# remove the multindex
full_df.columns = ['_'.join(col) for col in full_df.columns]

# Clean column names using regex
full_df.columns = [re.sub(r'Unnamed:.*?_level_0_', '', col).strip() for col in full_df.columns]

# reomove players that are on loan
full_df = full_df.iloc[:-9, :]

# remove last column and replace with 2 new columns
full_df = full_df.iloc[:, :-1]
full_df['Fouls_Committed'] = [10, 10, 7, 4, 0, 9, 8, 17, 5, 3, 2, 1, 1, 2, 3, pd.NA, 0, 0, 0, 0, 0, 0, 0, 0, 0]
full_df['Fouls_Won '] = [13, 16, 5, 10, 0, 8, 9, 9, 22, 3, 0, 5, 0, 7, 3, pd.NA, 0, 0, 0, 0, 0, 0, 0, 0, 0]
full_df['Saves'] = [0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

print(full_df)

# Save the DataFrame as a CSV file to the "data" folder
csv_file_path = os.path.join(data_folder, "sunderland_player_season_stats_ovr.csv")
full_df.to_csv(csv_file_path, index=False)


# Match result data
matchday_results = pd.read_html(response.text, attrs={"id":"matchlogs_for"})[0]

# drop row 1 becuase not a league game
matchday_results = matchday_results.drop(1) # drops the row with index 1
matchday_results = matchday_results.reset_index(drop=True) # resets index

matchday_results = matchday_results.iloc[:, 3:-3] # drop the first 3 columns and last 3 columns
print(matchday_results.head())

# Save the DataFrame as a CSV file to the "data" folder
csv_file_path = os.path.join(data_folder, "sunderland_match_results.csv")
matchday_results.to_csv(csv_file_path, index=False)


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


####################################################
#       Match 6: Sunderland vs Middlesbrough       #
####################################################

md6_url = "https://fbref.com/en/matches/084412ff/Sunderland-Middlesbrough-September-21-2024-Championship"
response = requests.get(md6_url, verify=False)
md6_df = pd.read_html(response.text, attrs={"id":"stats_8ef52968_summary"})[0]

# Join multiindex columns with an underscore and remove extra spaces
md6_df.columns = ['_'.join(col).strip() for col in md6_df.columns.values]

# Clean column names using regex
md6_df.columns = [re.sub(r'Unnamed:.*?_level_0_', '', col).strip() for col in md6_df.columns]

# remove bottom row so that we only have individual player data.
md6_df = md6_df.iloc[:-1]

# Assuming your dataframe is named 'df'
md6_df['Saves'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
md6_df['Fouls_Committed'] = [0, 2, 0, 0, 2, 0, 1, 0, 2, 0, 0, 1, 0]
md6_df['Fouls_Won'] = [2, 3, 0, 0, 1, 0, 3, 0, 1, 3, 0, 1, 0]

print(md6_df) # to display the dataframe with the new column names

# Looks good. Export to CSV
md6_df.to_csv(os.path.join(data_folder, "sunderland_md6_player_stats.csv"), index=False)


####################################################
#       Match 7: Sunderland vs Watford             #
####################################################

md7_url = "https://fbref.com/en/matches/4b6683f4/Watford-Sunderland-September-28-2024-Championship"
response = requests.get(md7_url, verify=False)
md7_df = pd.read_html(response.text, attrs={"id":"stats_8ef52968_summary"})[0]

# Join multiindex columns with an underscore and remove extra spaces
md7_df.columns = ['_'.join(col).strip() for col in md7_df.columns.values]

# Clean column names using regex
md7_df.columns = [re.sub(r'Unnamed:.*?_level_0_', '', col).strip() for col in md7_df.columns]

# remove bottom row so that we only have individual player data.
md7_df = md7_df.iloc[:-1]

# Assuming your dataframe is named 'df'
md7_df['Saves'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]
md7_df['Fouls_Committed'] = [0, 1, 1, 0, 1, 0, 1, 3, 1, 1, 1, 1, 0, 0]
md7_df['Fouls_Won'] = [1, 0, 4, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]

print(md7_df) # to display the dataframe with the new column names

# Export to CSV
md7_df.to_csv(os.path.join(data_folder, "sunderland_md7_player_stats.csv"), index=False)


####################################################
#         Match 8: Sunderland vs Derby             #
####################################################

md8_url = "https://fbref.com/en/matches/3a71cd8d/Sunderland-Derby-County-October-1-2024-Championship"
response = requests.get(md8_url, verify=False)
md8_df = pd.read_html(response.text, attrs={"id":"stats_8ef52968_summary"})[0]

# Join multiindex columns with an underscore and remove extra spaces
md8_df.columns = ['_'.join(col).strip() for col in md8_df.columns.values]

# Clean column names using regex
md8_df.columns = [re.sub(r'Unnamed:.*?_level_0_', '', col).strip() for col in md8_df.columns]

# remove bottom row so that we only have individual player data.
md8_df = md8_df.iloc[:-1]

# Assuming your dataframe is named 'df'
md8_df['Saves'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5]
md8_df['Fouls_Committed'] = [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 2, 1, 1, 0, 0]
md8_df['Fouls_Won'] = [2, 1, 2, 0, 0, 2, 1, 1, 1, 3, 4, 1, 0, 0, 0]

print(md8_df) # to display the dataframe with the new column names

# Export to CSV
md8_df.to_csv(os.path.join(data_folder, "sunderland_md8_player_stats.csv"), index=False)


####################################################
#         Match 9: Sunderland vs Leeds             #
####################################################

md9_url = "https://fbref.com/en/matches/d893be3b/Sunderland-Leeds-United-October-4-2024-Championship"
response = requests.get(md9_url, verify=False)
md9_df = pd.read_html(response.text, attrs={"id":"stats_8ef52968_summary"})[0]

# Join multiindex columns with an underscore and remove extra spaces
md9_df.columns = ['_'.join(col).strip() for col in md9_df.columns.values]

# Clean column names using regex
md9_df.columns = [re.sub(r'Unnamed:.*?_level_0_', '', col).strip() for col in md9_df.columns]

# remove bottom row so that we only have individual player data.
md9_df = md9_df.iloc[:-1]

# Assuming your dataframe is named 'df'
md9_df['Saves'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
md9_df['Fouls_Committed'] = [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 0]
md9_df['Fouls_Won'] = [2, 2, 2, 0, 1, 1, 2, 0, 4, 2, 0, 0, 0]

print(md9_df) # to display the dataframe with the new column names

# Export to CSV
md9_df.to_csv(os.path.join(data_folder, "sunderland_md9_player_stats.csv"), index=False)


####################################################
#         Match 10: Sunderland vs Hull             #
####################################################

md10_url = "https://fbref.com/en/matches/1663864f/Hull-City-Sunderland-October-20-2024-Championship"
response = requests.get(md10_url, verify=False)
md10_df = pd.read_html(response.text, attrs={"id":"stats_8ef52968_summary"})[0]

# Join multiindex columns with an underscore and remove extra spaces
md10_df.columns = ['_'.join(col).strip() for col in md10_df.columns.values]

# Clean column names using regex
md10_df.columns = [re.sub(r'Unnamed:.*?_level_0_', '', col).strip() for col in md10_df.columns]

# remove bottom row so that we only have individual player data.
md10_df = md10_df.iloc[:-1]

# Assuming your dataframe is named 'df'
md10_df['Saves'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
md10_df['Fouls_Committed'] = [1, 0, 2, 2, 0, 1, 0, 1, 0, 0, 0, 1, 0]
md10_df['Fouls_Won'] = [1, 3, 0, 0, 0, 1, 0, 1, 1, 1, 1, 3, 0]

print(md10_df) # to display the dataframe with the new column names

# Export to CSV
md10_df.to_csv(os.path.join(data_folder, "sunderland_md10_player_stats.csv"), index=False)


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







