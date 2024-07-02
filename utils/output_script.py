import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.preprocessing import LabelEncoder
import random
from IPython.display import display
from datetime import datetime
ranking_df = pd.read_csv('../datasets/fifa_ranking-2023-07-20.csv')
game_results_df = pd.read_csv('../datasets/results.csv')
rank_at_year_df = pd.read_csv('../datasets/rank_per_yr_T_sorted.csv')

rank_at_year_df.set_index('Country',inplace=True)
rank_at_year_df.info()

rank_at_year_df.columns = rank_at_year_df.columns.astype(int)
# Filter out games with teams not in the FIFA ranking
print('Before removal of unusable games:', game_results_df.shape)
game_results = game_results_df.loc[(game_results_df['home_team'].isin(values=rank_at_year_df.index.values)) & (
    game_results_df['away_team'].isin(values=rank_at_year_df.index.values))]
print('After removal:', game_results.shape)


game_results.loc[:, 'date'] = pd.to_datetime(game_results['date'])
game_results = game_results.drop(columns=['city'])

# Encoding 'home_team'
label_encoder = LabelEncoder()
game_results.loc[:, 'home_code'] = label_encoder.fit_transform(game_results['home_team'])

# Prepare for encoding 'away_team' and 'country'
all_teams = pd.concat([game_results['home_team'], game_results['away_team'], game_results['country']]).unique()
label_encoder.fit(all_teams)

# Encoding 'away_team' and 'country'
game_results.loc[:, 'away_code'] = label_encoder.transform(game_results['away_team'])
game_results.loc[:, 'hosting_country_code'] = label_encoder.transform(game_results['country'])
# Display the last few rows of the DataFrame
print(game_results.tail())



game_results['date'] = pd.to_datetime(game_results['date'])
ranking_df['rank_date'] = pd.to_datetime(ranking_df['rank_date'])

# Filter the games and rankings starting from 1993
game_results = game_results[game_results['date'] >= datetime(1993, 1, 1)]
ranking_df = ranking_df[ranking_df['rank_date'] >= datetime(1993, 1, 1)]

# Convert 'neutral' column to boolean format (1 for True, 0 for False)
game_results['neutral'] = game_results['neutral'].astype(int)

# Display the 'neutral' column to verify the changes
print(game_results['neutral'].head())
print(game_results.tournament.value_counts())

game_results['final_score'] = game_results.apply(lambda x: 2 if x.away_score < x.home_score else 0 if x.away_score > x.home_score else 1,axis=1)

game_results.reset_index(level=0,inplace=True)
game_results.drop('index',inplace=True,axis=1)
game_results['final_score'].value_counts()

game_results.tournament.value_counts()

def decide_competition(match_type):
   # Superclasico is a south american friendly
   if match_type.lower() == 'friendly' or match_type.lower().find('superclásico') != -1:
      return 2
   tournament_names = ['tournoi','uefa','tournament','qualifications','qualification','games','festival','league']
   cup_names = ['cup', 'final','championship','copa']
   # Tournament - 0
   for x in tournament_names:
      if match_type.lower().find(x) != -1:
         return 0
   # Cup - 1
   for x in cup_names:
      if match_type.lower().find(x) != -1:
         return 1

game_results['tournament_code'] = game_results['tournament'].apply(decide_competition)

game_results['tournament_code'] = game_results.tournament_code.astype(int)
game_results['tournament_code'].value_counts()

# game_results['most_recent_rank_home'] = game_results.apply(lambda x: rank_at_year.loc[x['home_team'],x['date'].year] ,axis=1)
# game_results['most_recent_rank_away'] = game_results.apply(lambda x: rank_at_year.loc[x['away_team'],x['date'].year] ,axis=1)
# game_results['home_advantage'] = 0
# game_results.loc[game_results['home_code'] == game_results['hosting_country_code'], 'home_advantage'] = 1
# game_results['home_advantage'].value_counts()
game_results['most_recent_rank_home'] = game_results.apply(lambda x: rank_at_year_df.loc[x['home_team'],x['date'].year] ,axis=1)
game_results['most_recent_rank_away'] = game_results.apply(lambda x: rank_at_year_df.loc[x['away_team'],x['date'].year] ,axis=1)

# Calculate the ranking difference directly using vectorized operations
game_results['most_recent_rank_difference'] = game_results['most_recent_rank_home'] - game_results['most_recent_rank_away']

# Create a new column 'head_to_head_last_5' with default value 0
game_results['head_to_head_last_5'] = 0

# Define a function to calculate the number of wins for the home team against the away team in the last 5 matches
def calculate_head_to_head(row):
    home_team = row['home_team']
    away_team = row['away_team']
    
    # Extract the relevant subset of matches
    relevant_matches = game_results[
        ((game_results['home_team'] == home_team) & (game_results['away_team'] == away_team)) |
        ((game_results['home_team'] == away_team) & (game_results['away_team'] == home_team))
    ]
    
    # Sort the subset by date
    relevant_matches = relevant_matches.sort_values(by='date')
    
    # Consider only the last 5 matches
    last_5_matches = relevant_matches[relevant_matches['date'] < row['date']].tail(5)
    
    # Count the number of wins for the home team
    wins_home_team = last_5_matches[
        (last_5_matches['home_team'] == home_team) &
        (last_5_matches['final_score'] == 2)
    ].shape[0] 
    
    return wins_home_team

# Apply the function to calculate the head-to-head performance for each row
game_results['head_to_head_last_5'] = game_results.apply(calculate_head_to_head, axis=1)
game_results['head_to_head_last_5'].value_counts()

game_results['most_recent_rank_difference'] = game_results.apply(lambda x: x.most_recent_rank_home - x.most_recent_rank_away,axis=1)

# Create new columns 'goal_difference_last_5_home' and 'goal_difference_last_5_away'
game_results['goal_difference_last_5_home'] = 0
game_results['goal_difference_last_5_away'] = 0

# Define a function to calculate the goal difference for the home team in the last five matches
def calculate_goal_difference_last_5(row, team_type):
    team = row[team_type]
    
    # Extract the relevant subset of matches
    relevant_matches = game_results[game_results[team_type] == team]
    
    # Sort the subset by date
    relevant_matches = relevant_matches.sort_values(by='date')
    
    # Consider only the last five matches
    last_5_matches = relevant_matches[relevant_matches['date'] < row['date']].tail(5)
    
    # Calculate the total goals scored by the team in the last five matches
    goals_scored = last_5_matches['{}_score'.format('home' if team_type == 'home_team' else 'away')].sum()
    # Calculate the total goals scored against the team in the last five matches
    goals_conceded = last_5_matches['{}_score'.format('home' if team_type == 'away_team' else 'away')].sum()
    # if (team == 'Australia') & (row['date'] < pd.to_datetime('2021-06-01')):
    #     print('Australia: {} - {} \ Difference: {}'.format(goals_scored,goals_conceded, goals_scored-goals_conceded))
    # Calculate the goal difference
    goal_difference = goals_scored - goals_conceded
    
    return goal_difference

# Apply the function to calculate the goal difference for the home team in the last five matches for each row
# TODO - USE MAPPED HOME TEAM
game_results['goal_difference_last_5_home'] = game_results.apply(calculate_goal_difference_last_5, team_type='home_team', axis=1)

# Apply the function to calculate the goal difference for the away team in the last five matches for each row
game_results['goal_difference_last_5_away'] = game_results.apply(calculate_goal_difference_last_5, team_type='away_team', axis=1)

game_results['goal_difference_last_5_home'].value_counts() # This seems fishy

# Let's find that team
game_results[game_results['goal_difference_last_5_home'] == 67]

game_results.loc[((game_results['home_team'] == 'Australia') | (game_results['away_team'] == 'Australia')) & (game_results['date'] < '2001-06-02') & (game_results['home_score'] > 10)].tail(5)

game_results.loc[((game_results['home_team'] == 'Australia') | (game_results['away_team'] == 'Australia')) & (game_results['date'] < '2001-06-02') & (game_results['home_score'] > 10)].tail(5)['home_score'].sum()

# Create a new column 'days_since_last_match' to represent the days since the last match for each team
game_results['days_since_last_match_home'] = game_results.groupby('home_team')['date'].diff().dt.days
game_results['days_since_last_match_away'] = game_results.groupby('away_team')['date'].diff().dt.days

# Fill NaN values with a default value (e.g., 0)
game_results['days_since_last_match_home'] = game_results['days_since_last_match_home'].fillna(0)
game_results['days_since_last_match_away'] = game_results['days_since_last_match_away'].fillna(0)

game_results['days_since_last_match_home'].value_counts()

game_results.columns

game_results.dtypes

# Let's look at Saudi Arabia's games
results = game_results.loc[('Saudi Arabia' == game_results['home_team'])]
display(results)

# We'll create a new column 'wins_last_10_games' with NaN values
game_results['wins_last_10_games'] = pd.Series(dtype=float)

# Define a function to calculate the number of wins in the last 10 games
def calculate_wins_last_10(row):
    team = row['home_team']
    date = row['date']
    last_10_games = game_results[(game_results['home_team'] == team) | (game_results['away_team'] == team)]
    last_10_games = last_10_games[last_10_games['date'] < date].tail(10)
    wins = last_10_games['final_score'].eq(2).sum()  # 2 represents a home win
    return wins

# Apply the function to calculate the number of wins in the last 10 games for each row
game_results['wins_last_10_games'] = game_results.apply(calculate_wins_last_10, axis=1)
game_results['wins_last_10_games'].value_counts()

game_results = game_results.drop(['home_team','away_team','home_score','away_score','tournament','country','date'],axis=1)

game_results.info()

corr_mtx = game_results.corr(numeric_only=True)
corr_mtx['final_score'].sort_values(ascending=False)

game_results.head()

# Determine the number of instances for the smallest class
min_class_count = 5499  # The count of the smallest class
# Use the same seed for reproducibility in sampling
random_state = 42
# Create a balanced DataFrame by sampling from each class
balanced_game_results = pd.DataFrame()
# If a class has fewer instances than the smallest class, we'll need to sample with replacement
for class_label in game_results['final_score'].unique():
    class_subset = game_results[game_results['final_score'] == class_label]
    if len(class_subset) > min_class_count:
        # If we have more than the minimum class count, we sample without replacement
        class_subset_sample = class_subset.sample(n=min_class_count, random_state=random_state)
    else:
        # If not, we sample with replacement to boost up to the minimum class count
        class_subset_sample = class_subset.sample(n=min_class_count, random_state=random_state, replace=True)
    balanced_game_results = pd.concat([balanced_game_results, class_subset_sample], axis=0)

# Now, each class should have the same number of instances
print(balanced_game_results['final_score'].value_counts())

# Update the game_results DataFrame to be the balanced one
game_results = balanced_game_results# Add your Boot Strap here and set it equal to game results 

# Define the classes based on final_score
X = game_results.drop(['final_score'], axis=1)
y = game_results['final_score']

from sklearn.model_selection import train_test_split
# Splitting of Data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, stratify=y,  random_state=42)

from sklearn.ensemble import RandomForestClassifier

# Create a RandomForestClassifier model with balanced class weights
clf = RandomForestClassifier(class_weight='balanced', bootstrap=True, random_state=42)
clf.fit(X_train, y_train)
# Makes Prediction on the test set 
y_pred = clf.predict(X_test)


from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

co_m = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(co_m, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()





















