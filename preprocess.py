import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.preprocessing import LabelEncoder
import random
from IPython.display import display
from datetime import datetime
ranking = pd.read_csv('./datasets/fifa_ranking-2023-07-20.csv')
game_results = pd.read_csv('./datasets/results.csv')
rank_at_year = pd.read_csv('./datasets/rank_per_yr_T_sorted.csv')

rank_at_year.set_index('Country',inplace=True)

rank_at_year.columns = rank_at_year.columns.astype(int)
# Remove games that have teams not present in the FIFA ranking
print('Before removal of invalid games:',game_results.shape)
# game_results = game_results.loc[(game_results['home_team'] in rank_at_year.index.values) & (game_results['away_team'] in rank_at_year.index.values)]
game_results = game_results.loc[(game_results['home_team'].isin(values=rank_at_year.index.values)) & (game_results['away_team'].isin(values=rank_at_year.index.values))]
print('After removal:',game_results.shape)
# Converting dates to datetime objects
game_results['date'] = pd.to_datetime(game_results['date'])
ranking['rank_date'] = pd.to_datetime(ranking['rank_date'])
game_results.drop(columns=['city'],inplace=True)


game_results['home_code'] = game_results.home_team.astype('category').cat.codes
l = len(game_results['home_team'].values)
country_codes = {game_results['home_team'].values[i]:game_results['home_code'].values[i] for i in range(l)}
# labelEnconder.fit(game_results.stack().unique())
code = max(country_codes.values()) + 1
# encode_dict = game_results.['home_team']
def encode_away_teams(x):
   global code
   val = country_codes.setdefault(x, code)
   if val == code:
      code+=1
   return val
game_results['away_code'] = game_results.away_team.apply(lambda x: encode_away_teams(x))
game_results['hosting_country_code'] = game_results.country.apply(lambda x: encode_away_teams(x))
# Since we only have fifa rankings starting from 1992-12-31, we'll be only keeping games from 1993-Present
game_results = game_results[game_results['date'] >= datetime(year=1993,month=1,day=1)]
ranking = ranking[ranking['rank_date'] >= datetime(year=1993,month=1,day=1)]
# Convert neutral column to bool
game_results['neutral'] = game_results['neutral'].astype(bool)

def decide_competition(match_type):
   tournament = [True if match_type.lower().find(x) != -1 else False for x in ['Tournoi','UEFA','tournament','qualifications','qualification','games','festival','League']]
   if True in tournament:
      return 'Tournament'
   cup = [True if match_type.lower().find(x) != -1 else False for x in ['cup', 'final','championship','copa']]
   if True in cup:
      return 'Cup'
   else:
      return 'Friendly'

# Reorganize the tournaments to either be cup, tournament, or friendly
game_results['tournament_code'] = game_results['tournament'].apply(decide_competition)

# 0 -> Cup, 1 -> Tournament, 2-> Friendly
game_results['tournament_code'] = game_results.tournament_code.astype('category').cat.codes

# Get the most recent FIFA rank at the time of match
game_results['most_recent_rank_home'] = game_results.apply(lambda x: rank_at_year.loc[x['home_team'],x['date'].year] ,axis=1)
game_results['most_recent_rank_away'] = game_results.apply(lambda x: rank_at_year.loc[x['away_team'],x['date'].year] ,axis=1)

# Get the difference in rank between the two teams
game_results['most_recent_rank_difference'] = game_results.apply(lambda x: x.most_recent_rank_home - x.most_recent_rank_away,axis=1)

# <1 -> Home loss,
# =1 -> Tie,
# >1 -> Home win
game_results['final_score'] = game_results.apply(lambda x: x.home_score if x.away_score == 0 else x.home_score / x.away_score,axis=1)
game_results.reset_index(level=0,inplace=True)
game_results.drop('index',inplace=True,axis=1)
print(game_results.head(-5))

print(game_results.dtypes)

print('\nShowing Saudi Arabia home games...')
results = game_results.loc[('Saudi Arabia'  == game_results['home_team'])]
print(results)

