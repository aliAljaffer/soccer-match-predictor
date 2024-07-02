from datetime import datetime

import numpy as np
import pandas as pd

ranking = pd.read_csv('./datasets/fifa_ranking-2023-07-20.csv')
game_results = pd.read_csv('./datasets/results.csv')

game_results['date'] = pd.to_datetime(game_results['date'])
ranking['rank_date'] = pd.to_datetime(ranking['rank_date'])

game_results.drop(columns=['city'], inplace=True)
# Since we only have fifa rankings starting from 1992-12-31, we'll be only keeping games from 1993-Present
game_results = game_results[game_results['date'] >= datetime(year=1993, month=1, day=1)]
ranking = ranking[ranking['rank_date'] >= datetime(year=1993, month=1, day=1)]

df = pd.DataFrame()
df['rank_date'] = ranking['rank_date'].dt.year
df.drop_duplicates(subset='rank_date', inplace=True)
df.index = df['rank_date']
df.drop(['rank_date'], axis=1, inplace=True)
countries = list(set(ranking['country_full'].values))
df = pd.concat([df, pd.DataFrame(columns=countries)], axis=1)
df[list(countries)].fillna(inplace=True, value=0)


def find_rank_at_year(country, date: int):
    year = datetime(year=date, day=1, month=1)
    most_recent = ranking.loc[(ranking['country_full'] == country) & (ranking['rank_date'].dt.year == year.year)]
    if most_recent.size > 0:
        rank = most_recent['rank'].values[0]
        return rank
    return np.nan


def find_fifa_points_at_year(country, date: int):
    year = datetime(year=date, day=1, month=1)
    most_recent = ranking.loc[(ranking['country_full'] == country) & (ranking['rank_date'].dt.year == year.year)]
    if most_recent.size > 0:
        pts = most_recent['total_points'].values[0]
        return pts
    return 0 # Not returning NaN to avoid filling with mean later


for year in df.index:
    df.loc[year] = df.columns.map(lambda country: find_rank_at_year(country, year))
    df.loc['{}_points'.format(year)] = 0
    df.loc['{}_points'.format(year)] = df.columns.map(lambda country: find_fifa_points_at_year(country,year))
    # print(df.columns.values)

print(df.tail())

def fix_countries(country1: str, country2: str):
    df[country1].fillna(df[country2], inplace=True)
    df[country2].fillna(df[country1], inplace=True)


fix_countries('Türkiye', 'Turkey')
fix_countries('Gambia', 'The Gambia')
fix_countries('Samoa', 'American Samoa')
fix_countries('St. Vincent / Grenadines', 'St Vincent and the Grenadines')
fix_countries('St. Vincent and the Grenadines', 'St Vincent and the Grenadines')
fix_countries('St. Vincent / Grenadines', 'St. Vincent and the Grenadines')
fix_countries('Curacao', 'Curaçao')
fix_countries('Aotearoa New Zealand','New Zealand')


df.fillna(df.mean().astype(int), inplace=True)

print(df.head())

df.sort_index(axis=1, ascending=True, inplace=True)
df.transpose().to_csv('./datasets/rank_per_yr_T_sorted.csv')
