from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from bs4 import BeautifulSoup
from requests import get
from scipy import stats
from utils import *

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import csv, warnings

## how many NaNs in a df --> print(data.isnull().sum().sum())
## OR
## null_cols = data.columns[data.isnull().any()]
## data[null_cols].isnull().sum()
##features = data.apply(lambda row: (row.PassYds_Off + row.RushYdsOff) - (row.PassYds_Def + row.RushYds_Def), axis=1)#.to_frame()
## model.fit() requires features to be DataFrame, assumes more than 1D
## z = np.abs(stats.zscore(df))
## print(np.where(z > 5))

def try_stat(game, statname):
    stat = game.find('td', {'data-stat': statname}).get_text()
    return int(stat) if stat else 0

def get_stats(game):
    # could find_all td's and just use counted index,
    # but harder to read and not necessarily consistent 
    # for every season and every team
    try:
        pts = int(game.find('td', {'data-stat': 'pts_off'}).get_text())
    except ValueError:
        return [np.nan] * 10

    location = game.find('td', {'data-stat': 'game_location'}).get_text()
    home = 0 if location else 1

    # offensive stats
    first_down_off = try_stat(game, 'first_down_off')
    pass_yds_off = try_stat(game, 'pass_yds_off')
    rush_yds_off = try_stat(game, 'rush_yds_off')
    turnovers = try_stat(game, 'to_off')

    # defensive stats
    first_down_def = try_stat(game, 'first_down_def')
    pass_yds_def = try_stat(game, 'pass_yds_def')
    rush_yds_def = try_stat(game, 'rush_yds_def')
    takeaways = try_stat(game, 'to_def')

    return [home, pts, first_down_off, pass_yds_off, rush_yds_off, turnovers, first_down_def, pass_yds_def, rush_yds_def, takeaways]

def build_dataframe(filename, url):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        teams = {rows[0].lower(): (rows[1].strip(), rows[2].strip()) for rows in reader if not rows[0].startswith('#')}
    stats = []
    for name, info in teams.items():
        for i in range(int(info[1]), 2019):
            new_url = url.format(info[0], str(i))
            print("team: {}\tyear: {}".format(name, i))
            response = get(new_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            try:
                season = soup.find('table', {'id': 'games'}).find('tbody').find_all('tr')
            except AttributeError:
                ## The Cleveland Browns were suspended b/t '95 and '99
                ## while NFL decided if they could move to Baltimore.
                ## Baltimore was added as an expansion team, CLE 
                ## remained
                continue
            for game in season:
                stats.append(get_stats(game))

    return pd.DataFrame.from_records(stats, columns=["Home", "Points", "1stD_Off", "PassYds_Off", "RushYdsOff", "Turnovers", "1stD_Def", "PassYds_Def", "RushYds_Def", "Takeaways"])

def crawl():
    url = "https://www.pro-football-reference.com/teams/{}/{}.htm"
    csvfile = 'teams.csv'
    data = build_dataframe(csvfile, url).dropna()
    save_df(data, './nfl_stats.pkl')


if __name__ == '__main__':

    df = load_df('nfl_stats.pkl')
    xmetric=""
    ymetric="Points"

    features = df.drop(ymetric, axis=1)
    target = df[ymetric]


    #X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.30)
    #clf = linear_model.LinearRegression()
    clf = linear_model.Lasso(alpha=0.4)
    scores = cross_val_score(clf, features, target, cv=10)
    print("Accuracy: {:.2f}% (+/- {:.2f})".format(scores.mean(), scores.std()*2))







    #model = clf.fit(X_train, y_train)
    #predictions = model.predict(X_test)
    #print("Score: {}".format(model.score(X_test, y_test)))


    #colors = (0,0,0)
    #area = np.pi*3
    #plt.scatter(turnovers, points, s=area, c=colors, alpha=0.5)
    #plt.xlabel("Turnovers")
    #plt.ylabel("Points Scored")
    #plt.title("Points vs. Turnovers")


