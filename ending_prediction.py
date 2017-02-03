from __future__ import division

import csv
import pandas as pd
import numpy as np
import os

from scipy.stats import randint as sp_randint
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV

from utilities import my_metrics, score_models

def splits(df):
    '''
    INPUT: Dataframe of movie data.
    OUTPUT: Train test split of data, ready to be handed to the models.
    '''
    df = df.dropna(axis=0, how='any')

    X = df[['predicted_cluster', u'window_0', u'window_1', u'window_2',
                       u'gross', u'year',
                       u'Sci-Fi',u'Romance', u'Comedy', u'Horror', u'Thriller', u'Drama', u'Action', u'Family', u'Adventre']]
    y = df['delta_happy']
    y = np.ravel(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=456)

    return X_train, X_test, y_train, y_test

def random_forest(X_train, X_test, y_train, y_test):
    '''
    Fits the random forest determined by grid search to the training data, returns various scores.
    INPUT: The train test split.
    '''
    rf_model = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=None, max_features=4, max_leaf_nodes=None,
            min_samples_leaf=9, min_samples_split=9,
            min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,
            oob_score=False, random_state=456, verbose=0,
            warm_start=False)

    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    print score_models(y_test, y_pred)

def majority_classifier(y_test):
    '''
    Scores a "stupid" model that just predicts the majority class every time (except once to avoid divide by zero error.)
    '''
    y_pred_always_happy = [1]*(len(y_test))
    print score_models(y_test, y_pred_always_happy)

def baseline_classifier(y_test):
    '''
    Scores a "stupid" model that just predicts the majority class with probability equal to the proportion of happy-ending movies.
    '''
    y_pred_base_odds = []

    for i in range(len(y_test)):
        num = np.random.choice(np.arange(0,2), p=[(1-.535), .535])
        y_pred_base_odds.append(num)

    print score_models(y_test, y_pred_base_odds)


if __name__ == '__main__':
    df = pd.read_csv('../data/all_the_movie_data.csv')
    df = df.drop(['title'], 1)
    X_train, X_test, y_train, y_test = splits(df)
    print "Random forest model"
    random_forest(X_train, X_test, y_train, y_test)
    print "Majority classifier"
    majority_classifier(y_test)
    print "Baseline classifier"
    baseline_classifier(y_test)
