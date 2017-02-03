#This thing needs to be tested.

'''
This script will read in all scripts, create their story arcs based on polary, cluster by story shape, and then output a CSV columns for movie title, sentiment change for each window, and shape cluster.
'''

from __future__ import division

import csv
import pandas as pd
import numpy as np
import os

from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn import preprocessing
from textblob import TextBlob

from utilities import reg_title, get_rating, happy_or_sad, make_dummies, dollars_to_floats, genre_dummies

def get_script_filepaths(filepath):
    '''
    Input: Path into folder full of scripts.
    Output: List of filepaths to each script.
    '''
    scripts = os.listdir(filepath)
    paths = []
    for script in scripts:
        new_path = '../data/scripts/{}'.format(script)
        paths.append(new_path)
    paths = paths[1:]
    return paths

def text_clean(filename):
    '''
    Input: File path of script.
    Output: List of all words in script lowercased, lemmatized, without punctuation.
    '''
    wnl = WordNetLemmatizer()
    word_list = [word.decode("utf8", errors='ignore') for line in open(filename, 'r') for word in line.split()]
    lemma_list = [wnl.lemmatize(word.lower()) for word in word_list]
    return lemma_list

def windows(text_list, size=100):
    '''
    Input:
    text_list = List of cleaned words
    size = size of window

    Output:
    Text as list of windows of size=size.
    '''
    window_list = []

    for i in xrange(0,len(text_list)-size,size):
        window = text_list[i:i+size]
        window = " ".join(window)
        window_list.append(window)

    return window_list

def get_polarity(text):
    '''
    Input: Text.
    Output: List of sentiment scores for each sentence in the text.
    '''
    blob = TextBlob(text)
    sentiments = []
    for sentence in blob.sentences:
        sentiments.append(sentence.sentiment.polarity)
    return sentiments

def polarity_windows(window_list):
    '''
    Input: List in which each item is the text of a window of the movie.
    Output: Mean polarity for each window in the list.
    '''
    window_sents = []
    for window in window_list:
        sent_list = get_polarity(window)
        window_sents.append(np.mean(sent_list))
    return window_sents

def make_array(paths, number_movies, window_divisor):
    '''
    Inputs: paths = list of filepaths to movie scripts
    number_movies = how many of the movies to use (for quicker runtime, choose smaller number)
    window_divisor = We will divide the length of the movie by this to get a proportionally sized window for each movie.
    Output: X is an array with dimensions (number_movies, window_divisor) that gives the sentiment for each window in the movie.
    '''
    movie_list = paths[0:number_movies]

    movie_names = []
    list_movie_meanwindowsent = []
    for movie in movie_list:
        clean_movie = text_clean(movie)
        if len(clean_movie) > 200:
            movie_windows = windows(clean_movie, int(len(clean_movie)/window_divisor + 1))
            window_polarity = polarity_windows(movie_windows)
            if len(window_polarity) == window_divisor-1:
                list_movie_meanwindowsent.append(window_polarity)
                movie_names.append(movie)

    #Make it just the name of the movie instead of the whole filepath
    for i,value in enumerate(movie_names):
        movie_names[i] = value.replace(value, value[16:-4])

    X = np.array(list_movie_meanwindowsent)
    movie_names = np.array(movie_names)

    return X, movie_names

def deltas(X):
    '''
    Input: Array of sentiment levels (each row is a movie, each column is a window).
    Output: Array of differences between each window, normalized (same # rows, 1 fewer columns).
    '''
    diff_array = []
    for row in X:
        diff_array.append(np.diff(row))

    diff_array = np.array(diff_array)
    diff_array = preprocessing.scale(diff_array)
    return diff_array

def shape_clusters(n_clusters, random_state, diff_array, omit_ending=False):
    '''
    Input: Number of clusters, random state of model, array of deltas, and
    boolean on whether to omit the last sentiment window when fitting the model
    for future prediction of ending.
    Output: vector with number of each cluster, array with both sentiment changes and which cluster each movie belongs to, and (if ending is omitted) a separate column vector of each movie's ending sentiment change.
    '''

    if omit_ending == False:
        model = KMeans(n_clusters=n_clusters, n_jobs=-2, random_state=random_state)
        cluster_numbers = model.fit_predict(diff_array)

        #Add the clusters as the last column of the data frame.
        X_and_clus = np.column_stack((diff_array, cluster_numbers))

        return cluster_numbers, X_and_clus

    elif omit_ending==True:

        #Save ending column as ending
        ending = diff_array[:,-1:]

        #Remove last column for ending
        diff_array = diff_array[:,:-1]

        model = KMeans(n_clusters=n_clusters, n_jobs=-2, random_state=random_state)
        cluster_numbers = model.fit_predict(diff_array)

        #Add the predicted clusters as the last column of the data frame.
        X_and_clus = np.column_stack((diff_array, cluster_numbers))

        return cluster_numbers, X_and_clus, ending

def add_features(X_and_clus, movie_names, ending=None):
    '''
    Input: Array of movie sentiments and clusters, array of movie names.
    Output: A dataframe of these things in addition to other web-scraped movie metadata.
    '''
    cols = []
    for i,column in enumerate(X_and_clus.T):
        col_name = 'window_{}'.format(i)
        cols.append(col_name)

    cols[-1] = cols[-1].replace(cols[-1], 'predicted_cluster')

    df = pd.DataFrame(X_and_clus, columns=cols)
    df['title'] = movie_names

    gross = pd.read_csv('../data/movie_gross.csv')
    genre = pd.read_csv('../data/movie_genre.csv')
    year = pd.read_csv('../data/movie_year.csv')

    other = pd.read_csv('../data/movie_other.csv')
    other['rating'] = other['rating'].apply(float)
    other['mpaa'] = other['mpaa'].apply(get_rating)

    #Finally, add the ending if you used the clusters that omit it.
    if ending != None:
        delta_happy = []

        for movie in ending:
            delta_happy.append(happy_or_sad(movie))
        df['delta_happy'] = delta_happy

    df = df.merge(gross,on='title').merge(genre,on='title').merge(year,on='title').merge(other,on='title')

    df['gross'] = df['gross'].apply(dollars_to_floats)
    df = genre_dummies(df)
    df = df.drop(['genre'], 1)
    df = df.dropna(axis=0, how='any')

    return df

if __name__ == '__main__':

    paths = get_script_filepaths('../data/scripts')

    #Set parameters for creating story arcs:
    window_divisor = 6
    n_clusters = 4
    rs = 456
    subset = len(paths) #for all movies
    #subset = 20

    #Create array of sentiment scores for each window of each movie
    sent_array, movie_names = make_array(paths, subset, window_divisor=window_divisor)

    #Create array of differences in sentiment between each window.
    diff_array = deltas(sent_array)

    #Cluster movies based on shape, omitting the ending:
    cluster_numbers, X_and_clus, ending = shape_clusters(n_clusters=n_clusters, random_state=rs, diff_array=diff_array, omit_ending=True)

    #Cluster movies based on shape, WITH the ending:
    #This is just for interest/plotting; it does NOT belong in the ending prediction model (due to obvious endogeneity).
    ENDcluster_numbers, ENDX_and_clus = shape_clusters(n_clusters=n_clusters, random_state=rs, diff_array=diff_array, omit_ending=False)

    #Put what we've created so far into a dataframe and save it as a CSV.
    df = add_features(X_and_clus, movie_names, ending)
    df.to_csv('../data/all_the_movie_data.csv',index=False)
