from __future__ import division

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from scipy.stats import hmean

'''
This script contains a few helper functions referenced in the other scripts.  There is nothing interesting here.
'''

def reg_title(title):
    '''
    Reformats title from weird way it appears from web-scrape.
    '''
    if title.split()[0] == 'The':
        rest = '-'.join(title.split()[1:])
        title = ''.join([rest, ',-The'])
    else:
        title = title.replace(" ", "-")
    return title

def get_rating(text):
    '''
    Turns rating variable into a scale of G=0, PG=1, PG-13=2, R=3, NC-17=4
    '''
    if type(text) == str:
        if "Rated G" in text:
            rating_number = 0
        elif "Rated PG" in text:
            rating_number = 1
        elif "Rated PG-13" in text:
            rating_number = 2
        elif "Rated R" in text:
            rating_number = 3
        elif "Rated NC-17" in text:
            rating_number = 4
        else:
            pass
        return rating_number

def happy_or_sad(x):
    '''
    This function is to apply to the vector of ending sentiments to classify as happy ending or not.
    Input: Any float
    Output: 1 if number is positive, 0 if negative.
    (I have this simple thing as its own function in case later want to change how we classify happy/sad endings, or include more categories than just happy or sad.)
    '''
    if x >= 0:
        return 1
    elif x < 0:
        return 0

def make_dummies(df):
    '''
    Input: dataframe
    Output: Same dataframe, but with dummy variables indicating which cluster each row belongs to.
    '''
    cluster_dummies = pd.get_dummies(df['predicted_cluster']).rename(columns = lambda x: 'cluster_'+str(x))
    df = pd.concat([df, cluster_dummies], axis=1)
    return df

def dollars_to_floats(x):
    '''
    Input: A dollar amount string (e.g. $4,000,000)
    Output: As a float (e.g. 4000000.0)
    '''
    if x:
        if str(x)[0] == '$':
            x = x[1:] #remove dollar sign
            x = ''.join(x.split(','))
            x = float(x)

        else:
            pass

        return x

def genre_dummies(df):
    '''
    Input: Dataframe.
    Output: Same dataframe, but with dummy variables for each genre.  Movies can belong to more than one genre D:
    '''
    big_genre_list = []
    #First, get the unique genres.
    for entry in df['genre']: #Each "entry" is a list of genres.
        if entry:
            entry = ''.join(c for c in str(entry) if c not in "[]''u,").split(' ')
            for g in entry: #Iterate through that list.
                big_genre_list.append(g)

    genres = set(big_genre_list) #Gets each unique genre.

    #Create variable in df for each genre.
    for genre1 in genres:
        df[genre1] = 0

    #Now add flag for each movie in each genre.
    for genre1 in genres:
        for i, entry in enumerate(df['genre']):
            entry = ''.join(c for c in str(entry) if c not in "[]''u,").split(' ')
            if genre1 in entry:
                df[genre1][i] = 1
            else:
                pass
    return df

def my_metrics(y_test, y_pred):

    cm=confusion_matrix(y_test, y_pred)

    #Be careful!  Sklearn confusion matrix is very confusing!
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]

    if FN == 0:
        FN = 1
    if TN == 0:
        TN = 1

    #Proportion of those identified as negative that actually are.
    unprecision = TN/(TN+FN)
    #print unprecision

    #Proportion of those *actually* negative identified as such.
    unrecall = TN/(FP+TN)
    #print unrecall

    #Get harmonic mean
    unf = hmean([unrecall, unprecision])
    #print unf

    #Get mean of this and the f1 score:
    harmonic_af = np.mean([f1_score(y_test, y_pred), unf])

    return harmonic_af

def score_models(y_test, y_pred):
    '''
    Prints out a range of classification metrics for a vector of true and predicted y.
    '''
    print "accuracy"
    print accuracy_score(y_test, y_pred)
    print "f1"
    print f1_score(y_test, y_pred)
    print "precision"
    print precision_score(y_test, y_pred)
    print "recall"
    print recall_score(y_test, y_pred)
    print "mean of f1 and anti-f1"
    print my_metrics(y_test, y_pred)
