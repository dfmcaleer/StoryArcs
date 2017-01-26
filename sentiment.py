import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import os
from collections import OrderedDict
from sklearn.cluster import KMeans
from sklearn import preprocessing

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

    windows = []
    for i in xrange(0,len(text_list)-size,size):
        window = text_list[i:i+size]
        window = " ".join(window)
        windows.append(window)

    return windows

def quick_n_dirty_polarity(text):
    blob = TextBlob(text)
    sentiments = []
    for sentence in blob.sentences:
        sentiments.append(sentence.sentiment.polarity)
    return sentiments

def dirty_windows(window_list):
    window_sents = []
    for window in window_list:
        sent_list = quick_n_dirty_polarity(window)
        window_sents.append(np.mean(sent_list))
    return window_sents

#Get list of paths to all scripts
scripts = os.listdir('scripts')
paths = []
for script in scripts:
    new_path = 'scripts/{}'.format(script)
    paths.append(new_path)

print "Made it to line 60"

def make_array(paths, number_movies, window_divisor):
    '''
    Inputs: paths = list of filepaths to movie scripts
    number_movies = how many you want to do this time (keep low for testing so it doesn't take forever!)
    window_divisor = We will divide the length of the movie by this to get a proportionally sized window for each movie.  25 seems to work well.
    '''

    movie_list = paths[0:number_movies]

    movie_names = []
    list_movie_meanwindowsent = []
    for movie in movie_list:
        clean_movie = text_clean(movie)
        movie_windows = windows(clean_movie, int(len(clean_movie)/window_divisor + 1))
        window_polarity = dirty_windows(movie_windows)
        if len(window_polarity) == window_divisor-1:
            list_movie_meanwindowsent.append(window_polarity)
            movie_names.append(movie)

    X = np.array(list_movie_meanwindowsent)
    movie_names = np.array(movie_names)

    return X, movie_names

print "made it to line 85"
X, movie_names = make_array(paths, len(paths), 6)
#To make all movies, set number_movies = len(paths)
# print X.shape


for i,value in enumerate(movie_names):
    movie_names[i] = value.replace(value, value[8:-4])

print movie_names

'''
TO DO:
Build elbow plots to justify choice of 3 clusters.
Show other cluster numbers to show that 3 is robust.
**You can evaluate the clusters by showing how much variation they are explaining.
'''

def deltas(X):
    '''
    Input: Array of sentiment levels (each row is a movie, each column is a window).
    Output: Array that is one column less that is the difference between each window.
    '''
    diff_array = []
    for row in X:
        diff_array.append(np.diff(row))

    diff_array = np.array(diff_array)
    return diff_array

diff_array = deltas(X)
#
# #Cluster!!!!!!!!!!!!!!!
model = KMeans(n_clusters=3)
predictions = model.fit_predict(diff_array)
#
# #Add the predicted clusters as the last column of the data frame.
X_and_pred = np.column_stack((diff_array, predictions))
print X_and_pred.shape
'''
REMEMBER: X_and_pred IS AN ARRAY WITH DIMENSIONS
(N_MOVIES-1, N_WINDOWS+1)
BECAUSE THE LAST COLUMN IS THE PREDICTED CLUSTER.

IF YOU WANT TO SEE HOW THE CLUSTERING IS DOING...
USE THE LAST COLUMN!

To do:
1. Turn into a Pandas dataframe.
2. Add the name of the movie as the index.
3. Get additional data about the movie to use as features.
'''

cols = []
for i,column in enumerate(X_and_pred.T):
    col_name = 'window_{}'.format(i)
    cols.append(col_name)

cols[-1] = cols[-1].replace(cols[-1], 'predicted_cluster')

df = pd.DataFrame(X_and_pred, columns=cols)
df = df.set_index(movie_names)

print "cluster 0"
print df[df['predicted_cluster']==0.0]

print "cluster 1"
print df[df['predicted_cluster']==1.0]

print "cluster 2"
print df[df['predicted_cluster']==2.0]


'''
#THIS PART IS IMPORTANT, BUT I'M NOT USING IT RIGHT NOW.

def make_splits(X, index):
    #Split dataframe based on what topic
    return X[X[:,-1] == float(index), :]

#def predictionary()
#Make this a function later, but remember the name "predictionary"

#Create dictionary with cluster name as key and movie sentiment deltas as values.
output = {}

for prediction in np.unique(predictions):
    title = "group_{}".format(prediction)
    output[title] = make_splits(X_and_pred, prediction)[:,:-1]

#Take mean for each cluster at each window
means_array = []
for group in output.keys():
    mean_group = np.mean(output[group], axis=0)
    means_array.append(mean_group)

#Turn it back into an array
means_array = np.array(means_array)

#Not sure when I transposed this, but better transpose it back:
means_array = means_array.T

#Scale or else everything will be bad
means_array = preprocessing.scale(means_array)

# #SHOW THE MOVIE MAGIC!
# plt.plot(means_array)
# plt.show()

'''



'''
Don't look at this part.

model = KMeans(n_clusters=6)
predictions = model.fit_predict(X)
print type(predictions)
print predictions.shape
print X.shape

X_and_pred = np.column_stack((X, predictions))
print X_and_pred.shape
print X_and_pred[:,24:]

def make_splits(X, index):
    return X[X[:, 25] == float(index), :]

output = {}

for prediction in np.unique(predictions):
    title = "group_{}".format(prediction)
    output[title] = make_splits(X_and_pred, prediction)[:,:-1]

means_array = []
for group in output.keys():
    mean_group = np.mean(output[group], axis=0)
    means_array.append(mean_group)

means_array = np.array(means_array)
means_array = means_array.T

print means_array.shape

plt.plot(means_array)
plt.show()
'''


'''
Ignore this too.

# clean_novel = text_clean('novel.txt')
# novel_windows = windows(clean_novel, int(len(clean_novel)/25.0 + 1))
# window_polarity = dirty_windows(novel_windows)
#
# novel_dict = {}
# for i, window in enumerate(window_polarity):
#     novel_dict[window] = novel_windows[i][0:500]
# #key is polarity score, value is chunk of novel
# print "#" * 1000
# od = OrderedDict(sorted(novel_dict.items()))
# for k, v in od.iteritems():
#     print k, v
#
# plt.plot(window_polarity)
# plt.show()
'''
