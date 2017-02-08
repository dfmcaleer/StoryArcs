'''
I haven't even started making these plots look decent.
'''

import matplotlib.pyplot as plt
import os
import pygal
import pandas as pd
from unidecode import unidecode
from pygal.style import DefaultStyle, DarkStyle, BlueStyle
from sklearn import preprocessing

def cols_to_keep(df):

    df = df.set_index('title')

    cols_to_keep = []

    for column in df.columns:
        if column[:6]=='window':
            cols_to_keep.append(column)

    cols_to_keep.append('predicted_cluster')

    df = df[cols_to_keep]

    return df

def cluster_plot(df, filename):
    grouped_by_cluster = df.groupby('predicted_cluster').mean()
    plt.plot(grouped_by_cluster)
    plt.savefig('../plots/{}.pdf'.format(filename))
    plt.show()

def pygal_clusterplot(df, filename):
    grouped_by_cluster = df.groupby('predicted_cluster').mean()
    chart = pygal.Line(interpolate='cubic', style=DarkStyle, show_dots=False, show_y_guides=False, stroke_style={'width': 5})
    for i in range(4):
        #print grouped_by_cluster.ix[i, :]
        chart.add('Cluster {}'.format(i), grouped_by_cluster.ix[i, :])
    chart.render_to_file(filename)

def plot_all_the_movies(df, filename):
    chart = pygal.Line(interpolate='cubic', style=DarkStyle, show_legend=False, show_dots=False, show_y_guides=False, stroke_style={'width': 1})
    for i in range(df.shape[0]):
        #print grouped_by_cluster.ix[i, :]
        chart.add('', df.ix[i, :])
    chart.render_to_file(filename)


# def plot_all_the_movies(df, filename):
#     df = df.ix[:,:-1].T
#     df.plot(legend=False)
#     plt.savefig('../plots/{}.pdf'.format(filename))
#     plt.show()


'''
*Make another figure that has all the movies with the lines colored by which cluster they are in and semi-transparent, then the cluster lines on top in the same color, but thicker and fully opaque.

*Also make a figure of a single movie shows what is going on at all the ups and downs.
'''


if __name__ == '__main__':
    # df = pd.read_csv('../data/movie_endings.csv')
    # df = cols_to_keep(df)
    # df = df.groupby('predicted_cluster').mean()
    # chart = pygal.Line(interpolate='cubic', style=DarkStyle, show_legend=False, show_dots=False, show_y_guides=False, stroke_style={'width': 5})
    # chart.add('', df.ix[0, :])
    # chart.render_to_file('test-0.svg')
    # df = pd.read_csv('../data/princess_bride.csv')
    # plot_all_the_movies(df, 'princess_bride.svg')


    # pygal_clusterplot(df, 'test-single.svg')
    #plot_all_the_movies(df, 'all_movies.svg')

    line_chart = pygal.Bar(style = BlueStyle, show_y_guides=False, width=400)
    line_chart.x_labels = ['Majority', 'Baseline', 'Features', 'Sentiment']
    line_chart.add('Accuracy', [{'value': .48, 'color': 'white'}, {'value': .52, 'color': 'white'}, {'value': .57, 'color': 'white'}, {'value': .70, 'color': 'white'}])
    line_chart.add('F1',  [{'value': .65, 'color': 'black'}, {'value': .52, 'color': 'black'}, {'value': .62, 'color': 'black'}, {'value': .71, 'color': 'black'}])
    line_chart.render_to_file('classification_metric_barchart.svg')
