import pandas as pd
import numpy as np
import csv
import scipy.stats as scs

def make_dummies(df):
    cluster_dummies = pd.get_dummies(df['predicted_cluster']).rename(columns = lambda x: 'cluster_'+str(x))
    df = pd.concat([df, cluster_dummies], axis=1)
    return df

def dollars_to_floats(x):
    if x:
        if str(x)[0] == '$':
            x = x[1:] #remove dollar sign
            x = ''.join(x.split(','))
            x = float(x)

        else:
            pass

        return x

def t_tests(num_clusters, df):
    for i in range(num_clusters):
        sample1 = df[df['predicted_cluster']==float(i)]['gross']
        print sample1.shape
        sample2 = df[df['predicted_cluster']!=float(i)]['gross']
        print sample2.shape



        print "t test for cluster {}".format(i)
        print scs.ttest_ind(sample1.dropna(), sample2.dropna(), equal_var=False)



if __name__ == '__main__':

    df = pd.read_csv('cluster_data2.csv')
    df['gross'] = df['gross'].apply(dollars_to_floats)
    print type(df['gross'][0])
    print df['gross'][0]
    num_clusters = len(df['predicted_cluster'].unique())
    print t_tests(num_clusters, df)
