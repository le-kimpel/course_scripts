import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

'''''
TDA.py - Coding Homework 3
---------------------------
Examines a set of .csv files for persistent homologies.
Performs some analysis. 

In other words: for each feature vector, compute the difference between the 15 points. Use these differences to compute a min threshold and a max threshold.
For each of these distances: 
Get the points of distance d from one another; construct 0,1, and 2-simplices.
'''''

def rank_order(df):
    '''
    Rank order the metric distances from point to point. (vector to vector.)
    '''
    # extract the feature vectors
    df.pop('partno')

    # calculate the euclidean distance between them
    distances = pdist(df.values, metric='euclidean')
    dist_matrix = squareform(distances)
    return dist_matrix

if __name__ == "__main__":
    df1 = pd.read_csv("Data/CDHWdata_1.csv")
    print(df1)
    rank_order(df1)
