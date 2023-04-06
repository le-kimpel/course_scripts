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

def get_simplex(data, d, dimension):
    '''
    From data in matrix, construct a simplex of a particular dimension; this is for each distance d. 
    '''
    # 0-simplices:
    if (dimension == 0):
        simplex = []
        for i in range(0, len(data)):
            for j in range(0,len(data)):
                if (data[i][j] == d):
                    simplex.append(i)
                    simplex.append(j)
    
    # 1-simplices:
    if (dimension == 1):
        simplex = []
        for i in range(0,len(data)):
            for j in range(0, len(data)):
                if (data[i][j] == d):
                    simplex.append((i,j))
    return list(set(simplex))

def get_distances(data):
    '''
    Get the 1-dimensional vector of all possible distances from the dist. matrix.
    '''
    vec = []
    for i in range(0,len(data)):
        for j in range(0, len(data)):
            vec.append(data[i][j])
    vec = sorted(list(set(vec)))
    return vec

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
    D1 = rank_order(df1)
    distances = get_distances(D1)
    A = get_simplex(D1, distances[1], 1)
    print(A)
