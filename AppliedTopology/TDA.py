import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from simplicial_complex import SimplicialComplex

'''''
TDA.py - Coding Homework 3
---------------------------
Examines a set of .csv files for persistent homologies.
Performs some analysis. 

In other words: for each feature vector, compute the difference between feature vectors. Use these differences to compute a min threshold and a max threshold.
For each of these distances: 
Get the points of distance d from one another; construct 0,1,2, and 3-simplices.
'''''
def get_union(A,B):
    '''
    Gets the union of tuples
    '''
    return tuple(set(A).union(set(B)))
def get_intersection(A, B):
    '''
    Gets the intersection of tuples
    '''
    return tuple(set(A) & set(B))
def get_simplex(data, d, dimension):
    '''
    From data in matrix, construct a simplex of a particular dimension; this is for each distance d. NOT !!!! OPTIMIZED !!!!
    '''
    # 0-simplices:
    if (dimension == 0):
        simplex = []
        for i in range(0, len(data)):
            for j in range(0, len(data)):
                if (data[i][j] < d) and i not in simplex:
                    simplex.append(i)
    # 1-simplices:
    if (dimension == 1):
        simplex = []
        for i in range(0, len(data)):
            for j in range(i+1, len(data)):
                if (data[i][j] < d):
                    if (j,i) not in simplex:
                        simplex.append((i,j))
    # now we need to get creative
    if (dimension == 2):
        temp = []
        for i in range(0, len(data)):
            s = ()
            for j in range(i+1, len(data)):
                if (data[i][j] < d):
                    if (j,i) not in temp:
                        s = s + (i,j)
                        temp.append(s)
        simplex = []
        # get the union of tuples
        for i in range(0, len(temp)):
            for j in range(i+1, len(temp)):
                K = get_intersection(temp[i], temp[j])
                if (K!= ()):
                    t = get_union(temp[i], temp[j])
                    simplex.append(t)

    if (dimension == 3):
        temp1 = []
        for i in range(0, len(data)):
            s = ()
            for j in range(i+1, len(data)):
                if (data[i][j] < d):
                    if (j,i) not in temp1:
                        s = s + (i,j)
                        temp1.append(s)
        temp2 = []
        # get the union of tuples
        for i in range(0, len(temp1)):
            for j in range(i+1, len(temp1)):
                K = get_intersection(temp1[i], temp1[j])
                if (K!= ()):
                    t = get_union(temp1[i], temp1[j])
                    temp2.append(t)
        simplex = []
        for i in range(0, len(temp2)):
            for j in range(i+1, len(temp2)):
                K = get_intersection(temp1[i], temp1[j])
                if (K!= ()):
                    t = get_union(temp1[i], temp1[j])
                    simplex.append(t)
        
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

    # construct the simplices
    C = [get_simplex(D1, distances[10], dim) for dim in range(0,4)]
    print(C)
    # build the simplicial complex
    Complex = SimplicialComplex(C)
    
    
    
