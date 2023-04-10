import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from simplicial_complex import SimplicialComplex, compute_boundary_with_matrix
from itertools import chain, combinations

'''''
TDA.py - Coding Homework 3
---------------------------
Examines a set of .csv files for persistent homologies.
Performs some analysis. 

In other words: for each feature vector, compute the difference between feature vectors. Use these differences to compute a min threshold and a max threshold.
For each of these distances: 
Get the points of distance d from one another; construct 0,1,2, and 3-simplices.
'''''
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
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

def check_faces(sc, dimension):
    '''
    A way to try and remove faces from a simplicial complex that don't make any logical sense
    '''
    if (dimension > sc.dimension):
        return "Dimension out of bounds"
    p = sc.get_pchains(dimension)
    k = sc.get_pchains(dimension-1)

    nlist = []
    
    for subchain in k:
        nlist.append(tuple(subchain.mdata))

    for chain in p:
        S = powerset(chain.mdata)
        for s in S:
            print(s)
            if s not in nlist and len(s) == dimension - 1:
                sc.pchains.remove(chain)

    if (sc.get_pchains(dimension) == []):
        sc.dimension -= 2
    # now we need to update the pchain list...
    return 0
    
def get_simplex(data, l, u,  dimension):
    '''
    From data in matrix, construct a simplex of a particular dimension; this is for each distance d. NOT !!!! OPTIMIZED !!!!
    '''
    if (l > u):
        return "ERROR: Lower bound must be less than upper bound"
    # 0-simplices:
    if (dimension == 0):
        simplex = []
        for i in range(0, len(data)):
            for j in range(0, len(data)):
                if (data[i][j] < u) and (data[i][j] > l) and i not in simplex:
                    simplex.append(i)
    # 1-simplices:
    if (dimension == 1):
        simplex = []
        for i in range(0, len(data)):
            for j in range(i+1, len(data)):
                if (data[i][j] < u) and (data[i][j]) > l:
                    if (j,i) not in simplex:
                        simplex.append((i,j))
    # now we need to get creative
    if (dimension == 2):
        temp = []
        for i in range(0, len(data)):
            s = ()
            for j in range(i+1, len(data)):
                if (data[i][j] < u) and (data[i][j] > l):
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

    '''
    TODO: We also need to make sure that the faces of this complex actually make sense.
    '''
    if (dimension == 3):
        temp1 = []
        for i in range(0, len(data)):
            s = ()
            for j in range(i+1, len(data)):
                if (data[i][j] < u) and (data[i][j] > l):
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
                K = get_intersection(temp2[i], temp2[j])
                if (K!= ()):
                    t = get_union(temp2[i], temp2[j])
                    # permute to get the different sets of length 4.
                    res = list(powerset(t))
                    for item in res:
                        if len(item) == 4:
                            simplex.append(item)
        
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

def graph_persistent_homology():
    '''
    Graph the persistent homologies
    '''
    return 

if __name__ == "__main__":
    
    df1 = pd.read_csv("Data/CDHWdata_1.csv")
    print(df1)
    D1 = rank_order(df1)
    distances = get_distances(D1)

    # construct the simplices
    C = [get_simplex(D1, distances[4], distances[20], dim) for dim in range(0,4)]
    print(C)
    
    # build the simplicial complex
    Complex = SimplicialComplex(C)
    print(check_faces(Complex, 3))

    ch = Complex.get_pchains(3)
    for chain in ch:
        print(chain.mdata)

    p = Complex.get_pchains(2)
    for chain in p:
        print(chain.mdata)
    
    H0 = Complex.compute_homologies(1)
    H1 = Complex.compute_homologies(2)
    H2 = Complex.compute_homologies(3)
    
    #print("H0: " + str(H0))
    #print("H1: " + str(H1))
    #print("H2: " + str(H2))

    print(Complex.compute_homology_rank(1))
    print(Complex.compute_homology_rank(2))
   
    print(Complex.compute_euler_characterisic())


    '''
    Ci = [[(1),(2),(3),(4)], [(1,2),(1,3),(2,4),(2,3),(3,4)]]
    A = SimplicialComplex(Ci)
    
    
    H0 = A.compute_homologies(1)
    H1 = A.compute_homologies(2)
    H2 = A.compute_homologies(3)
    
    rankH0 = A.compute_homology_rank(1)
    rankH1 = A.compute_homology_rank(2)
    rankH2 = A.compute_homology_rank(3)

    print("Rank H0: " + str(rankH0))
    print("Rank H1: " + str(rankH1))
    print("Rank H2: " + str(rankH2))
    
    '''
    
