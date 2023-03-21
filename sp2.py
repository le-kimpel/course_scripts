import numpy

class SimplicialComplex:
    '''
    Python representation of a generic simplicial complex.
    '''
    def __init__(self, dimension):
        self.dimension = dimension
        self.pchains = {}
        return

def get_pchains(sc, p):
    '''
    Return the p-chains of dimension p
    '''
    return

def build_complex(data):
    '''
    Build a simplicial complex from the ingested data
    '''
    return
  
if __name__ == "__main__":

    # first, process the data    
    with open("complexA.txt") as f1:
        s1 = f1.read()

    with open("complexB.txt") as f2:
        s2 = f2.read()
