import numpy

class pchain:
    '''
    Python representation of a pchain object
    '''
    def __init__(self, data)
    self.mdata = data
    self.boundary = None
    self.dimension = 0
    return

class SimplicialComplex:
    '''
    Python representation of a generic simplicial complex.
    '''
    def __init__(self, deltas):
        self.dimension = len(deltas) + 1
        self.deltas = deltas
        self.pchains = {}
        return
    def init_pchains(self, deltas):
        '''
        Iteratively build  out pchain objects for each of the p-dimensional chains in deltas
        '''
        for chain in deltas:
            for data in chain:
                # get the dimension of the pchain
                dim = len(data)
                p = pchain(data)
                p.dimension = dim
                self.pchains.append(p)     
        return

def get_pchains(sc, p):
    '''
    Return the p-chains of dimension p
    '''
    return

def build_complex(data, deltas):
    '''
    Build a simplicial complex from the ingested data
    '''
    return
  
if __name__ == "__main__":

    # try not to neglect the vertices here either
    delta = [("horse"), ("cow"), ("rabbit"), ("dog"), ("fish"), ("oyster"), ("dolphin"), ("broccoli"), ("fern"), ("onion"), ("apple")]
    
    # manual representation grabbed from the painstaking labor in coding hw #1
    delta1 =  [ ("cow","rabbit"),
      ("cow", "horse"),
      ("cow", "dog"),
      ("rabbit", "horse"),
      ("rabbit", "dog"),
      ("horse", "dog"),
      ("fish", "dolphin"),
      ("fish", "oyster"),
      ("dolphin", "oyster"),
      ("broccoli", "fern"),
      ("fern", "onion"),
      ("fern","apple"),
    ("onion", "apple"),
    ("broccoli", "fern"),
    ("broccoli", "apple"),
    ("broccoli", "onion")]

     delta2 = [("cow","rabbit", "horse"), ("cow", "rabbit", "dog"), ("cow", "horse", "dog"),
                   ("rabbit", "horse", "dog"), ("fish", "dolphin", "oyster"), ("broccoli", "fern", "onion"), ("broccoli", "fern", "apple"), ("broccoli", "onion", "apple"), ("fern", "onion", "apple")]
     # this is essentially Cp 
     deltas = {delta, delta1, delta2}
