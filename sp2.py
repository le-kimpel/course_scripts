import numpy

class SimplicialComplex:
    '''
    Python representation of a generic simplicial complex.
    '''
    def __init__(self, dimension, deltas):
        self.dimension = dimension
        self.pchains = get_pchains(deltas)
        return

class pchain:
    '''
    Python representation of a pchain object
    '''
    def __init__(self, deltas)
    self.mdata = deltas
    self.boundary = None
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
     deltas = {delta1, delta2}
