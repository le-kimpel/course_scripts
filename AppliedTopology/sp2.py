import numpy as np

class pchain:
    '''
    Python representation of a pchain object
    '''
    def __init__(self, data):
        self.mdata = data
        self.boundary = None
        self.dimension = data.size
        return
    def compute_boundary(self):
        # recall the equation for computing the boundary of a pchain!
        total = np.array(0)
        if (self.dimension > 1):
            for i in range(0,self.mdata.size):
                arr = np.delete(self.mdata, i)
                err = ((-1)**self.dimension) * arr % max(self.mdata)
                total = total + err
        return total
class SimplicialComplex:
    '''
    Python representation of a generic simplicial complex.
    '''
    def __init__(self, deltas):
        self.dimension = len(deltas) + 1
        self.mdata = deltas
        self.pchains = init_pchains(deltas)
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
    def compute_boundary_matrix(self, Cp):
        return 
    
def get_pchains(sc, p):
    '''
    Return the p-chains of dimension p
    '''
    res = []
    for pchain in sc.pchains:
        if(pchain.dimension == p):
            res.append(pchain)
    return res
  
if __name__ == "__main__":

    # an integer representation
    cow = 1
    rabbit = 2
    horse = 3
    dog = 4
    fish = 5
    dolphin = 6
    oyster = 7
    broccoli = 8
    fern = 9
    onion = 10
    apple = 11
    
    
    # try not to neglect the vertices here either
    delta = [(horse), (cow), (rabbit), (dog), (fish), (oyster), (dolphin), (broccoli), (fern), (onion), (apple)]
    
    # manual representation grabbed from the painstaking labor in coding hw #1
    delta1 =  [ (cow,rabbit),
      (cow, horse),
      (cow, dog),
      (rabbit, horse),
      (rabbit, dog),
      (horse, dog),
      (fish, dolphin),
      (fish, oyster),
      (dolphin, oyster),
      (broccoli, fern),
      (fern, onion),
      (fern,apple),
    (onion, apple),
    (broccoli, fern),
    (broccoli, apple),
    (broccoli, onion)]

    delta2 = [(cow,rabbit, horse), (cow, rabbit, dog), (cow, horse, dog), (rabbit, horse, dog), (fish, dolphin, oyster), (broccoli, fern, onion), (broccoli, fern, apple), (broccoli, onion, apple), (fern, onion, apple)]
    # this is essentially Cp 
    deltas = [delta, delta1, delta2]

    for data in deltas:
        for stuff in data:
            stuff = np.array(stuff)
            p = pchain(stuff)
            print(p.compute_boundary())
            

