import numpy as np

class pchain:
    '''
    Python representation of a pchain object
    '''
    def __init__(self, data):
        self.mdata = data
        self.dimension = 0
        self.boundary, self.boundary_pretty_print  = self.compute_boundary()
        return
    def compute_boundary(self):
        # recall the equation for computing the boundary of a pchain!
        total = np.array(0)
        eqn = ''
        if (self.dimension > 1):
            for i in range(0,self.mdata.size):
                arr = np.delete(self.mdata, i)
                err = ((-1)**i) * arr
                if (i is not self.mdata.size-1):
                    eqn += "(-1)^(" + str(i) + ")" + str(arr) + " + "
                else:
                    eqn += "(-1)^(" + str(i) + ")" + str(arr)
                total = total + err
            return total, "Boundary of " + str(self.mdata) + ": " + eqn
        else:
            return total, "Boundary of " + str(self.mdata) + ": 0"
class SimplicialComplex:
    '''
    Python representation of a generic simplicial complex.
    '''
    def __init__(self, Cp):
        self.dimension = len(Cp) + 1
        self.Cp = Cp
        self.pchains = self.init_pchains()
        return
    def init_pchains(self):
        '''
        Iteratively build  out pchain objects for each of the p-dimensional chains in deltas
        '''
        plist = []
        for chain in self.Cp:
            for data in chain:
                # get the dimension of the pchain
                if not isinstance(data, int):
                    dim = len(data)
                else:
                    dim = 1
                p = pchain(np.array(data))
                p.dimension = dim
                print(p.dimension)
                plist.append(p) 
        return plist 
    def compute_boundary_matrix(self, dimension):
        '''
        Returns the boundary matrix representation of the chains that span Cp
        and Cp-1
        '''
        Cp = self.get_pchains(dimension)
        C_ = self.get_pchains(dimension - 1)
            
        # build an m x n numpy matrix
        D = np.zeros((len(C_), len(Cp)))

        # set an index equal to 1 if Cp-1 belongs to the boundary of Cp, 0 if not
        for i in range(0, len(C_)):
            for j in range(0, len(Cp)):
                b,p = Cp[j].compute_boundary()
                index = p.find(str(C_[i].mdata))
                if index != -1:
                    D[i][j] = 1
                else:
                    D[i][j] = 0
        return D
    
    def get_pchains(self, p):
        '''
        Return the p-chains of dimension p
        '''
        res = []
        for pchain in self.pchains:
            if(pchain.dimension == p):
                res.append(pchain)
        return res


def compute_boundary_with_matrix(sc, dimension):
    '''
    Pass in a simplicial complex, and compute the boundary matrix
    associated with a particular dimension. 
    Then use that matrix to produce the boundaries of each relevant chain.
    '''
    
    return
    
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
    C0 = [(horse), (cow), (rabbit), (dog), (fish), (oyster), (dolphin), (broccoli), (fern), (onion), (apple)]
    
    # manual representation grabbed from the painstaking labor in coding hw #1
    C1 =  [ (cow,rabbit),
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

    C2 = [(cow,rabbit, horse), (cow, rabbit, dog), (cow, horse, dog), (rabbit, horse, dog), (fish, dolphin, oyster), (broccoli, fern, onion), (broccoli, fern, apple), (broccoli, onion, apple), (fern, onion, apple)]
    
    # this is essentially Cp 
    Cp = [C0, C1, C2]

    A = SimplicialComplex(Cp)
    A.compute_boundary_matrix(2)

    
    # set up the pchains
    for data in Cp:
        for chain in data:
            chain = np.array(chain)
            p = pchain(chain)
            #print(p.boundary_pretty_print)
            

