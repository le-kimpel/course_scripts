import numpy as np
from sympy import Matrix, linsolve, symbols

class pchain:
    '''
    Python representation of a pchain object
    '''
    def __init__(self, data):
        self.mdata = data
        self.dimension = 0
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
                    if (i%2 == 0):
                        eqn += "(p)" + str(arr) + " + "
                    else:
                        eqn += "(n)" + str(arr) + " + "
                else:
                    if (i%2 == 0):
                        eqn += "(p)" + str(arr)
                    else:
                        eqn += "(n)" + str(arr)
                total = total + err
            return total, "Boundary of " + str(self.mdata) + ": " + eqn
        else:
            return total, "Boundary of " + str(self.mdata) + ": 0"
class SimplicialComplex:
    '''
    Python representation of a generic simplicial complex.
    '''
    def __init__(self, Cp):
        self.dimension = len(Cp)+1
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
                res = p.split(":")[1:]
                index = res[0].find(str(C_[i].mdata))
                if index != -1:
                    # this is really, really sketchy, but it'll be what we do for now.
                    num = str(res[0][index-3])
                    if num == "p":
                        D[i][j] = 1
                    elif num == "n":
                        D[i][j] = -1
                    elif num == "(":
                        if (str(res[0][index-2]) == "p"):
                            D[i][j] = 1
                        elif (str(res[0][index-2]) == "n"):
                            D[i][j] = -1
                else:
                    D[i][j] = 0
        return D

    def compute_cycles(self, dimension):
        '''
        Row-reduce to find the kernel of the boundary map as the kernel of a linear map.
        '''

        # C0 = spanZ0!
        if (dimension == 1):
            return self.get_pchains(dimension)
        
        # first, compute the boundary matrix
        M = self.compute_boundary_matrix(dimension)
        
        num_rows, num_cols = M.shape
        symb = symbols('a0:' + str(num_cols))
        M = Matrix(M)
        
        # now get the row dimensions of M
        C_ = self.get_pchains(dimension - 1)

        # nullspace we'll be solving for
        null = Matrix(np.zeros((len(C_), 1)))
        system = (M,null)
        kernel = linsolve(system, symb)

        # format this output so we can produce the kernel generators
        return kernel
        
    def get_pchains(self, p):
        '''
        Return the p-chains of dimension p
        '''
        res = []
        for pchain in self.pchains:
            if(pchain.dimension == p):
                res.append(pchain)
        return res

    def compute_homologies(self, dimension):
        '''
        Returns the entire homologies of the simplicial complex.
        All we need to do here is compute *ALL* cycles within the complex
        and remove those in a boundary. 
        '''
        Bp = []
        Zp = []
        res = '<'
        res2 = '<'

        if (dimension > 1 and dimension < self.dimension):
            boundary_pchains = self.get_pchains(dimension+1)

            for chain in boundary_pchains:
                b,p = chain.compute_boundary()
                p = p.split(":")[1:]
                Bp.append(p)
    

        
        kernel_pchains = self.get_pchains(dimension)
        kernel = self.compute_cycles(dimension)
        for i in kernel:
            Zp.append(i)
        
        # stuff every possible chain into the kernel equation and then ensure that the members of Bp do not belong to the vector spanned by result
        indx = 0
        for image in Bp:
            res += str(image) + ","
            if indx+1 == len(Bp):
                res += str(image) + ">"
            indx+=1

        indx = 0
        for k in Zp:
            res2 += str(k) + ","
            if indx+1 == len(Zp):
                res2 += str(k) + ">"
            indx+=1
        return res2 + " / " + res
    
    def compute_boundary_rank(self, dimension):
        '''
        Compute the ranks of the boundaries
        '''
        p = self.get_pchains(dimension)
        
        if dimension >= self.dimension:
            return 0
        M = self.compute_boundary_matrix(dimension)
        rank = np.linalg.matrix_rank(M)
        return rank

    def compute_cycle_rank(self, dimension):
        '''
        Compute the ranks of the cycles: from the Rank-Nullity Theorem,
        we have that 

        rank Zp = col(M) - rank(M)
        '''
        if (dimension == 1):
            return len(self.get_pchains(1))
       
        boundary_rank = self.compute_boundary_rank(dimension)
        M = Matrix(self.compute_boundary_matrix(dimension))
        M_rref = np.array(M.rref()[0])

        cols = 0
        for col in zip(*M_rref):
            total = sum(col)
            if (total == 0 and -1 in col or total!=0):
                cols+=1
                
        return  cols - boundary_rank
        
    def compute_homology_rank(self, dimension):
        '''
        Compute the rank of the homologies:
        first, take the rank of the cycles, and then the rank of the boundaries
        of the next dimension, and then do the arithmetic.
        E.g.:
        rank Hp = rank Zp - rank Bp. 
        '''
        Zp = self.compute_cycle_rank(dimension)
        Bp = self.compute_boundary_rank(dimension+1)
        return Zp - Bp


def compute_boundary_with_matrix(M):
    '''
    Pass in a boundary matrix;  
    then use that matrix to produce the boundaries of each relevant chain.

    Right now we're just going to get the generators for this group, 
    returning them as columns of M.
    '''
   
    boundaries = []
    rows, cols = M.shape
    for i in range(0, cols):
        boundaries.append(M[:i])
    return boundaries

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

    print("Cycle rank Z0: " + str(A.compute_cycle_rank(1)))
    print("Boundary rank B0: " + str(A.compute_boundary_rank(2)))
    print("Homology rank H0: " + str(A.compute_homology_rank(1)))

    print("")
    
    print("Cycle rank Z1: " + str(A.compute_cycle_rank(2)))
    print("Boundary rank B1: " + str(A.compute_boundary_rank(3)))
    print("Homology rank H1: " + str(A.compute_homology_rank(2)))

    print("")
    
    print("Cycle rank Z2: " + str(A.compute_cycle_rank(3)))
    print("Boundary rank B2: " + str(A.compute_boundary_rank(4)))
    print("Homology rank H2: " + str(A.compute_homology_rank(3)))
    
    print("Homology: " + A.compute_homologies(2))
