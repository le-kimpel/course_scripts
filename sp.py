import networkx as nx
import  matplotlib.pyplot as plt

'''
Here is a straightforward interpretation of 2 abstract simplicial complexes. 
I decided to represent them as graphs. Construct the graph as follows: 

An edge exists between simplices iff one is a subset of the other. This should be an onto map.
'''
# vertices: cow, rabbit, horse, dog, fish, dolphin, oyster, broccoli, fern, onion, apple....and connections

def check_subset(A, B):
    return set(A).issubset(B)

if __name__=="__main__":
    
    # now construct a graphical interpretation of the complex mentioned for part (A);
    # that is, for each element of A, draw edges between its members.
    # Manually:
    
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


    # now for the other dimension
    delta2 = [("cow","rabbit", "horse"), ("cow", "rabbit", "dog"), ("cow", "horse", "dog"),
                   ("rabbit", "horse", "dog"), ("fish", "dolphin", "oyster"), ("broccoli", "fern", "onion"), ("broccoli", "fern", "apple"), ("broccoli", "onion", "apple"), ("fern", "onion", "apple")]

    # add the vertices
    A = nx.DiGraph()
    for v in delta1:
        A.add_node(v)
    for v in delta2:
        A.add_node(v)
    
    # now add an edge between elements iff one is a subset of the other. This is not optimized.
    for i in range(0,len(delta1)):
        for j in range(0, len(delta2)):
            if(check_subset(delta1[i], delta2[j]) == True):
                A.add_edge(delta1[i], delta2[j])
    

    nx.draw(A, with_labels="True")
    plt.show()


    B = nx.DiGraph()
    

    delta3 = [("cow", "rabbit"), ("cow", "fish"), ("cow", "oyster"), ("cow", "oyster"), ("cow", "broccoli"), ("cow", "onion"), 
("cow", "apple"), ("rabbit", "fish"), ("rabbit", "oyster"), ("rabbit", "broccoli"), ("rabbit", "onion"),  ("rabbit", "apple"), ("fish", "oyster"), ("fish", "broccoli"), ("fish", "onion"), ("fish", "apple"), ("oyster", "broccoli"), ("oyster", "onion"), ("oyster", "apple"), ("broccoli", "onion"), ("broccoli", "apple"), ("onion", "apple"), ("horse", "dog"), ("horse", "dolphin"), ("horse", "fern"), ("dog", "dolphin"), ("dog", "fern"), ("dolphin", "fern")]

    delta4 = [("cow", "broccoli", "apple"), ("cow", "onion", "apple"), ("rabbit", "broccoli", "apple"),  ("rabbit", "onion", "apple"), ("fish", "broccoli", "apple"), ("fish", "onion", "apple"),  
("oyster", "broccoli", "apple"), ("oyster", "onion", "apple")]

    for v in delta3:
        B.add_node(v)
    for v in delta4:
        B.add_node(v)
    
    # now add an edge between elements iff one is a subset of the other. This is not optimized.
    for i in range(0,len(delta3)):
        for j in range(0, len(delta4)):
            if(check_subset(delta3[i], delta4[j]) == True):
                B.add_edge(delta3[i], delta4[j])
    

    nx.draw(B, with_labels="True")
    plt.show()

