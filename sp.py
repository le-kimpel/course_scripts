import networkx as nx
import  matplotlib.pyplot as plt

'''
Here is a straightforward interpretation of 2 abstract simplicial complexes. 
I decided to represent them as graphs. Construct the graph as follows: 

1) Each vertex of the ASC is a vertex of its graph. 
2) An edge in the graph is equivalent to saying that the vertex  is a member of a simplex.

So if {A,B,C} is a simplex, then there exists an edge between A,B,C. This edge should be labeled.
Then the degree of a vertex in a particular cluster corresponds to the dimension of its facets. 
The potential downside of this representation is that it only gives a view of the facets of the complex and does not do a great job of providing insight about subsets, unless extra labelings are provided, or if this becomes a multigraph. (So edges between various simplices are colored differently.)
'''

# vertices: cow, rabbit, horse, dog, fish, dolphin, oyster, broccoli, fern, onion, apple....and connections
if __name__=="__main__":
    complexA = nx.Graph()
    complexB = nx.Graph()

    vertices = ["cow", "rabbit",  "horse", "dog", "fish", "dolphin", "oyster", "broccoli", "fern", "onion", "apple"]

    for v in vertices:
        complexA.add_node(v)
        complexB.add_node(v)

    # now construct a graphical interpretation of the complex mentioned for part (A);
    # that is, for each element of A, draw edges between its members.
    # Manually:
    edgelistA =  [ ("cow","rabbit"),
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

    for edge in edgelistA:
        complexA.add_edge(edge[0],edge[1])

    # produce the visual representation; here it should be a disconnected graph.
    nx.draw(complexA, with_labels=True)
    plt.show()


    edgelistB = [("cow", "rabbit"), ("cow", "fish"), ("cow", "oyster"), ("cow", "oyster"), ("cow", "broccoli"), ("cow", "onion"), 
("cow", "apple"), ("rabbit", "fish"), ("rabbit", "oyster"), ("rabbit", "broccoli"), ("rabbit", "onion"),  
("rabbit", "apple"), ("fish", "oyster"), ("fish", "broccoli"), ("fish", "onion"), ("fish", "apple"), 
("oyster", "broccoli"), ("oyster", "onion"), ("oyster", "apple"), ("broccoli", "onion"), ("broccoli", "apple"), 
("onion", "apple"), 
("horse", "dog"), ("horse", "dolphin"), ("horse", "fern"), ("dog", "dolphin"), ("dog", "fern"), 
("dolphin", "fern"), 
("cow", "broccoli", "apple"), ("cow", "onion", "apple"), ("rabbit", "broccoli", "apple"),  
("rabbit", "onion", "apple"), ("fish", "broccoli", "apple"), ("fish", "onion", "apple"),  
("oyster", "broccoli", "apple"), ("oyster", "onion", "apple")]


    for edge in edgelistB:
        complexB.add_edge(edge[0],edge[1])

    # produce the visual representation; here it should be a disconnected graph.
    nx.draw(complexB, with_labels=True)
    plt.show()
