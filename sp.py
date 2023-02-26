import networkx as nx
import  matplotlib.pyplot as plt

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


    edgelistB = []
