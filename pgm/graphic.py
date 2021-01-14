import networkx as nx
import matplotlib.pyplot as plt


def draw_G(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G,
                           pos=pos,
                           node_size=1800)
    nx.draw_networkx_labels(G,
                            pos=pos,
                            font_size=10,
                            font_color="white",
                            labels=G.nodes())
    nx.draw_networkx_edges(G, pos)
    plt.show()
