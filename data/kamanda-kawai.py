import networkx as nx

G = nx.complete_graph(5)

# Standard one-liner
# weight=None tells it to use "hop count" as the distance
pos = nx.kamada_kawai_layout(G, weight=None)

# To get a training-ready array:
import numpy as np
coords = np.array(list(pos.values()))