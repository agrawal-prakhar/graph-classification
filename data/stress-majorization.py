from sklearn.manifold import MDS
import networkx as nx

# 1. Create your graph
G = nx.path_graph(10) 

# 2. Get the "Ground Truth" distance matrix (Graph Distances)
dist_matrix = nx.floyd_warshall_numpy(G)

# 3. Use SMACOF (Stress Majorization) to find 2D coordinates
# n_init=4 runs it multiple times to avoid local minima (good for training data)
mds = MDS(n_components=2, dissimilarity='precomputed', normalized_stress='auto', n_init=4)
coords = mds.fit_transform(dist_matrix) 

# 'coords' is now a NumPy array of shape (N, 2) ready for your ML model