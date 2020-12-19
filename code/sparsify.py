import networkx as nx
import numpy as np
from networkx.algorithms.connectivity.utils import build_auxiliary_edge_connectivity
from networkx.algorithms.flow import build_residual_network
from networkx.algorithms.connectivity.connectivity import local_edge_connectivity


# implements "compress" algorithm given in the paper. G is the graph to be compressed,
# epsilon is the error of the sparsifier, balance is the balance coefficient, and given
# d, this constructs a valid sparsifier with probability 1 - 1/n^d. This compression
# algorithm uses the edge indices given by G[u][v]["index"] to determine the probabilities
# with which an edge is sampled.
def compress(G, epsilon=0.2, balance=1, d=2):
    n = G.number_of_nodes()
    gamma = (1+balance) * (3 + np.log2(n))
    c = 43 * (d+7)
    ro = (c * gamma * np.log(n)) / (epsilon**2)
    print(ro/(2*np.log2(n)*np.log(n)))
    H = nx.DiGraph()
    H.add_nodes_from(range(n))
    for (u, v, w) in G.edges.data("weight", default=1):
        lam = G[u][v]["index"]
        p = min(ro/lam, 1)
        x = np.random.binomial(w, p)
        if x > 0:
            H.add_edge(u, v, weight=x/p)
    return H


# given a weighted digraph G, compute local edge connectivity of all
# edges, and store the connectivity of (u,v) at G[u][v]["index"]
def compute_edge_conn(G):
    A = build_auxiliary_edge_connectivity(G)
    R = build_residual_network(G, 'weight')
    for (u,v) in G.edges:
        c = local_edge_connectivity(G,u,v,auxiliary=A,residual=R)
        G[u][v].update({"index":c})
    return G


# given two graphs on the same vertex set, and integer k,
# generate k random directed cuts and compare the weights
# crossing them in the forward direction between G and H.
# Return tuple containing the min, max, and average
# ratios in cut values between G and H
def compare_cuts(G, H, k=None):
    n = G.number_of_nodes()
    if k is None:
        k = n
    # generate a random directed cut:
    cut_ratios = []
    for _ in range(k):
        cut = np.random.choice(2, n)
        c_G, c_H = weight_of_cut(G,cut), weight_of_cut(H,cut)
        if c_G == 0:
            if c_H == 0:
                cut_ratios.append(1)
            else:
                cut_ratios.append(np.infty)
        else:
            cut_ratios.append(c_H/c_G)

    return min(cut_ratios), max(cut_ratios), sum(cut_ratios)/len(cut_ratios)


# given a directed graph G = V, E and a directed cut
# specified as a vector {0,1}^|V|, compute the weight
# crossing the cut in the 0 -> 1 direction
def weight_of_cut(G,cut):
    c = 0
    for (u, v, w) in G.edges.data("weight", default=1):
        if cut[u] == 0 and cut[v] == 1:
            c += w
    return c


# makes a Gnp graph, with edge weights drawn independently and
# uniformly from {1, ..., w}
def make_gnp(n, p, w=1):
    G = nx.DiGraph()
    G.add_nodes_from(range(1, n))
    for u in range(n):
        for v in range(n):
            if u != v and np.random.binomial(1,p):
                G.add_edge(u, v, weight=np.random.randint(1, 1+w))
    return G


# parse connectome file, set weights to 1 (all are unweighted)
def parse_connectome(file):
    G = nx.read_graphml(file)
    H = nx.DiGraph()
    H.add_nodes_from([int(u[1:]) for u in G.nodes()])
    for u, v in G.edges():
        H.add_edge(int(u[1:]),int(v[1:]))
        H[int(u[1:])][int(v[1:])]["weight"] = 1
    return H


if __name__ == '__main__':
    G = make_gnp(50, 0.5)
    # G = parse_connectome('rhesus_brain_1.graphml')
    G = compute_edge_conn(G)
    H = compress(G,epsilon=1)

    print("G has ", G.number_of_edges(), " H has ", H.number_of_edges())
    # for (u, v, w) in G.edges.data("weight", default=1):
    #     print(u, v, w)
    #
    min_ratio, max_ratio, avg_ratio = compare_cuts(G,H, k=1000)
    print("Max cut ratio is ", max_ratio)
    print("Min cut ratio is ", min_ratio)
    print("Mean cut ratio is", avg_ratio)

