import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import random
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
    print(ro/(np.log2(n)*np.log(n)))
    print(np.log2(n)*np.log(n))
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


# sparsify G by choosing to maintain each edge with probability p
# and multiplying the weights of every edge we keep by 1/p
def randomly_sparsify(G, p):
    edges_kept = []
    for u, v, w in G.edges.data("weight", default=1):
        if np.random.binomial(1,p):
            edges_kept.append((u,v,w))
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    for u, v, w in edges_kept:
        H.add_edge(u, v)
        H[u][v]["weight"] = w/p
    return H


# computes the normalized laplacian of G
def laplacian(G):
    n = G.number_of_nodes()
    A  = np.zeros((n,n))
    for u, v, w in G.edges.data("weight", default=1):
        A[u][v] = w
    D = np.zeros((n,n))
    for v in G.nodes():
        if G.out_degree(v) != 0:
            D[v][v] = 1/np.sqrt(G.out_degree(v))
    return D - A


# generate a random unit vector in R^d
def make_rand_vector(d):
    vec = [random.gauss(0, 1) for _ in range(d)]
    mag = sum(x**2 for x in vec) ** .5
    return np.array([x/mag for x in vec])


# given two graph G, H on the same vertex set, and integer k,
# generate k random unit vectors and compute the absolute
# difference of x^T L_G x and x^T L_H x for all x, where
# L_G denotes the normalized laplacian.
# Return tuple containing the min, max, and average
# differences between G and H over all k trials
def compare_spectra(G, H, k=None):
    n = G.number_of_nodes()
    if k is None:
        k = n
    # generate a random directed cut:
    spectra_ratios = []
    for _ in range(k):
        u = make_rand_vector(n)
        L_G, L_H = laplacian(G), laplacian(H)
        c_G, c_H = np.dot(u.T, (np.dot(L_G, u))), np.dot(u.T, (np.dot(L_H, u)))
        if c_G == 0:
            if c_H == 0:
                spectra_ratios.append(1)
            else:
                spectra_ratios.append(np.infty)
        else:
            spectra_ratios.append(np.abs(c_H - c_G))

    return min(spectra_ratios), max(spectra_ratios), sum(spectra_ratios)/len(spectra_ratios)


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


# returns a random graph from G_{n,p,q}
def gnpq(n, p, q):
    assert 0 < p**2 < q < p < 1
    p_prime = 1 - math.sqrt(1 - 2*p + q)
    q_prime = (p - p_prime) / (p_prime * (1 - p_prime))
    G = make_gnp(n, p_prime)
    for u in range(n):
        for v in range(n):
            if G.has_edge(u,v) and not G.has_edge(v, u) == 0:
                if random.random() < q_prime:
                    G.add_edge(v, u)
                    G[v][u]["weight"] = 1
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

# run cut sparsifier tests
def run_cut_tests(G, precision=.01):
    min_ratios, max_ratios, avg_ratios = [], [], []
    p_values = [x*precision for x in range(round(1/precision))]
    for p in p_values:
        H = randomly_sparsify(G, p)
        min_ratio, max_ratio, avg_ratio = compare_cuts(G, H, k=1000)
        min_ratios.append(min_ratio)
        max_ratios.append(max_ratio)
        avg_ratios.append(avg_ratio)
    return min_ratios, max_ratios, avg_ratios, p_values


# plot results of cut tests
def plot_cut_results(min_ratios, max_ratios, avg_ratios, p_values):
    min_line, = plt.plot(p_values, min_ratios, label='Minimum Cut Ratio Sampled')
    max_line, = plt.plot(p_values, max_ratios, label='Maximum Cut Ratio Sampled')
    avg_line, = plt.plot(p_values, avg_ratios, label='Average Cut Ratio Sampled')
    plt.legend(handles=[min_line, max_line, avg_line])
    plt.xlabel("$\gamma$ Values")
    plt.ylabel("Cut Weight Ratio of G vs. H")
    plt.ylim(0, 2)
    plt.xlim(0, 1)
    plt.show()
    plt.close()


# run spectral sparsification tests
def run_spectrum_tests(G, precision=.01):
    min_ratios, max_ratios, avg_ratios = [], [], []
    p_values = [x*precision for x in range(round(1/precision))]
    for p in p_values:
        H = randomly_sparsify(G, p)
        min_ratio, max_ratio, avg_ratio = compare_spectra(G, H, k=1000)
        min_ratios.append(min_ratio)
        max_ratios.append(max_ratio)
        avg_ratios.append(avg_ratio)
    return min_ratios, max_ratios, avg_ratios, p_values


# plot results of spectrum tests
def plot_spectra_results(min_ratios, max_ratios, avg_ratios, p_values):
    # min_line, = plt.plot(p_values, min_ratios, label='Minimum Quadratic Form Difference Sampled')
    max_line, = plt.plot(p_values, max_ratios, label='Maximum Quadratic Form Difference Sampled')
    avg_line, = plt.plot(p_values, avg_ratios, label='Average Quadratic Form Difference Sampled')
    plt.legend(handles=[max_line, avg_line])
    plt.xlabel("$\gamma$ Values")
    plt.ylabel("Quadratic Form Difference of G vs. H")

    plt.show()
    plt.close()


if __name__ == '__main__':
    G = make_gnp(50, 0.5)
    # G = gnpq(50, 0.5, 0.3)
    # G = parse_connectome("rhesus_cerebral.cortex_1.graphml")
    # G = parse_connectome("rhesus_brain_1.graphml")


    # min_ratios, max_ratios, avg_ratios, p_values = run_cut_tests(G)
    # plot_cut_results(min_ratios, max_ratios, avg_ratios, p_values)
    #

    min_ratios, max_ratios, avg_ratios, p_values = run_spectrum_tests(G)
    plot_spectra_results(min_ratios, max_ratios, avg_ratios, p_values)


    # G = compute_edge_conn(G)
    # H = compress(G,epsilon=1)
    # print("G has ", G.number_of_edges(), " H has ", H.number_of_edges())
    # min_ratio, max_ratio, avg_ratio = compare_cuts(G,H, k=1000)
    # print("Max cut ratio is ", max_ratio)
    # print("Min cut ratio is ", min_ratio)
    # print("Mean cut ratio is", avg_ratio)

