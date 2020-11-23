import random
import numpy as np
import matplotlib.pyplot as plt
import math

def gnp(n, p, directed=False):
	adj_matrix = (np.random.rand(n, n) <= p).astype(int)
	if not directed:
		for u in range(n):
			for v in range(u+1, n):
				adj_matrix[u, v] = adj_matrix[v, u]
	# zeros out self-loops
	adj_matrix = adj_matrix * (np.ones((n, n)) - np.eye(n))
	return adj_matrix

def random_song_graph(n, p, q):
	assert 0 < p**2 < q < p < 1
	p_prime = 1 - math.sqrt(1 - 2*p + q)
	q_prime = (p - p_prime) / (p_prime * (1 - p_prime))
	adj_matrix = gnp(n, p_prime, directed=True)
	for u in range(n):
		for v in range(n):
			if adj_matrix[u, v] == 1 and adj_matrix[v, u] == 0:
				if random.random() < q_prime:
					adj_matrix[v, u] = 1
	return adj_matrix

def edge_density(adj_matrix):
	num_edges = sum(sum(adj_matrix))
	max_edges = (len(adj_matrix) - 1) * len(adj_matrix)
	return num_edges / max_edges

def edge_symmetry(adj_matrix):
	num_two_way_edges = sum(sum(adj_matrix * np.transpose(adj_matrix)))
	max_edges = (len(adj_matrix) - 1) * len(adj_matrix)
	return num_two_way_edges / max_edges

def get_adjacency_matrix(graph):
	vertices = graph[0]
	edges = graph[1]
	adj_matrix = np.zeros((len(vertices), len(vertices)))
	for edge in edges:
		adj_matrix[edge[0], edge[1]] = 1
	return adj_matrix

def get_edges_vertices(adj_matrix):
	vertices = range(len(adj_matrix))
	edges = [(u, v) for u in vertices for v in vertices if adj_matrix[u, v] == 1]
	return (vertices, edges)

def get_laplacian(adj_matrix):
	# tbd: figure out if we care about degree or subdegree
	degrees = sum(adj_matrix)
	laplacian = degrees * np.eye(len(adj_matrix)) - adj_matrix
	return laplacian

def get_spectrum(adj_matrix, plot=False):
	laplacian = get_laplacian(adj_matrix)
	evals = sorted(np.linalg.eig(laplacian)[0])
	if plot:
		plt.hist(evals, bins=len(evals))
		plt.show()
	return evals

def iid_sparsify(adj_matrix, p, directed=False):
	retain_matrix = gnp(len(adj_matrix), p, directed=directed)
	return adj_matrix * retain_matrix

def cut_density(adj_matrix, cut):
	complement = [u for u in range(len(adj_matrix)) if not u in cut]
	num_edges_in_cut = sum([adj_matrix[u, v] for u in cut for v in complement])
	max_edges_in_cut = len(cut) * len(complement)
	return num_edges_in_cut / max_edges_in_cut

def get_random_cut(n):
	return [u for u in range(n) if random.random() < 0.5]


def random_cut_density(adj_matrix, num_trials=100):
	for i in range(num_trials):
		random_cut = get_random_cut(len(adj_matrix))
		print(random_cut)
		print(cut_density(adj_matrix, random_cut))

def random_cut_density_change(adj_matrix_init, adj_matrix_sparsified, num_trials=100):
	density_changes = np.zeros(num_trials)
	for i in range(num_trials):
		random_cut = get_random_cut(len(adj_matrix))
		cut_density_init = cut_density(adj_matrix_init, random_cut)
		cut_density_sparsified = cut_density(adj_matrix_sparsified, random_cut)
		density_changes[i] = cut_density_sparsified - cut_density_init
		print("Cut Size: {}, Initial Density: {}, Sparsified Density: {}"
			.format(len(random_cut), cut_density_init, cut_density_sparsified))
	print("MSE: {}, Max Change: {}".format(np.max(np.abs(density_changes)), np.linalg.norm(density_changes)))

song_graph = random_song_graph(1000, 0.4, 0.25)
print(edge_density(song_graph))
print(edge_symmetry(song_graph))

sparsified_song_graph = iid_sparsify(song_graph, 0.75, directed=True)
print(edge_density(sparsified_song_graph))
print(edge_symmetry(sparsified_song_graph))
