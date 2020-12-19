import random
import numpy as np
import matplotlib.pyplot as plt
import math


def gnp(n, p, directed=False):
	'''
	Generates a random graph from the G_{n,p} model
	'''
	adj_matrix = (np.random.rand(n, n) <= p).astype(int)
	if not directed:
		for u in range(n):
			for v in range(u+1, n):
				adj_matrix[u, v] = adj_matrix[v, u]
	# zeros out self-loops
	adj_matrix = adj_matrix * (np.ones((n, n)) - np.eye(n))
	return adj_matrix


def gnpq(n, p, q):
	'''
	Generates a random graph from the G_{n,p,q} model
	'''
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


def geom(n, p, c):
	'''
	Generates a random graph from the G_{V,phi} model where V consists of
	n points chosen uniformly at random from [0,1]^d and phi(z) = A exp(-cz),
	where A is a normalization constant to ensure that the average probability
	of drawing an edge is p.
	'''
	points = np.random.random((n, 2))
	distances = np.asarray([[np.linalg.norm(x - y) for x in points] for y in points])
	prob_matrix = np.exp(-1 * c * distances) - np.eye(n)
	prob_matrix = prob_matrix * n * (n-1) / np.sum(prob_matrix) * p
	adj_matrix = (np.random.rand(n, n) <= prob_matrix).astype(int)
	return adj_matrix


def edge_density(adj_matrix):
	'''
	Finds the percentage of possible edges contained in an adjacency matrix
	'''
	num_edges = sum(sum(adj_matrix))
	max_edges = (len(adj_matrix) - 1) * len(adj_matrix)
	return num_edges / max_edges


def edge_symmetry(adj_matrix):
	'''
	Finds the percentage of bidirectional edges out of all possible edges in an
	adjacency matrix
	'''
	num_two_way_edges = sum(sum(adj_matrix * np.transpose(adj_matrix)))
	max_edges = (len(adj_matrix) - 1) * len(adj_matrix)
	return num_two_way_edges / max_edges


def count_2_subgraphs(gnp_graph, gnpq_graph, geom_graph):
	'''
	Categorizes each pair of vertices in several graphs into one of three 
	subgroups, counts each such group, and plots a histogram comparing 
	the distributions for each graph with the empirical counts from the 
	connectome from Song et al.
		0: u    v
		1: u--->v
		2: u<-->v
	'''
	adj_matrices = [gnp_graph, gnpq_graph, geom_graph]
	num_graphs = 3

	num_subgroups = 3

	song_counts = np.asarray([3312, 495, 218])
	song_freq = song_counts / sum(song_counts)

	counts = np.zeros((num_graphs, num_subgroups))
	freq = np.zeros((num_graphs, num_subgroups))
	
	for i, adj_matrix in enumerate(adj_matrices):
		num_vertices = len(adj_matrix)
		for u in range(num_vertices):
			for v in range(u, num_vertices):
				if adj_matrix[u, v] == 1 and adj_matrix[v,u] == 1:
					counts[i, 2] += 1
				elif adj_matrix[u, v] == 0 and adj_matrix[v,u] == 0:
					counts[i, 0] += 1
				else:
					counts[i, 1] += 1
		freq[i] = counts[i] / sum(counts[i])

	fig, ax = plt.subplots()
	x = np.arange(1, num_subgroups+1)
	x_labels = [r'$u \quad v$', r'$u \rightarrow v$', r'$u \leftrightarrow v$']
	width = 1. / (num_graphs + 2) # width of bar
	labels = [r'$\mathbf{G}_{n,p}$', r'$\mathbf{G}_{n,p,q}$', r'$\mathbf{G}_{V,\phi}$']

	# Plots frequencies if true, counts if false
	plot_freq = True
	if plot_freq:
		for i in range(num_graphs):
			ax.bar(x + (i * width), freq[i, :], width, label=labels[i])
		ax.bar(x + (num_graphs * width), song_freq, width, label="Song Connectome")
		ax.set_ylabel('Frequency')
		ax.set_title('Frequency of 2-Subgroups')
	else:
		for i in range(num_graphs):
			ax.bar(x + (i * width), counts[i, :], width, label=labels[i])
		ax.bar(x + (num_graphs * width), song_counts, width, label="Song Connectome")
		ax.set_ylabel('Count')
		ax.set_title('Count of 2-Subgroups')

	ax.set_xticks(x + width)
	ax.set_xticklabels(x_labels)
	ax.set_xlabel('Subgroup')
	ax.legend()
	plt.show()


def count_3_subgraphs(gnp_graph, gnpq_graph, geom_graph):
	'''
	Categorizes each tripe of vertices in several graphs into one of sixteen 
	subgroups, counts each such group, and plots a histogram comparing 
	the distributions for each graph with the empirical counts from the 
	connectome from Song et al.
		0: u    v    w    u
		1: u--->v    w    u
		2: u<-->v    w    u
		3: u<---v--->w    u
		4: u--->v<---w    u
		5: u--->v--->w    u
		6: u--->v<-->w    u
		7: u<---v<-->w    u
		9: u<---v--->w--->u
		10: u--->v--->w--->u
		8: u<-->v<-->w    u
		11: u<---v--->w<-->u
		12: u--->v--->w<-->u
		13: u--->v<---w<-->u
		14: u<-->v<-->w--->u
		15: u<-->v<-->w<-->u
	'''
	adj_matrices = [gnp_graph, gnpq_graph, geom_graph]
	num_graphs = 3

	num_subgroups = 16

	song_counts = np.asarray([1375, 579, 274, 33, 25, 41, 41, 24, 17, 9, 4, 6, 4, 5, 6, 3])
	song_freq = song_counts / sum(song_counts)

	counts = np.zeros((num_graphs, num_subgroups))
	freq = np.zeros((num_graphs, num_subgroups))
	for i, adj_matrix in enumerate(adj_matrices):
		num_vertices = len(adj_matrix)
		num_triangles = num_vertices * (num_vertices-1) * (num_vertices-2) / 6
		for u in range(num_vertices):
			for v in range(u+1, num_vertices):
				for w in range(v+1, num_vertices):
					# Categorizes each triple based on the existence of different edges
					uv = adj_matrix[u, v]
					vu = adj_matrix[v, u]
					uw = adj_matrix[u, w]
					wu = adj_matrix[w, u]
					vw = adj_matrix[v, w]
					wv = adj_matrix[w, v]
					total_edges = uv + vu + uw + wu + vw + wv
					uv_bi = uv * vu
					vw_bi = vw * wv
					wu_bi = wu * uw
					contains_bi = (uv_bi + vw_bi + wu_bi > 0)
					contains_2_bi = (uv_bi + vw_bi + wu_bi > 1)
					u_source = uv * uw
					v_source = vu * vw
					w_source = wu * wv
					contains_source = (u_source + v_source + w_source > 0)
					u_sink = vu * wu
					v_sink = uv * wv
					w_sink = uw * vw
					contains_sink = (u_sink + v_sink + w_sink > 0)
					forward_cycle = uv * vw * wu
					back_cycle = vu * wv * uw
					contains_cycle = (forward_cycle + back_cycle > 0)
					if total_edges == 0:
						subgraph = 0
					elif total_edges == 1:
						subgraph = 1
					elif total_edges == 2:
						if contains_bi:
							subgraph = 2
						elif contains_source:
							subgraph = 3
						elif contains_sink:
							subgraph = 4
						else:
							subgraph = 5
					elif total_edges == 3:
						if contains_cycle:
							subgraph = 10
						elif contains_bi and contains_sink:
							subgraph = 6
						elif contains_bi and contains_source:
							subgraph = 7
						else:
							subgraph = 9
					elif total_edges == 4:
						if contains_2_bi:
							subgraph = 8
						elif (u_source and vw_bi) or (v_source and wu_bi) or (w_source and uv_bi):
							subgraph = 11
						elif (u_sink and vw_bi) or (v_sink and wu_bi) or (w_sink and uv_bi):
							subgraph = 13
						else:
							subgraph = 12
					elif total_edges == 5:
						subgraph = 14
					elif total_edges == 6:
						subgraph = 15
					counts[i, subgraph] += 1
		freq[i] = counts[i] / num_triangles

	fig, ax = plt.subplots()
	x = np.arange(1, num_subgroups+1)
	width = 1. / (num_graphs + 2) # width of bar
	labels = [r'$\mathbf{G}_{n,p}$', r'$\mathbf{G}_{n,p,q}$', r'$\mathbf{G}_{V,\phi}$']

	# Plots frequencies if true, counts if false
	plot_freq = True
	if plot_freq:
		for i in range(num_graphs):
			ax.bar(x + (i * width), freq[i, :], width, label=labels[i])
		ax.bar(x + (num_graphs * width), song_freq, width, label="Song Connectome")
		ax.set_ylabel('Frequency')
		ax.set_title('Frequency of 3-Subgroups')
	else:
		for i in range(num_graphs):
			ax.bar(x + (i * width), counts[i, :], width, label=labels[i])
		ax.bar(x + (num_graphs * width), song_counts, width, label="Song Connectome")
		ax.set_ylabel('Count')
		ax.set_title('Count of 3-Subgroups')

	# Uses log scale if true
	log_scale = True
	if log_scale:
		ax.set_yscale('log')

	ax.set_xticks(x + width)
	ax.set_xticklabels(x)
	ax.set_xlabel('Subgroup')
	ax.legend()
	plt.show()


def generate_plots():
	'''
	Generates plots used in report.
	'''
	song_p = 0.116
	song_q = 0.054
	n = 100

	gnp_graph = gnp(100, song_p, directed=True)
	gnpq_graph = gnpq(100, song_p, song_q)
	geom_graph = geom(100, song_p, 8)

	count_2_subgraphs(gnp_graph, gnpq_graph, geom_graph)
	count_3_subgraphs(gnp_graph, gnpq_graph, geom_graph)


generate_plots()
