import online_dictionary_learning.omp as omp
from sklearn.cluster import SpectralClustering
import numpy as np

def ssc(signals, dictionary, n_clusters, sparsity=5, verbose=False, eps=None):
	# Compute Sparse Coefficients
	coefs = np.zeros((signals.shape[0], dictionary.shape[1]))
	if verbose:
		print("Computing sparse coefficients...")
	for i in range(coefs.shape[0]):
		coefs[i]=omp.omp(dictionary, signals[i], L=sparsity, eps=eps)
		if verbose:
			if i+1<signals.shape[0]:
				if (i+1)%100==0:
					print('[' + str(i+1) + '] ' + str(np.round(((i+1)/signals.shape[0])*100,2)) + '%', end='\r')
			else:
				print('[' + str(i+1) + '] ' + str(np.round(((i+1)/signals.shape[0])*100,2)) + '%', end='\n')

	# Run SC
	if verbose:
		print("Clustering...")
	sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=0).fit(coefs)

	# Get labels
	labels = sc.labels_

	return labels
