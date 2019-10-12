'''
Joshua Bruton
Last updated: 23 May 2019
Two implementations of Orthogonal Matching Pursuit
Adapted from:
	https://github.com/davebiagioni/pyomp/blob/master/omp.py, and
	https://github.com/mitscha/ssc_mps_py/blob/master/matchingpursuit.py
'''
import numpy as np
from numpy import linalg
from copy import deepcopy
import spams

# This version does not target specific sparsity levels
def orthogonal_mp(D, x, L=100):
	n, K = D.shape
	A = np.ndarray((D.shape[1],))
	a = 0
	residual = deepcopy(x)
	indx = np.zeros((L,), dtype=int)
	for j in range(L):
		proj = np.dot(D.T, residual)
		pos = np.argmax(np.abs(proj))
		indx[j] = int(pos)
		a = np.dot(np.linalg.pinv(D[:,indx[0:j+1]]),x)
		if np.sum(residual**2) < 1e-5:
			break
	temp = np.zeros((K,))
	temp[indx[0:j+1]] = deepcopy(a)
	A[:,] = temp
	return A

# Dynamic stopping criteria, very effective
def omp(D, x, L=100, eps=None):
	residual = x
	idx = []
	if eps == None:
		stopping_condition = lambda: len(idx) == L
	else:
		stopping_condition = lambda: np.inner(residual, residual) <= eps
	while not stopping_condition():
		lam = np.abs(np.dot(residual, D)).argmax()
		idx.append(lam)
		gamma, _, _, _ = linalg.lstsq(D[:, idx], x, rcond=None)
		residual = x - np.dot(D[:, idx], gamma)
	alpha = np.zeros(D.shape[1])
	for i,val in enumerate(idx):
		alpha[val]=gamma[i]
	return alpha
