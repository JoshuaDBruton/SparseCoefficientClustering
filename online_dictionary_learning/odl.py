'''
Joshua Bruton
Last updated: 23 May 2019
Implementation of Online Dictionary Learning
See: J Mairal "Online Dictionary Learning for Sparse Coding
'''
import numpy as np
import random
import matplotlib.pyplot as plt
import online_dictionary_learning.omp as omp

# Dictionary update step from ODL, taken from J Mairal "Online Dictionary Learning for Sparse Coding"
def updateDict(D, A, B):	
	D = D.copy()
	DA  = D.dot(A)
	count = 0
	for j in range(D.shape[1]):
		u_j = (B[:, j] - np.matmul(D, A[:, j])) / A[j, j] + D[:, j]
		D[:, j] = u_j/max([1, np.linalg.norm(u_j)])
	return D

class Dict:
	'This class creates, trains and outputs dictionaries with the online dictionary learning algorithm'
	def __init__(self, num_coms):
		self.num_coms = num_coms
		self.signals = None
		self.atoms = None
		self.coefs = None
		self.regTerm = None
		self.max_iter = None
		self.prog = None
	# Outputs a the learnt dictionary
	def showDict(self):
		self.atoms=self.atoms.transpose()
		split=int(np.sqrt(self.atoms.shape[1]))
		new_tiles=np.zeros(((self.num_coms,split,split)))
		for i in range(self.num_coms):
			new_tiles[i]=np.reshape(self.atoms[i][:split*split-self.atoms.shape[1]], (split,split))
		f, ax=plt.subplots(self.num_coms//split,split)
		count=0
		for i in range(self.num_coms//split):
			for j in range(split):
				ax[i][j].imshow(new_tiles[count],cmap="gray")
				ax[i][j].axis("off")
				count+=1
		plt.show()
		self.atoms=self.atoms.transpose()

	# Returns the dictionary as array of atoms, the most recently saved one
	def getAtoms(self):
		return self.atoms

	# SETS and returns coeficients
	def getCoefs(self):
		self.coefs = np.zeros((self.signals.shape[0],self.num_coms))
		for i in range(self.signals.shape[0]):
			self.coefs[i] = omp.omp(self.atoms, self.signals[i], self.regTerm)
			if self.prog:
				if i+1<self.signals.shape[0]:
					print('[' + str(i+1) + '] ' + str(np.round(((i+1)/self.signals.shape[0])*100,2)) + '%', end='\r')
				else:
					print('[' + str(i+1) + '] ' + str(np.round(((i+1)/self.signals.shape[0])*100,2)) + '%', end='\n')
		return self.coefs

	# Shows the current error in the sparse approximation
	def showError(self, recal=True):
		if recal:
			self.getCoefs()
		res = self.signals - (self.coefs.dot(self.atoms.T))
		errors = np.linalg.norm(res, axis=1)**2
		overall_error = (1/self.signals.shape[0])*np.sum(errors)
		#print('Representation Error: ' + str(np.round(overall_error,2)), end='\n')
		return overall_error

	# Draws a random signal from the data
	def drawRand(self):
		rand = random.randint(0,self.signals.shape[0]-1)
		return self.signals[rand]

	# Gets a single coefficient (from OMP implementation)
	def get_co(self, signal):
		return omp.omp(self.atoms, signal, self.regTerm)

	# Initialises dictionary to random signals from data
	def initialDict(self):
		self.atoms = np.zeros((self.signals.shape[1], self.num_coms))
		for i in range(self.num_coms):
			self.atoms[:,i] = self.drawRand()

	# Prints the progress to terminal (nicely)
	def update_progress(self, curr_it):
		if (curr_it+1) < self.max_iter:
			print('[{0}] {1}%'.format((curr_it+1), np.round(((curr_it+1)/self.max_iter)*100,2)), end='\r')
		else:
			print('[{0}] {1}%'.format((curr_it+1), np.round(((curr_it+1)/self.max_iter)*100,2)), end='\n')

	# Trains
	def fit(self, signals, reg_term=100, max_iter=100, showDictionary=False, showRecExample=False, prog=False, comet=None, logRep=False, save=False, label='Dictionary'):
		# Initialisation
		self.orig = signals
		self.regTerm=reg_term
		self.signals = signals
		self.initialDict()
		self.max_iter = max_iter
		self.prog=prog

		signal = self.drawRand()
		A = np.zeros((self.num_coms, self.num_coms))
		B = np.zeros((self.signals.shape[1], self.num_coms))
		for t in range(1,max_iter):
			signal = self.drawRand()
			alpha = self.get_co(signal)
			A += (alpha.dot(alpha.T))
			B += (signal[:, None]*alpha[None,:])
			self.atoms = updateDict(self.atoms, A, B)
			if prog==True:
				self.update_progress(t)
			if logRep:
				if t%100==0:
					comet.log_metric('Representation error', self.showError(), step=t)
		if save:
			np.save('%s.npy' %label, self.atoms)
			if logRep:
				comet.log_asset('%s.npy' %label)

		if showRecExample == True:
			alpha = omp.omp(self.atoms, signal, reg_term)
			plt.plot(signal, '-g', label='Original')
			plt.plot((self.atoms.dot(alpha)), ':b', label='Reconstruction')
			plt.legend()
			plt.show()

		if showDictionary==True:
			self.showDict()
