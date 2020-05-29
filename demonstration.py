import scipy.io as spio
import numpy as np
from online_dictionary_learning import odl
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from clustering.sparse_spectral_clustering import ssc
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_mutual_info_score as ami

# Set hyperparameters
atom_num=220
sparsity=5
iterations=5000
k=6
label="dictionary/example_dictionary"
eps=None

# Read-in and reshape example data
salinas_mat=spio.loadmat('example_data/Salinas/salinas.mat')
salinas_gt_mat=spio.loadmat('example_data/Salinas/salinas_gt.mat')
salinas_image=salinas_mat['salinasA_corrected']
salinas_gt=salinas_gt_mat['salinasA_gt']
original_shape=salinas_gt.shape
signals=np.array(np.reshape(salinas_image, (-1, salinas_image.shape[2])), dtype=np.float64)
ground_truth=np.reshape(salinas_gt, -1)

# Normalise and shuffle example data for dictionary learning
signals_norm=signals.copy()-np.mean(signals.copy())
signals_norm/=np.std(signals)
np.random.shuffle(signals_norm)

# Create instance of dictionary
dictionary=odl.Dict(num_coms=atom_num)

'''
Dictionary Learning is very sensitive to initialisation, we trained some discriminative dictionaries available at https://www.comet.ml/joshuabruton/honours-project/view/, see the paper for more details
One of those dictionaries for Salinas is made available here, use atoms=np.load("dictionary/t5000_k220_L2.npy") WITH sparsity=5 for the paper result
'''
# Learn and save dictionary
print("Training dictionary...")
dictionary.fit(signals_norm, reg_term=sparsity, max_iter=iterations, showDictionary=False, prog=True, save=True, label=label, eps=eps)
atoms=np.load("%s.npy"%label)

'''
Uncomment next line to load dictionary for paper result (can comment out actual training step for time)
'''
# atoms=np.load("dictionary/t5000_k220_L2.npy")

# Remove misc. class
signals_cluster=signals[ground_truth!=0].copy()

'''
Can normalise training data aswell (remember that sparse coding does not actually require a *learned* dictionary)
'''
# signals_cluster=signals_cluster-np.mean(signals_cluster)
# signals_cluster/=np.std(signals_cluster)

# Cluster
labels=ssc(signals_cluster, atoms, k, sparsity=sparsity, verbose=True, eps=eps)
no_zeros = ground_truth[ground_truth!=0]
accuracy = ami(labels, no_zeros)

print("Saving...")

# Reconstruct image for display
count, loc, n = 0, 0, ground_truth.shape[0]
final = np.zeros(n)
while count<labels.shape[0]:
	if (ground_truth[loc]!=0):
		final[loc]=labels[count]+1
		count+=1
	loc+=1

# Display
plt.imshow(np.reshape(final, original_shape))
print("The adjusted mutual information score was %f."%accuracy)
plt.savefig("demonstration.png")
