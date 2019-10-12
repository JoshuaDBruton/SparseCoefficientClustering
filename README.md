# Dictionary Learning for Clustering on Hyperspectral Images
## Overview
Framework for Spectral Clustering on the Sparse Coefficients of Learned Dictionaries. This framework was created as a part of the project I presented for completion of my Computer Science Honours' degree at the University of the Witwatersrand.  
We will make the paper available either through a journal or archive by the end of January 2020, we would appreciate it if you cite it where appropriate.
  
Authored by: Joshua Bruton  
Supervised by: Dr. Hairong Wang
## Usage
I have created a pipenv with a lockfile so just download and use   
~~~
pipenv install
~~~
and requirements should be met. Then just run:
~~~
python demonstration.py
~~~
and the demonstration should run. It will train a dictionary and then use it for spectral clustering as discussed in the paper.
## Previous work
One working discriminative dictionary has been provided in the repository, all of the others are available as assets on Comet.ml (https://www.comet.ml/joshuabruton/honours-project/view/). They were all trained using the implementation of ODL provided in this repository. Bare in mind that dictionary learning is extremely sensitive to the initialisation of the dictionary; results for different dictionaries will vary drastically.  
  
scikit-learn was used extensively throughout this project (https://scikit-learn.org/stable/). Thanks also go to https://github.com/davebiagioni/pyomp/blob/master/omp.py, https://github.com/mitscha/ssc_mps_py/blob/master/matchingpursuit.py, and https://dl.acm.org/citation.cfm?id=1553463.  
  
We compiled a paper discussing the work related to this framework and it will be submitted to a journal for review in early November. We will expand this section after we have been notified of the result.

## Future work
This repository is licensed under the GNU General Public License and therefore is completely free to use for any project you see fit. If you do use or learn from our work, we would appreciate citations where appropriate and once we make the paper available.

## Suggestions
If there are any pressing problems with the code please open an issue and I will attend to it as timeously as is possible.
