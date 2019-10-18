# Dictionary Learning for Clustering on Hyperspectral Images
## Overview
Framework for Spectral Clustering on the Sparse Coefficients of Learned Dictionaries. This framework was created as a part of the project I presented for completion of my Computer Science Honours' degree at the University of the Witwatersrand.  
We will make the paper available either through conference proceedings or arXiv by the end of January 2020.
  
Authored by: Joshua Bruton  
Supervised by: Dr. Hairong Wang

## Contents
This repository contains implementations or usage of the following techniques:
1. Online Dictionary Learning
2. Orthogonal Matching Pursuit with dynamic stopping criteria
3. Spectral Clustering (sk-learn)
4. Hyperspectral Image processing

The repository also contains the SalinasA hyperspectral image. This and other hyperspectral data sets are available on the Grupo de Inteligencia Computacional website [here](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).

## Usage
I have created a pipenv with a lockfile so just download and use  
~~~
pipenv install
~~~
and requirements should be met. Then just run:
~~~
python demonstration.py
~~~
and the demonstration should run. It will train a dictionary and then use it for spectral clustering as discussed in the paper (yet to be released).
## Previous work
One working discriminative dictionary has been provided in the repository, all of the others are available as assets on [Comet.ml](https://www.comet.ml/joshuabruton/honours-project/view/). They were all trained using the implementation of ODL provided in this repository. Bare in mind that dictionary learning is extremely sensitive to the initialisation of the dictionary; results for different dictionaries will vary drastically.  
  
[scikit-learn](https://scikit-learn.org/stable/) was used extensively throughout this project for more stable implementations. Thanks also go to [Dave Biagioni](https://github.com/davebiagioni/pyomp/blob/master/omp.py), [mitscha](https://github.com/mitscha/ssc_mps_py/blob/master/matchingpursuit.py), and the authors of [this article](https://dl.acm.org/citation.cfm?id=1553463).
  
We compiled a paper discussing the work related to this framework and it has been submitted to a conference for review. We will expand this section after we have been notified of the result.

## Future work
This repository is licensed under the GNU General Public License and therefore is completely free to use for any project you see fit. If you do use or learn from our work, we would appreciate citations where appropriate and once we make the paper available.

## Suggestions
If there are any pressing problems with the code please open an issue and I will attend to it as timeously as is possible.
