# Dictionary Learning for Clustering on Hyperspectral Images

<div align="center">

[![GitHub issues](https://img.shields.io/github/issues/JoshuaDBruton/SparseCoefficientClustering)](https://github.com/JoshuaDBruton/SparseCoefficientClustering/issues)
[![GitHub stars](https://img.shields.io/github/stars/JoshuaDBruton/SparseCoefficientClustering)](https://github.com/JoshuaDBruton/SparseCoefficientClustering/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/JoshuaDBruton/SparseCoefficientClustering)](https://github.com/JoshuaDBruton/SparseCoefficientClustering/network)
[![GitHub license](https://img.shields.io/github/license/JoshuaDBruton/SparseCoefficientClustering)](https://github.com/JoshuaDBruton/SparseCoefficientClustering/blob/master/LICENSE)

</div>

## Overview
Framework for Spectral Clustering on the Sparse Coefficients of Learned Dictionaries. This framework was created as a part of the project I presented for completion of my Computer Science Honours' degree at the University of the Witwatersrand.  
A paper was produced for this research, it was published by Springer's Journal of Signal, Image and Video Processing. The paper can be read for free here: https://rdcu.be/b5Vsq. Please look below for citation details.

Authored by: Joshua Bruton  
Supervised by: Dr. Hairong Wang

## Contents
This repository contains implementations or usage of the following techniques:
1. Online Dictionary Learning
2. Orthogonal Matching Pursuit with dynamic stopping criteria
3. Spectral Clustering (sk-learn)

The repository also contains the SalinasA hyperspectral image. This and other hyperspectral data sets are available on the Grupo de Inteligencia Computacional website [here](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).

## Usage
I have created a requirements file. I recommend using [pipenv](https://pypi.org/project/pipenv/) with Python 3.6 to open a shell and then using
~~~
pipenv install -r requirements.txt
~~~
and requirements should be met. Then just run:
~~~
python demonstration.py
~~~
and the demonstration should run. It will train a dictionary and then use it for spectral clustering as discussed in the paper.
## Previous work
One working discriminative dictionary has been provided in the repository, all of the others are available as assets on [Comet.ml](https://www.comet.ml/joshuabruton/honours-project/view/). They were all trained using the implementation of ODL provided in this repository. Bare in mind that dictionary learning is extremely sensitive to the initialisation of the dictionary; results for different dictionaries will vary drastically.  
  
[scikit-learn](https://scikit-learn.org/stable/) was used extensively throughout this project for more stable implementations. Thanks also go to [Dave Biagioni](https://github.com/davebiagioni/pyomp/blob/master/omp.py), [mitscha](https://github.com/mitscha/ssc_mps_py/blob/master/matchingpursuit.py), and the authors of [this article](https://dl.acm.org/citation.cfm?id=1553463).

## Future work
This repository is licensed under the GNU General Public License and therefore is completely free to use for any project you see fit. If you do use or learn from our work, we would appreciate if you cited the following details:
```
@article{10.1007/s11760-020-01750-z, 
  author = {Bruton, Joshua and Wang, Hairong}, 
  title = {{Dictionary learning for clustering on hyperspectral images}}, 
  issn = {1863-1703}, 
  doi = {10.1007/s11760-020-01750-z},
  pages = {1--7}, 
  journal = {Signal, Image and Video Processing}, 
  year = {2020}
}

```
Or:    
Bruton, J., Wang, H. Dictionary learning for clustering on hyperspectral images. SIViP (2020). https://doi.org/10.1007/s11760-020-01750-z

[The paper can be read for free.](https://rdcu.be/b5Vsq)

## Suggestions
If there are any pressing problems with the code please open an issue and I will attend to it as timeously as is possible.
