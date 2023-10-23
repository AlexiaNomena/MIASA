### MIASA framework
**M**etrized **I**dentification and **A**nalysis of **S**imilarity and **A**ssociation patterns in multivariate datasets

### Theory
Paper TBA

### Software Requirement
python version: 3.10 or +

pakages: [scikit-learn](https://scikit-learn.org/stable/), [umap](https://umap-learn.readthedocs.io/en/latest/), [numpy](https://numpy.org/), [scipy](https://scipy.org/), ... (find missing ones from the code run error logs)

### In folder Manuscripts_examples:
This folder contains all the codes that were used to produce the manuscripts results

`miasa_Steps.ipynb`: jupyter notebook for a step by step guidance through the MIASA framework.

`miasa_Dist.ipynb`, `miasa_Corr.ipynb`, `miasa_GRN.ipynb`: python code for using MIASA as a blackbox for the three dataset problems highlighted in the paper (similarity distances are Euclidean).

`miasa_NonEucl_Dist.ipynb`, `miasa_NonEucl_Corr.ipynb`, `miasa_NonEucl_GRN.ipynb`: python code for using MIASA as a blackbox for the three dataset problems highlighted in the paper (similarity distances are non-Euclidean).

`class_experiment.py`: python code for classification experiments when the true clusters are known and included in the data generating function which must return data in a specific format (e.g. function `generate_data_dist` in module `Methods/simulate_class_data.py`)


### Snakemake pipeline
For a general execution of the MIASA framework


