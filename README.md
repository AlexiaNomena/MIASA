
# **M**etrized **I**dentification and **A**nalysis of **S**imilarity and **A**ssociation

### Operating System
This workflow was tested on macOS Monterey Versoin 12.5 and CentOS Linux release 7.9.2009 Core [(HPC Fu Berlin)](https://www.fu-berlin.de/en/sites/high-performance-computing/index.html)

### Prerequisites
#### Python

version 3.10.4

Packages:
numpy (1.21.5),
scipy (1.7.3)
pip,
pandas (1.4.3),
seaborn,
scikit-learn (1.1.3),
matplotlib (3.5.2),
openpyxl,
statsmodels (0.13.2)

#### Install Conda/Miniconda
Conda will manage the dependencies of our pipeline. Instructions can be found here:
[https://docs.conda.io/projects/conda/en/latest/user-guide/install](https://docs.conda.io/projects/conda/en/latest/user-guide/install)

Create a new environment from the given environment config in [env.yml](https://github.com/KleistLab/VASIL/blob/main/env/env.yml)

```
conda env create -f env/env.yml
```

This step may take a few minutes. 

To activate the enviromnent 
```
conda activate VASIL
```

### In folder Manuscripts_examples:
This folder contains all the codes that were used to produce the manuscripts results

Add environment to jupyter notebook

`conda deactivate MIASA` (if the environment is activated)

`conda install -c anaconda ipykernel` (only if `ipykernel` is not yet installed)

`python -m ipykernel install --user --name=MIASA`

#### List of codes
`miasa_Steps.ipynb`: jupyter notebook for a step by step guidance through the MIASA framework.

`miasa_Dist.ipynb`, `miasa_Corr.ipynb`, `miasa_GRN.ipynb`: python code for using MIASA as a blackbox for the three dataset problems highlighted in the paper (similarity distances are Euclidean).

`miasa_NonEucl_Dist.ipynb`, `miasa_NonEucl_Corr.ipynb`, `miasa_NonEucl_GRN.ipynb`: python code for using MIASA as a blackbox for the three dataset problems highlighted in the paper (similarity distances are non-Euclidean).

`class_experiment.py`: python code for classification experiments when the true clusters are known and included in the data generating function which must return data in a specific format (e.g. function `generate_data_dist` in module `Methods/simulate_class_data.py`)


### Snakemake pipeline
For a general execution of the MIASA framework

#### Install Snakemake
Snakemake is the workflow management system we use. Install it in your activated environment like this:

```
conda install -c conda-forge -c bioconda snakemake
```

NOTE: In case conda is not able to find the packages for snakemake (which was the case for the Linux version), you can install mamba in your environment

```
conda install -c conda-forge mamba
```

and download snakemake with

```
mamba install -c conda-forge -c bioconda snakemake
```

Detailed Snakemake installation instruction using mamba can be found here:
[https://snakemake.readthedocs.io/en/stable/getting_started/installation.html](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html)


