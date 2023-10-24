
# **M**etrized **I**dentification and **A**nalysis of **S**imilarity and **A**ssociation

### Operating System
This workflow was tested on macOS Monterey Versoin 12.5

### Prerequisites
#### Python

version 3.10.4

Packages:
numpy (1.21.5),
scipy (1.7.3)
pip,
pandas (1.4.3),
seaborn,
regex,
scikit-learn (1.1.3),
matplotlib (3.5.2),
openpyxl,
xlrd,
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
conda activate MIASA
```
Install  other packages with pip
```
pip install scikit-learn-extra
pip install xlrd
```

```
pip install umap-learn
```

### In folder Manuscripts_examples:
This folder contains all the codes that were used to produce the manuscripts results

Add environment to jupyter notebook

`conda deactivate` (only if the environment is activated)

`conda install -c anaconda ipykernel` (only if `ipykernel` is not yet installed)

`python -m ipykernel install --user --name=MIASA`

#### List of codes
`miasa_Steps.ipynb`: jupyter notebook for a step by step guidance through the MIASA framework.

`miasa_Dist.ipynb`, `miasa_Corr.ipynb`, `miasa_GRN.ipynb`: python code for using MIASA for the three dataset problems highlighted in the paper (similarity distances are Euclidean).

`miasa_NonEucl_Dist.ipynb`, `miasa_NonEucl_Corr.ipynb`, `miasa_NonEucl_GRN.ipynb`: python code for using MIASA for the three dataset problems highlighted in the paper (similarity distances are non-Euclidean).

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

## Input
As an input, the pipeline requires the paths to the consonar data, Escape data, and GInPipe case ascertainment data.
These variables are stored in [`config.yaml`](https://github.com/KleistLab/VASIL/blob/main/config.yaml).
For more information about the YAML markup format refer to documentation: [https://yaml.org](https://yaml.org)

### Available Clustering Method Options
```
"Agglomerative_*"# where * is a linkage method of `sklearn.cluster.AgglomerativeClustering', 
"Kmeans", sklearn.cluster.KMeans
"Kmedoids", sklearn_extra.cluster.KMedoids
"Spectral", sklearn.cluster.SpectralClustering
"GMM", # sklearn.mixture.GaussianMixture
BayesianGMM", # sklearn.mixture.BayesianGaussianMixture
"DBSCAN", # sklearn.cluster.DBSCAN
```
## Execution

If your environment is not yet activated, type

```
conda activate MIASA
```
Go to the pipeline directory (where the Snakefile named [`VASIL`](https://github.com/AlexiaNomena/MIASA/blob/main/MIASA) is located) and enter the following command to execute the pipeline

```
snakemake --snakefile MIASA --configfile path/to/config.yaml -j -d path/to/workdir
```
With parameter `--configfile` you can give the configuration file, described above. The `-j` parameter determines the number of available CPU cores to use in the pipeline. Optionally you can provide the number of cores, e.g. `-j 4`. With parameter `-d` you can set the work directory, i.e. where the results of the pipeline are written to.

## Output
The main pipeline (`config.yaml`) creates a folder *results*, containing all (intermediate) output, with the following structure:

```
|-- results
 	|-- miasa_results.pck	# pickled python dictionary containing the results
|-- plots	
	|-- UMAP_One_Panel.pdf/.svg # UMAP projection of the results
	|-- UMAP_Separate_Panels.pdf/.svg # UMAP projection separate predicted panels
	|-- tSNE_One_Panel.pdf/.svg # t-SNE projection of the results 
	|-- tSNE_Separate_Panels.pdf/.svg # t-SNE projection of the results, separate predicted panels
|-- scores
	|-- scored_miasa_results.pck # pickled results containing cluster evaluation scores
	|-- Cluster_scores.pdf/.svg # cluster score plots (Elbow, Distortion, Silhouette)	
```

## Demo
Demo datasets are provided in the repository folder [`demo`](https://github.com/KleistLab/VASIL/tree/main/demo)

If your environment is not yet activated, type

```
conda activate MIASA
```

To run the pipeline go into the repository where the snakefile [`MIASA`](https://github.com/AlexiaNomena/MIASA/blob/main/MIASA) is located and run

```
snakemake --snakefile MIASA --configfile demo/demo_config.yaml -j -d demo

```

### Expected Runtime for demo
Less than 5 min

Deactivate the environment to exit the pipeline
```
conda deactivate
```

The result folder is created in the [`demo`](./demo) folder where you find the output files, as described above. 

The projection of all predicted cluster on the same pannel looks like
<img src="https://github.com/AlexiaNomena/MIASA/blob/main/demo/plots/UMAP_One_Panel.svg" width="500" height="500">

The projection of all predicted cluster separated pannels:

- With convex hull of prediction
<img src="https://github.com/AlexiaNomena/MIASA/blob/main/demo/plots/UMAP_Separate_Panels_p1.svg" width="500" height="500">

- Without convex hull of prediction
<img src="https://github.com/AlexiaNomena/MIASA/blob/main/demo/plots/UMAP_Separate_Panels_p2.svg" width="500" height="500">

Cluster evaluation plot looks like this
<img src="https://github.com/AlexiaNomena/MIASA/blob/main/demo/scores/Cluster_scores.svg" width="1000" height="300">

## Caution
Caution must be taken for all re-parameterization of simulations made with `config.yaml`, snakemake does not execute the rules for which the result files are already present (unless an input file is updated by another rule), remove older files from the *results* folder when needed.

## Resolving some package issues
Some package-related issues might still arise during code excecution, however, most solutions to these issues can be found online. For example here are some issue we encountered

### Issue 1
Error message about parameter issues in snakemake file. This might be a snakefile formating issue, which can be solved by

First install [snakefmt](https://github.com/snakemake/snakefmt) into the `VASIL` enviroment
```
pip install snakefmt
```
Then, when needed, reformat snakefile

```
snakefmt VASIL
```
In case you had to interrupt snakemake run code (e.g. by Ctr + Z), you need to remove the folder `workdir/.snakemake/locks/`

```
rm -rf workdir/.snakemake/locks
```

### Issue 2
```
Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed
```

Typing the following sequence of code solves this issue [(see stackoverflow)](https://stackoverflow.com/questions/58868528/importing-the-numpy-c-extensions-failed)
```
pip uninstall -y numpy
pip uninstall -y setuptools
pip install setuptools
pip install numpy
```

### Issue 3

```
File ... from scipy.optimize import root ... .../usr/lib/liblapack.3.dylib (no such file)   
```
This is a problem with scipy that is resolved by unistalling and re-installing scipy with `pip` [saturncloud](https://saturncloud.io/blog/pip-installation-of-scipyoptimize-or-scipy-correctly/)

```
pip unistall scipy
```
```
pip install scipy

```

### Issue 4

```
AttributeError: module 'lib' has no attribute 'OpenSSL_add_all_algorithms'
```

Solution:
```
pip uninstall pyOpenSSL
pip install pyOpenSSL
```

### Issue 5 (Apple silicon)
```
Library not loaded: @rpath/liblapack.3.dylib
```

Solution:

```
pip install --upgrade --force-reinstall scikit-learn
```

### Issue 6 
If the conda installation fails, please use the following commands to install it manually:

```
conda create --name MIASA
conda activate MIASA
conda install -c conda-forge -c bioconda -c anaconda python==3.10.4 numpy==1.21.5 scipy==1.7.3 openpyxl pandas==1.4.3 matplotlib seaborn joblib regex pip scikit-learn==1.1.3  
```
Proceed above or use `pip` to install all other possible missing packages prompt by error messages


