## **M**ulti-**I**nput data **AS**sembly for joint **A**nalysis
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![DOI](https://zenodo.org/badge/681107366.svg)](https://doi.org/10.5281/zenodo.10124274)
### Web App available for small datasets
For manageable dataset: **less than 1000 objects/items** in total to analyze, you can use
[pythonanywhere platform: MIASA app](https://projects-raharinirina.pythonanywhere.com/miasa/miasa_input/)

Unfortunately, the above app is not feasible for larger datasets due to computational power limitations.
### Workflow installation Operating Systems
This workflow was tested on macOS Monterey Version 12.5 and CentOS Linux 7 (Core)

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
sklearn.som
#### Install Conda/Miniconda
Conda will manage the dependencies of our pipeline. Instructions can be found here:
[https://docs.conda.io/projects/conda/en/latest/user-guide/install](https://docs.conda.io/projects/conda/en/latest/user-guide/install)

Create a new environment from the given environment config in [env.yml](https://github.com/AlexiaNomena/MIASA/blob/main/env/env.yml)

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
pip install sklearn.som
```

```
pip install umap-learn
```
### Important Note
Higher versions of the above packages could be also be suitable for MIASA. Feel free to contact us in case you encounter unresolved problems.

Because of package changes and updates (clustering methods, UMAP, t-SNE, ...), the figures shown in the Manuscript might not be reproducible. However, results obtained should still be essentially the same.

### In the folder Manuscript_examples:
This folder contains all the codes that were used to produce the manuscripts results

Add environment to jupyter notebook

`conda deactivate` (only if the environment is activated)

`conda install -c anaconda ipykernel` (only if `ipykernel` is not yet installed)

`python -m ipykernel install --user --name=MIASA`

#### List of codes

`miasa_Dist.ipynb`, `miasa_Corr.ipynb`, `miasa_GRN.ipynb`: python code for using MIASA for the three dataset problems highlighted in the paper (similarity distances are Euclidean).

**NB**: the folder dataset `2mRNA_100000` for two-Gene regulatory models must be dowloaded from [(here)](https://drive.google.com/drive/folders/1n_NhI-72qFdEA_d4jrpUyMqhbd6gGkdJ?usp=sharing) and placed in the folder `Manuscript_examples/Data/` (without changing the folder names)

`miasa_NonEucl_Dist.ipynb`, `miasa_NonEucl_Corr.ipynb`, `miasa_NonEucl_GRN.ipynb`:  using MIASA for the three dataset problems highlighted in the paper (similarity distances are non-Euclidean).

`miasa_Dist_SOM.ipynb`, `miasa_Dist_SOM_MIASA.ipynb`, `miasa_Dist_NN.ipynb`, `miasa_Dist_SVM.ipynb`:  machine learning experiments using MIASA for the distribution dataset as highlighted in the paper (similarity distances are non-Euclidean).

`class_experiment.py`: python code for classification experiments when the true clusters are known and included in the data generating function which must return data in a specific format (e.g. function `generate_data_dist` in module `Methods/simulate_class_data.py`)


### Snakemake workfow
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
These variables are stored in [`config.yaml`](https://github.com/AlexiaNomena/MIASA/blob/main/config.yaml).
For more information about the YAML markup format refer to documentation: [https://yaml.org](https://yaml.org)
### Dataset Requirement
All input datasets must be pre-processed to follow the following requirements

- Format must be `.xlsx` or `.csv`
- A column `variable` must be included indicating the variable labels (see all example datasets in the folder `dataset_requirement`)

### Available Clustering Method Options
```
"Agglomerative_*"# where * is a linkage method of `sklearn.cluster.AgglomerativeClustering', 
"Kmeans", # sklearn.cluster.KMeans
"Kmedoids", # sklearn_extra.cluster.KMedoids
"Spectral", # sklearn.cluster.SpectralClustering
"GMM", # sklearn.mixture.GaussianMixture
BayesianGMM", # sklearn.mixture.BayesianGaussianMixture
"DBSCAN", # sklearn.cluster.DBSCAN
"MLPClassifer/<true labels file (.xlsx or .csv) with a column true label aligned with original dataset Xs first and then the Ys (see demo/true_labels_files.xlsx)>/<percentage train randomly chosen within the true labels>" # "/" separated, sklearn.neural_network.MLPClassifer using the parameters of manuscript results
"MLPRegressor/<true labels file (.xlsx or .csv) with a column true label aligned with original dataset Xs first and then then Ys (see demo/true_labels_files.xlsx)>/<percentage train randomly chosen within the true labels>" # sklearn.neural_network.MLPRegressor using the parameters of manuscript results
"SVM_SVC/<true labels file (.xlsx or .csv) with a column true label aligned with original dataset Xs first and then the Ys (see demo/true_labels_files.xlsx)>/<percentage train randomly chosen within the true labels>" # sklearn.svm.SVC using the parameters of manuscript results
"SOM/<a positive number  (that will be multiplied with 1/c3zeta to give the learning rate parameter)>" # sklearn.som default initialization 
```
## Execution

If your environment is not yet activated, type

```
conda activate MIASA
```
Go to the pipeline directory (where the Snakefile named [`MIASA`](https://github.com/AlexiaNomena/MIASA/blob/main/MIASA) is located) and enter the following command to execute the pipeline

```
snakemake --snakefile MIASA --configfile path/to/config.yaml -j -d path/to/workdir

```
```
**CAUTION**:  Please delete the all the files generated in the folder `plots/` 
and re-run the above code line to make sure that the plots corresponds to the results
```
With parameter `--configfile` you can give the configuration file, described above. The `-j` parameter determines the number of available CPU cores to use in the pipeline. Optionally you can provide the number of cores, e.g. `-j 4`. With parameter `-d` you can set the work directory, i.e. where the results of the pipeline are written to.

## Output
The main pipeline (`config.yaml`) creates a folder *results*, containing all (intermediate) output, with the following structure:

```
|-- results
	|-- qEE_Transformed_Dataset.xlsx/csv # standard machine readable formats of the transformed dataset via qEE-Transition (with 10 decimals numerical precision)
 	|-- miasa_results.pck	# pickled python dictionary containing the results  with essential keys: "Coords" (embedded coordinates on the rows: for X dataset starting from row 1 and ordered as in the original dataset with or without an origin, at row index M+1 and the rest is the Y dataset ordered as in the original dataset) and "Class_pred" (the predicted cluster indexes for the rows of "Coords")                                
|-- plots	
	|-- UMAP_One_Panel.pdf/.svg # UMAP projection of the results
	|-- UMAP_Separate_Panels.pdf/.svg # UMAP projection separate predicted panels
	|-- tSNE_One_Panel.pdf/.svg # t-SNE projection of the results 
	|-- tSNE_Separate_Panels.pdf/.svg # t-SNE projection of the results, separate predicted panels
|-- scores
	|-- scored_miasa_results.pck # pickled python dictionary containing the results   including cluster score vectors  in the keys "silhouette", "elbow", "distortion", corresponding to the array of number of clusters saved as key "list_num".
	|-- Cluster_scores.pdf/.svg # cluster score plots (Elbow, Distortion, Silhouette)	
```

## Demo
Demo datasets are provided in the repository folder [`demo`](https://github.com/AlexiaNomena/MIASA/tree/main/demo)

If your environment is not yet activated, type

```
conda activate MIASA
```

To run the pipeline go into the repository where the snakefile [`MIASA`](https://github.com/AlexiaNomena/MIASA/blob/main/MIASA) is located and run

```
snakemake --snakefile MIASA --configfile demo/demo_config.yaml -j -d demo

```
```
**CAUTION**:  Please delete the all the files generated in the folder `plots/` 
and re-run the above code line to make sure that the plots corresponds to the results
```

### Expected Runtime for demo
Less than 5 min

Deactivate the environment to exit the pipeline
```
conda deactivate
```

The result folder is created in the [`demo`](./demo) folder where you find the output files, as described above. 

The projection of all predicted cluster on the same pannel indicating convex hull of predicted clusters (some outliers excluded) and labels of true cluster members (if true labels are given as a parameter in config file)
<img src="https://github.com/AlexiaNomena/MIASA/blob/main/demo/plots/UMAP_One_Panel.svg" width="500" height="500">

The projection of all predicted cluster separated pannels:

- With convex hull of prediction (some outliers excluded)
<img src="https://github.com/AlexiaNomena/MIASA/blob/main/demo/plots/UMAP_Separate_Panels_p1.svg" width="500" height="500">

- Without convex hull of prediction
<img src="https://github.com/AlexiaNomena/MIASA/blob/main/demo/plots/UMAP_Separate_Panels_p2.svg" width="500" height="500">

Cluster evaluation [(Wikipedia)](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set)
<img src="https://github.com/AlexiaNomena/MIASA/blob/main/demo/scores/Cluster_scores.svg" width="1000" height="300">

## Caution
Caution must be taken for all re-parameterization of simulations made with `config.yaml`, snakemake does not execute the rules for which the result files are already present (unless an input file is updated by another rule), remove older files from the *results* folder when needed.

## Resolving some package issues
Some package-related issues might still arise during code excecution, however, most solutions to these issues can be found online. For example here are some issue we encountered

### Issue 1
Error message about parameter issues in snakemake file. This might be a snakefile formating issue, which can be solved by

First install [snakefmt](https://github.com/snakemake/snakefmt) into the `MIASA` enviroment
```
pip install snakefmt
```
Then, when needed, reformat snakefile

```
snakefmt MIASA
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


