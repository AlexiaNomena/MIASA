
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
conda activate MIASA
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
 	|-- miasa_results.pck	# pickled python dictionary containing the results of miasa
	|-- UMAP_One_Pannel.pdf/.svg # UMAP projection of the results of miasa
	|-- UMAP_Separate_Pannel.pdf/.svg # UMAP projection of the results of miasa, separate predicted pannels
	|-- tSNE_One_Pannel.pdf/.svg # t-SNE projection of the results of miasa
	|-- tSNE_Separate_Pannel.pdf/.svg # t-SNE projection of the results of miasa,  separate predicted pannels
```

## Demo
Demo datasets are provided in the repository folder [`demo`](https://github.com/KleistLab/VASIL/tree/main/demo)

If your environment is not yet activated, type

```
conda activate MIASA
```

To run the pipeline go into the repository where the snakefile [`MIASA`](https://github.com/AlexiaNomena/MIASA/blob/main/MIASA) is located and run

```
snakemake --snakefile VASIL --configfile demo/demo_config.yaml -j -d demo

```

### Expected Runtime for demo
Less than 1 min

Deactivate the environment to exit the pipeline
```
conda deactivate
```

