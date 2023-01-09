### MIASA framework
**M**etric-based **I**dentification of **S**imilarity and **A**ssociation patterns in multivariate datasets

### Theory 
TBA

### Software Requirement
python version: 3.10 or +

pakages: [scikit-learn](https://scikit-learn.org/stable/), [umap](https://umap-learn.readthedocs.io/en/latest/), [numpy](https://numpy.org/), [scipy](https://scipy.org/), ... (find missing ones from the code run error logs)

### Description
`miasa_guided.py`: python code for a step by step guidance through the MIASA framework.

`miasa_guided.ipynb`: jupyter notebook for a step by step guidance through the MIASA framework.

`miasa_blackbox.py`: python code for using MIASA as a blackbox.

`miasa_blackbox.ipynb`: jupyter notebook for using MIASA as a blackbox.

`class_experiment.py`: python code for classification experiments when the true clusters are known and included in the data generating function which must return data in a specific format (e.g. function `generate_data_dist` in module `Methods/simulate_class_data.py`)

`class_experiment_extended.py`: same as above but extended by including more options for clustering algorithm and distance models 


### License
The following modules are under [GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.html): `Methods/Core/CosLM.py` ([source](https://github.com/AlexiaNomena/PSD_cosine_law_matrix)), `Methods/Core/qEmbedding.py`

Everything else besides **Software Requirements**: [Open Source Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)