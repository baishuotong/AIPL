# AIPL
Here is the relevant open-source code for the article titled “Improving Issue-PR Link Prediction via Knowledge-aware Heterogeneous Graph”
## Introduction
In this work, we designe an approach, named AIPL, capable of predicting Issue-PR links on GitHub. It leverages the heterogeneous graph to model multi-type GitHub data and employ the
metapath-based technology to incorporate crucial information transmitting among multi-type data. When given a pair of an issue and a PR, AIPL can
suggest whether there could be a link.
## Environment
AIPL is implemented by PyTorch over a server equipped with NVIDIA GTX 1060 GPU (3G memory).
## Dependencies
- python 3.7.3
- PyTorch 1.13.1
- NumPy 1.21.5
- Pandas 1.3.4
- scikit-learn 1.0.2
- scipy 1.7.3
- DGL 0.6.1
- NetworkX 2.6.3
## File Introduction
### Dataset
We release our annotated dataset in this file dir.
#### facebook/react & vuejs/vue
Annotated dataset based on repositories facebook_react and vuejs/vue <br />
- __Index__  Information of nodes and edges on heterogeneous graph <br />
- __Features__ Embeddings of nodes <br />
- __Training set& Test set__ Annotated dataset <br />
- __Metapath__ Information of metapaths <br />
- __adjM.npz__ The adjacency matrix of heterogeneous graph <br />
### Code
- __baseline__
The code of our baselines, including iLinker, A-M, random walk, metapath2vec, R-GCN, GTN, Simple-HGN, HGT, HAN, Sehgnn, and MECCH.<br />
- __AIPL__
The code of AIPL, please read the following introduction for a better understanding. <br />
## Code Functions
The relevant codes of our method include building heterogeneous graph, constrcuting metapath and training graph-based model. <br />
The first step is to run ```build_graph.py``` . The second step is to run ```construct_metapath.py```. The third strp is to run ```AIPL_main.py``` <br />
The detailed explanations are as follows:<br /> 
```build_graph.py```  <br />
The code snippet constructs a heterogeneous graph and generates node features for users, repositories (repos), issues, and pull requests (PRs).<br />
It loads data related to various relationships like user-repo, user-issue, user-PR, repo-repo, repo-issue, repo-PR, issue-issue, issue-PR, and PR-PR from corresponding directories and creates an adjacency matrix (adjM).<br /> 
Additionally, it extracts feature vectors such as title vectors from CSV files to create features for repos, issues, and PRs. <br />
The code then saves the adjacency matrix and node features in numpy arrays for further analysis. <br />
```construct_metapath.py``` <br />
The code first loads data from various edge and index files, including user-repo, user-issue, user-pr, repo-repo, repo-issue, repo-pr, issue-issue, issue-pr, and pr-pr.  <br />
It then loads adjacency matrices and organizes them into lists based on different node types such as users, repositories, issues, and prs.   <br />
Next, the code generates expected metapaths based on predefined patterns. These metapaths are then mapped to corresponding indices and stored in pickle files, numpy arrays, and adjacency lists for further analysis and processing. <br />
```AIPL_main.py```  <br />
The code is related to the model training and model inferences. User can train and evaluate AIPL by running ```AIPL_main.py```. <br />
The script handles data loading, model setup, training with early stopping, and evaluation using metrics like accuracy, precision, recall, and F1-score. <br />
Specifically, the functions of loading data and batching are called using the files ```data.py```, ```preprocess.py```, and ```tools.py``` in the 'utils' folder.  <br />
Regarding the construction of the AIPL model, it includes intra-metapath aggregation, inter-metapath aggregation, and attention mechanism.  <br />
These codes are presented in the ```base_magnn.py``` and ```magnn_lp.py``` under the 'magnn_model' directory, directly called by 'AIPL_main'." <br />
## Example Presentation
1. Example 1
![image](https://github.com/baishuotong/AIPL/assets/38210633/74e60014-6fd4-485f-babb-e86bdc96bfd8)
2. Example 2
 ![image](https://github.com/baishuotong/AIPL/assets/38210633/81877e7d-f814-4c02-a10d-b89333c29713)
3. Example 3
![image](https://github.com/baishuotong/AIPL/assets/38210633/39fce051-8c9d-432c-8ba0-fbb25fd3e2c9)

## Copyright
All copyright of the tool is owned by the author of the paper.
