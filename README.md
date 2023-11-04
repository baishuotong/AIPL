# AIPL
Here is the relevant open-source code for the article titled “Improving Issue-PR Link Prediction via Knowledge-aware Heterogeneous Graph”
## Introduction
In this work, we designe an approach, named AIPL, capable of predicting Issue-PR links on GitHub. It leverages the heterogeneous graph to model multi-type GitHub data and employ the
metapath-based technology to incorporate crucial information transmitting among multi-type data. When given a pair of an issue and a PR, AIPL can
suggest whether there could be a link.
## Environment
AIPL is implemented by PyTorch over a server equipped with NVIDIA GTX 1060 GPU (3G memory).
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
The code of our baselines, including iLinker, A-M, random walk, R-GCN and HAN.<br />
- __AIPL__
The code of AIPL, please read the following introduction for a better understanding. <br />
## Code Functions
The relevant codes of our method include building heterogeneous graph, constrcuting metapath and training graph-based model. <br />
```build_graph.py``` The code related to the heterogeneous graph building. User can construct the heterogeneous graph based on our datasets by running build_graph.py. <br />
```construct_metapath.py``` The code related to the metapath construction. User can establish all metapath instances by running construct_metapath.py. <br />
```AIPL_main.py``` The code related to the model training. User can train and evaluate AIPL by running magnn_main.py_. <br />
## Example Presentation
1. Example 1
![image](https://github.com/baishuotong/AIPL/assets/38210633/74e60014-6fd4-485f-babb-e86bdc96bfd8)
2. Example 2
 ![image](https://github.com/baishuotong/AIPL/assets/38210633/81877e7d-f814-4c02-a10d-b89333c29713)
3. Example 3
![image](https://github.com/baishuotong/AIPL/assets/38210633/39fce051-8c9d-432c-8ba0-fbb25fd3e2c9)

## Copyright
All copyright of the tool is owned by the author of the paper.
