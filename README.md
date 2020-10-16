# Motif-informed embeddings for disease pathway prediction

## Motifs detection
FANMOD: python-igraph package (brew install igraph) => estimation of all graphlets type in the network  

Graphlets type can be found in this paper: https://pubmed.ncbi.nlm.nih.gov/17237089/  

![Graphlets](./figures/btl301f1.jpeg)

## Graph embedding
node2vec: python3 implementation (pip3 install node2vec)
GCN & graphSAGE: stellar-graph (pip3 install stellargraph)

## Related work
[Large-Scale Analysis of Disease Pathways in the Human Interactome](http://snap.stanford.edu/pathways/) 

- [paper](http://psb.stanford.edu/psb-online/proceedings/psb18/agrawal.pdf) 
- [code](https://github.com/mims-harvard/pathways)
- Further analysis in `data`

[Graph Convolutional Networks to explore Drug and
Disease Relationships in Biological Networks](http://snap.stanford.edu/class/cs224w-2017/projects/cs224w-41-final.pdf)

Exploratory Data Analysis Inspiration:

- http://snap.stanford.edu/deepnetbio-ismb/ipynb/Human+Disease+Network.html
- http://snap.stanford.edu/deepnetbio-ismb/ipynb/Graph+Convolutional+Prediction+of+Protein+Interactions+in+Yeast.html
- http://snap.stanford.edu/deepnetbio-ismb/slides/deepnetbio-part1-embeddings.pdf
- http://snap.stanford.edu/deepnetbio-ismb/slides/deepnetbio-part2-gcn.pdf
