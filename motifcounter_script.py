import numpy as np
import scipy.sparse as sp
import networkx as nx
import pandas as pd
from texttable import Texttable
from tqdm import tqdm
from networkx.generators.atlas import *

def load_graph(graph_path):
    """
    Reading an egde list csv as an NX graph object.
    :param graph_path: Path to the edgelist.
    :return graph: Networkx Object.
    """
    graph = nx.from_edgelist(pd.read_csv(graph_path).values.tolist())
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph

graph_path = './data/bio-pathways-network.csv'
G = load_graph(graph_path)

df = pd.read_csv('./data/binary-classes.csv')
disease_related_protein = df[df['diseased'] == 1]['Unnamed: 0'].tolist()
G1 = G.subgraph(disease_related_protein)

        
class MotifCounterMachine1(object):
    """
    Connected motif orbital role counter.
    """
    def __init__(self, graph, graphlet_size, output):
        """
        Creating an orbital role counter machine.
        :param graph: NetworkX graph.
        :param args: Arguments object.
        """
        self.graph = graph
        self.output = output
        self.graphlet_size = graphlet_size

    def create_edge_subsets(self):
        """
        Enumerating connected subgraphs with size 2 up to the graphlet size.
        """
        print("\nEnumerating subgraphs.\n")
        self.edge_subsets = dict()
        subsets = [[e[0],e[1]] for e in self.graph.edges()]
        self.edge_subsets[2] = subsets

        if self.graphlet_size > 2:
            for i in range(3, self.graphlet_size+1):
                print("Enumerating graphlets with size: " +str(i) + ".")
                unique_subsets = dict()
                for subset in tqdm(subsets):
                    for node in subset:
                        for neb in self.graph.neighbors(node):
                            new_subset = subset+[neb]
                            if len(set(new_subset)) == i:
                                new_subset.sort()
                                unique_subsets[tuple(new_subset)] = 1
                subsets = np.asarray(list(unique_subsets.keys()), dtype=np.int32)
                self.edge_subsets[i] = subsets
            for i in range(2, self.graphlet_size):
                del self.edge_subsets[i]

    def enumerate_graphs(self):
        """
        Creating a hash table of the benchmark motifs.
        """
        graphs = graph_atlas_g()
        self.interesting_graphs = []
        for graph in graphs:
            if graph.number_of_nodes() == self.graphlet_size:
                if nx.is_connected(graph):
                    self.interesting_graphs.append(graph)

    def enumerate_categories(self):
        """
        Creating a hash table of benchmark orbital roles.
        """
        main_index = 0
        self.orbital_position = dict()
        self.orbital_position[self.graphlet_size] = dict()
        for index, graph in enumerate(self.interesting_graphs):
            self.orbital_position[self.graphlet_size][index] = dict()
            degrees = list(set([graph.degree(node) for node in graph.nodes()]))
            for degree in degrees:
                self.orbital_position[self.graphlet_size][index][degree] = main_index
                main_index += 1
        self.unique_motif_count = main_index

    def setup_features(self):
        """
        Counting all the orbital roles.
        """
        print("\nCounting orbital roles.\n")
        self.features = {node: {i:0 for i in range(self.unique_motif_count)} for node in self.graph.nodes()}
        for size, node_lists in self.edge_subsets.items():
            for nodes in tqdm(node_lists):
                sub_gr = self.graph.subgraph(nodes)
                for index, graph in enumerate(self.interesting_graphs):
                    if nx.is_isomorphic(sub_gr, graph):
                        for node in sub_gr.nodes():
                            self.features[node][self.orbital_position[size][index][sub_gr.degree(node)]] += 1
                        break

    def create_tabular_motifs(self):
        """
        Creating a table with the orbital role features.
        """
        print("Saving the dataset.")
        self.binned_features = {node: [] for node in self.graph.nodes()}
        self.motifs = [[n]+[self.features[n][i] for i in  range(self.unique_motif_count)] for n in self.graph.nodes()]
        self.motifs = pd.DataFrame(self.motifs)
        self.motifs.columns = ["id"] + ["role_"+str(index) for index in range(self.unique_motif_count)]
        self.motifs.to_csv(self.output, index=None)

    def extract_features(self):
        """
        Executing steps for feature extraction.
        """
        self.create_edge_subsets()
        print("edge subsets created")
        self.enumerate_graphs()
        print("graphs enumerated")
        self.enumerate_categories()
        print("categories enumerated")
        self.setup_features()
        print("features set up")
        self.create_tabular_motifs()
        print("tabular motifs created")

model1 = MotifCounterMachine1(G1, 4, 'result4_bis.csv')
model1.extract_features()
