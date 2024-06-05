import os
from conllu import parse as parse_conllu

import networkx as nx
import torch
import numpy as np
from matplotlib import pyplot as plt
import random


class Loader():
    def __init__(self, set, batch_size, skip_long=True):
        self.set = set
        self.batch_size = batch_size
        self.skip_long = skip_long

        self.path = os.path.join("ud-en-ewt",f"{self.set}.conllu")
        f = open(self.path, 'r', encoding='utf-8')
        self.parsed_file = parse_conllu(f.read())

    def generate(self):
        start_batch=True
        batch_num = 0
        random.shuffle(self.parsed_file)

        for sentence in self.parsed_file:
            if start_batch:
                start_batch = False
                batch_txt = []
                batch_G = []
                batch_pdist = []
            
            pdist, G = self._get_labels(sentence)

            token_list = [token["form"] for token in sentence if type(token["id"])==int]
            txt = " ".join(token_list)

            if len(token_list) != len(pdist):
                continue
            elif len(token_list)==1:
                continue
            elif (".com" in txt) or ("@" in txt):
                continue
            elif (len(token_list) > 20) and self.skip_long:
                continue

            batch_txt.append(txt)
            batch_pdist.append(pdist)
            batch_G.append(G)

            if len(batch_txt) == self.batch_size:
                start_batch = True
                batch_num += 1
                yield batch_txt, batch_pdist, batch_G

    def _construct_graph(self,sentence):
        G_directed = nx.DiGraph()

        for token in sentence:
            if (type(token["head"]) != int) or (type(token["id"]) != int):
                continue
            if token["head"] == 0:
                G_directed.add_node(token['id'], root=True)
            elif token["head"] not in [0, None]:
                G_directed.add_node(token['id'], root=False)

        for token in sentence:
            if (type(token["head"]) != int) or (type(token["id"]) != int):
                continue
            elif (token['head'] not in [0, None]):
                rel_type = token["deprel"].split(":")[0]
                G_directed.add_edge(token['head'], token['id'], rel_type=rel_type)

        return G_directed
    
    def _calc_distance_from_G(self,G):
        G = G.to_undirected()
        shortest_path_lengths = nx.floyd_warshall(G)
        num_nodes = len(G.nodes())
        distance_matrix = [[0] * num_nodes for _ in range(num_nodes)]

        for source, distances in shortest_path_lengths.items():
            for target, distance in distances.items():
                distance_matrix[source-1][target-1] = distance
        
        distance_matrix = torch.tensor(distance_matrix)
        return distance_matrix

    def _get_labels(self,sentence):
        G_directed = self._construct_graph(sentence)
        pdist = self._calc_distance_from_G(G_directed)
        return pdist, G_directed
