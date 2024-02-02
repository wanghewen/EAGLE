import CommonModules as CM
import os

import dgl
import numpy as np
import pandas as pd
import scipy.sparse
import torch

from datasets.base.EdgeAttributed import EdgeAttributed


# Refer to https://docs.dgl.ai/en/latest/new-tutorial/6_load_data.html
# And https://github.com/minhduc0711/gcn/blob/2d09dd2f5c5400e4b53f74e2b35a5a612904ae48/src/data/datasets.py
from global_parameters import AmazonReviewCountVectorizerDataFolder, AmazonBipartiteTextDataFolder, \
    GoogleLocalBipartiteTextDataFolder


class GoogleEdgeAttributed(EdgeAttributed):
    #  nodes,  edges ->  nodes,  edges
    def __init__(self, subset=None, no_node_features=False, initialized_instance=None, add_self_loop=True,
                 randomize_train_test=False, randomize_by_node=False, sample_size=None,
                 sample_way="bfs", reduced_dim=None, is_bipartite=False, process_text=True,
                 use_text_cache=True, product_category=None, **kwargs):
        # AmazonEdgeToNodeDataFolder = "/data/PublicGraph/AmazonReview/data_sentence_transformer"
        if is_bipartite:
            data_folder = os.path.join(GoogleLocalBipartiteTextDataFolder, product_category)
        else:
            raise NotImplementedError
        super().__init__(name="GoogleEdgeAttributed", data_folder=data_folder,
                         subset=subset, no_node_features=no_node_features,
                         initialized_instance=initialized_instance, add_self_loop=add_self_loop,
                         randomize_train_test=randomize_train_test, randomize_by_node=randomize_by_node,
                         sample_way=sample_way, sample_size=sample_size, reduced_dim=reduced_dim,
                         is_bipartite=is_bipartite, process_text=process_text, use_text_cache=use_text_cache,
                         **kwargs)


if __name__ == "__main__":
    dataset = GoogleEdgeAttributed(subset="train", is_bipartite=True, process_text=True, use_text_cache=True,
                                   product_category="reviews_Movies_and_TV_5")
    graph = dataset[0]
    # graph = dgl.to_homogeneous(graph, ["user", "transaction"])
    print(graph)