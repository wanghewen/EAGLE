import CommonModules as CM
import os
import pandas as pd

from datasets.base.EdgeToNode import EdgeToNode


# Refer to https://docs.dgl.ai/en/latest/new-tutorial/6_load_data.html
# And https://github.com/minhduc0711/gcn/blob/2d09dd2f5c5400e4b53f74e2b35a5a612904ae48/src/data/datasets.py
from global_parameters import AmazonReviewCountVectorizerDataFolder


class AmazonEdgeToNode(EdgeToNode):
    #  nodes,  edges ->  nodes,  edges
    def __init__(self, subset=None, no_node_features=False, homo=True, aggregate=False, initialized_instance=None,
                 randomize_train_test=False, bidirectional=True, randomize_by_node=False, reduced_dim=None, **kwargs):
        # AmazonEdgeToNodeDataFolder = "/data/PublicGraph/AmazonReview/data_sentence_transformer"
        data_folder = AmazonReviewCountVectorizerDataFolder
        super().__init__(name="AmazonEdgeToNode", data_folder=data_folder, subset=subset,
                         no_node_features=no_node_features, homo=homo, aggregate=aggregate,
                         initialized_instance=initialized_instance, randomize_train_test=randomize_train_test,
                         bidirectional=bidirectional, randomize_by_node=randomize_by_node,
                         reduced_dim=reduced_dim, **kwargs)




if __name__ == "__main__":
    dataset = AmazonEdgeToNode(subset="train")
    graph = dataset[0]
    # graph = dgl.to_homogeneous(graph, ["user", "transaction"])
    print(graph)