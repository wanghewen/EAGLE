from datasets.base.EdgeAttributed import EdgeAttributed
from global_parameters import OAGCountVectorizerDataFolder, OAGBipartiteTextDataFolder


class OAGEdgeAttributed(EdgeAttributed):
    #  nodes,  edges ->  nodes,  edges
    def __init__(self, num_classes=10, subset=None, no_node_features=False,
                 initialized_instance=None, add_self_loop=True,
                 randomize_train_test=False, randomize_by_node=False, sample_size=None,
                 sample_way="bfs", reduced_dim=None, is_bipartite=False, process_text=True,
                 use_text_cache=True, **kwargs):
        if is_bipartite:
            data_folder = OAGBipartiteTextDataFolder
        else:
            data_folder = OAGCountVectorizerDataFolder
        super().__init__(name="OAGEdgeAttributed", data_folder=data_folder,
                         subset=subset, no_node_features=no_node_features,
                         initialized_instance=initialized_instance, add_self_loop=add_self_loop,
                         randomize_train_test=randomize_train_test,
                         randomize_by_node=randomize_by_node, num_classes=num_classes,
                         sample_way=sample_way, sample_size=sample_size, reduced_dim=reduced_dim,
                         is_bipartite=is_bipartite, process_text=process_text, use_text_cache=use_text_cache, **kwargs)