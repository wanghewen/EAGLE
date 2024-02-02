import argparse
import traceback

import torch

from dataloaders.GNN_dataloader import GNNDataloaderInitializer
from datasets.AmazonAndGoogle.AmazonEdgeAttributed import AmazonEdgeAttributed
from datasets.AmazonAndGoogle.GoogleEdgeAttributed import GoogleEdgeAttributed
from datasets.Citations.AMinerEdgeAttributed import AMinerEdgeAttributed
from datasets.Citations.DBLPEdgeAttributed import DBLPEdgeAttributed
from datasets.Citations.OAGEdgeAttributed import OAGEdgeAttributed
from global_parameters import wrapped_partial, data_folder, max_epochs
from models.BipartiteEdge import BipartiteEdge
from train import main


def run_vanilla_experiments_edge_attributed(dataset_class, sample_size, product_category=None):
    if product_category is not None:
        dataset_class_name = dataset_class.__name__ + f"_{product_category}"
    else:
        dataset_class_name = dataset_class.__name__
    print("Current dataset:", dataset_class_name)

    use_pl = False
    svd_dim = 16
    ppr_decay_factor = 0.5
    beta = 0.5
    gamma = 1e-4
    # for svd_dim in [8, 16, 32, 64, 128, 256, 512, None]:
    # for svd_dim in [None]:
    # for beta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    # for svd_dim in [256]:
    # for gamma in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
    # for ppr_decay_factor in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    for ppr_decay_factor in [ppr_decay_factor]:
        model_parameter_dict = {"n_classes": "auto",
                                "n_layers": 2,
                                "activation": torch.nn.ReLU(),
                                "dropout": 0.5,
                                # "dropout": 1,
                                "structure": "GCN",
                                "edge_attributed": True,
                                "ppr_decay_factor": ppr_decay_factor,
                                "svd_dim": svd_dim,
                                "beta": beta,
                                "gamma": None,
                                "force_redo_svd": False,
                                "train_ratio": 1.0,
                                "classifier": "mlp"}
        if dataset_class in [DBLPEdgeAttributed]:
            if sample_size is None:
                model_parameter_dict["train_ratio"] = 0.1  # Reduce training size

        dataloader_class = wrapped_partial(GNNDataloaderInitializer, use_raw_text_attributes=False)
        model_class = BipartiteEdge
        default_parameter_dict = model_parameter_dict | {"error_analysis": False}


        print("######################### BipartiteEdgeAERTransformBefore ##########################")
        logger_name = dataset_class_name + "__BipartiteEdgeAERTransformBefore__baseline"
        update_parameter_dict = {
            "cache_folder_path": None,
            # "cache_folder_path": os.path.join(data_folder, "cache", dataset_class.__name__ + f"_{sample_size}"),
            # "force_redo_svd": True,
            "aggregator": None,
            "use_aer": True, "transform_x": True}
        model_parameter_dict = default_parameter_dict | update_parameter_dict
        main(logger_name, dataset_class, dataloader_class, model_class, model_parameter_dict, use_pl)

        # print("######################### BipartiteEdgeAERTransformBeforeSeparateEmbSum #############")
        # logger_name = dataset_class_name + "__BipartiteEdgeAERTransformBeforeSeparateEmbSum__baseline"
        # update_parameter_dict = {
        #     "cache_folder_path": None,
        #     # "cache_folder_path": os.path.join(data_folder, "cache", dataset_class.__name__+f"_{sample_size}"),
        #     # "force_redo_svd": True,
        #     "use_aer": True, "transform_x": True,
        #     "aggregator": "sum", "separate_aer_embeddings": True}
        # model_parameter_dict = default_parameter_dict | update_parameter_dict
        # main(logger_name, dataset_class, dataloader_class, model_class, model_parameter_dict, use_pl)
        #
        # print("######################### BipartiteEdgeAERTransformBeforeSeparateEmbMax #############")
        # logger_name = dataset_class_name + "__BipartiteEdgeAERTransformBeforeSeparateEmbSum__baseline"
        # update_parameter_dict = {
        #     "cache_folder_path": None,
        #     # "cache_folder_path": os.path.join(data_folder, "cache", dataset_class.__name__+f"_{sample_size}"),
        #     # "force_redo_svd": True,
        #     "use_aer": True, "transform_x": True,
        #     "aggregator": "max", "separate_aer_embeddings": True}
        # model_parameter_dict = default_parameter_dict | update_parameter_dict
        # main(logger_name, dataset_class, dataloader_class, model_class, model_parameter_dict, use_pl)
        #
        # print("######################### BipartiteEdgeAERTransformBeforeSeparateEmbConcat #############")
        # logger_name = dataset_class_name + "__BipartiteEdgeAERTransformBeforeSeparateEmbSum__baseline"
        # update_parameter_dict = {
        #     "cache_folder_path": None,
        #     # "cache_folder_path": os.path.join(data_folder, "cache", dataset_class.__name__ + f"_{sample_size}"),
        #     # "force_redo_svd": True,
        #     "use_aer": True, "transform_x": True,
        #     "aggregator": "concat", "separate_aer_embeddings": True}
        # model_parameter_dict = default_parameter_dict | update_parameter_dict
        # main(logger_name, dataset_class, dataloader_class, model_class, model_parameter_dict, use_pl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--dataset', type=str, help='dataset class name', default=None)
    args = parser.parse_args()

    if args.dataset is not None:
        dataset_classes = [eval(args.dataset)]
    else:
        dataset_classes = [DBLPEdgeAttributed, AmazonEdgeAttributed, GoogleEdgeAttributed, OAGEdgeAttributed,
                           AMinerEdgeAttributed]

    for dataset_class in dataset_classes:
        try:
            print("dataset_class", dataset_class.__name__)

            if dataset_class in [DBLPEdgeAttributed, OAGEdgeAttributed, AMinerEdgeAttributed,
                                 AmazonEdgeAttributed, GoogleEdgeAttributed]:  # Select product category if any
                if dataset_class in [AmazonEdgeAttributed]:
                    product_category = "reviews_Movies_and_TV_5"
                elif dataset_class in [GoogleEdgeAttributed]:
                    product_category = "review-Hawaii_10"
                else:
                    product_category = None
            if dataset_class in [DBLPEdgeAttributed, OAGEdgeAttributed, AMinerEdgeAttributed,
                                 AmazonEdgeAttributed, GoogleEdgeAttributed]:  # Select sample size
                for sample_size in [40000]:
                    if dataset_class in [OAGEdgeAttributed, AMinerEdgeAttributed] and sample_size is None:
                        continue  # Cannot finish for some reason

                    new_dataset_class = wrapped_partial(dataset_class, add_self_loop=False, randomize_train_test=False,
                                                        randomize_by_node=False, sample_size=sample_size,
                                                        is_bipartite=True, process_text=True, use_text_cache=True,
                                                        product_category=product_category)
                    # For citations, already split when preprocessing
                    run_vanilla_experiments_edge_attributed(new_dataset_class, sample_size, product_category=product_category)
            else:
                # For Yelpchi, process_text should be False
                new_dataset_class = wrapped_partial(dataset_class, add_self_loop=False, randomize_train_test=False,
                                                    randomize_by_node=False, sample_size=None,
                                                    is_bipartite=True, process_text=False, use_text_cache=True)
                run_vanilla_experiments_edge_attributed(new_dataset_class, None)
        except Exception as e:
            traceback.print_exc()
