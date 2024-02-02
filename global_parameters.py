import os
import uuid
from functools import partial, update_wrapper

import numpy as np
import scipy.sparse

import inspect
import torch
os.environ["MKL_INTERFACE_LAYER"] = "ILP64"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import datetime
import pytorch_lightning as pl
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
# else:
#     device = torch.device("cpu")
torch.set_float32_matmul_precision("highest")
pl.seed_everything(12)
log_file = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + ".log"
import CommonModules as CM

def convert_matrix_to_numpy(m):
    if isinstance(m, scipy.sparse.spmatrix):
        m = m.A
    elif isinstance(m, np.matrix):
        m = m.A
    elif isinstance(m, torch.Tensor):
        m = m.cpu().numpy()
    return m


def wrapped_partial(func, *args, **kwargs):
    if isinstance(func, partial):
        for kwarg in kwargs:
            func.keywords[kwarg] = kwargs[kwarg]
        return func
    else:
        partial_func = partial(func, *args, **kwargs)
        update_wrapper(partial_func, func)
        # Set class function to partial_func
        for method_name in dir(func):
            if callable(getattr(func, method_name)) and not method_name.startswith("__"):
                setattr(partial_func, method_name, getattr(func, method_name))
        return partial_func


def print_result_for_pl(trainer: pl.Trainer, checkpoint_callback, train_dataloader, val_dataloader, test_dataloader,
                        log_file=log_file):
    last_checkpoint = checkpoint_callback.last_model_path
    print(last_checkpoint)
    if train_dataloader:
        print("train_dataloader, last_checkpoint", trainer.test(dataloaders=train_dataloader, ckpt_path=last_checkpoint,
                                                                verbose=False), flush=True)
    if val_dataloader:
        print("val_dataloader, last_checkpoint", trainer.test(dataloaders=val_dataloader, ckpt_path=last_checkpoint,
                                                              verbose=False), flush=True)
    if test_dataloader:
        print("test_dataloader, last_checkpoint", trainer.test(dataloaders=test_dataloader, ckpt_path=last_checkpoint,
                                                               verbose=False), flush=True)
    print(checkpoint_callback.best_model_path)
    best_model_path = checkpoint_callback.best_model_path
    print(best_model_path)
    # trainer.validate(gcn_baseline, val_dataloaders=val_dataloader)
    # trainer.validate(dataloaders=[val_dataloader, test_dataloader], ckpt_path=best_model_path)
    train_result = {}
    if train_dataloader:
        train_result = trainer.test(dataloaders=train_dataloader, ckpt_path=best_model_path,
                                    verbose=False)[0]
    if val_dataloader:
        val_result = trainer.test(dataloaders=val_dataloader, ckpt_path=best_model_path,
                                  verbose=False)[0]
    if test_dataloader:
        start_testing_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
        test_result = trainer.test(dataloaders=test_dataloader, ckpt_path=best_model_path,
                                   verbose=False)[0]
        end_testing_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
        print(f"Testing time elapsed: {end_testing_time - start_testing_time} sec")
    if train_dataloader:
        print("train_dataloader, best_model", train_result)
    if val_dataloader:
        print("val_dataloader, best_model", val_result)
    if test_dataloader:
        print("test_dataloader, best_model", test_result)
        print(f"{best_model_path},{test_result['test/accuracy_epoch']},"
              f"{test_result['test/ap_epoch']},{test_result['test/topk_precision_epoch']},"
              f"{test_result['test/macro_f1_epoch']},{test_result['test/f1_anomaly_epoch']},"
              f"{test_result['test/auc_epoch']},{test_result.get('test/best_anomaly_f1_epoch', test_result['test/f1_anomaly_epoch'])},"
              f"{test_result.get('test/best_macro_f1_epoch', test_result['test/macro_f1_epoch'])},{test_result['test/loss_epoch']},"
              f"{train_result.get('test/loss_epoch', None)}", flush=True)
        with open(log_file, "a+") as fd:
            print(f"{best_model_path},{test_result['test/accuracy_epoch']},"
                  f"{test_result['test/ap_epoch']},"
                  f"{test_result['test/topk_precision_epoch']},"
                  f"{test_result['test/macro_f1_epoch']}," 
                  f"{test_result['test/f1_anomaly_epoch']},"
                  f"{test_result['test/auc_epoch']},"
                  f"{test_result.get('test/best_anomaly_f1_epoch', test_result['test/f1_anomaly_epoch'])},"
                  f"{test_result.get('test/best_macro_f1_epoch', test_result['test/macro_f1_epoch'])},"
                  f"{test_result['test/loss_epoch']},"
                  f"{train_result.get('test/loss_epoch', None)}", file=fd, flush=True)
        return test_result.get('test/best_anomaly_f1_epoch', test_result['test/f1_anomaly_epoch']), \
            test_result['test/auc_epoch'], test_result['test/ap_epoch'], \
            test_result.get('test/best_macro_f1_epoch', test_result['test/macro_f1_epoch'])
    else:
        return None, None, None, None, None, None


def find_gpus(num_of_cards_needed=4, model_parameter_dict={}):
    """
    Find the GPU which uses least memory. Should set CUDA_VISIBLE_DEVICES such that it uses all GPUs.

    :param num_of_cards_needed:
    :return:
    """
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    if (torch.cuda.is_available() and not (model_parameter_dict.get("force_cpu", False)))\
            and (os.environ["CUDA_VISIBLE_DEVICES"] not in ["-1", ""]):
        tmp_file_name = f'.tmp_free_gpus_{uuid.uuid4()}'
        os.system(f'nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >~/{tmp_file_name}')
        # If there is no ~ in the path, return the path unchanged
        with open(os.path.expanduser(f'~/{tmp_file_name}'), 'r') as lines_txt:
            frees = lines_txt.readlines()
            idx_freeMemory_pair = [(idx, int(x.split()[2]))
                                   for idx, x in enumerate(frees)]
        os.remove(os.path.expanduser(f'~/{tmp_file_name}'))
        idx_freeMemory_pair.sort(reverse=True)
        idx_freeMemory_pair.sort(key=lambda my_tuple: my_tuple[1], reverse=True)
        usingGPUs = [idx_memory_pair[0] for idx_memory_pair in
                     idx_freeMemory_pair[:num_of_cards_needed]]
        # usingGPUs = ','.join(usingGPUs)
        print('using GPUs:', end=' ')
        for pair in idx_freeMemory_pair[:num_of_cards_needed]:
            print(f'{pair[0]} {pair[1] / 1024:.1f}GB')
        accelerator = "gpu"
    else:
        usingGPUs = None
        accelerator = "cpu"
    return usingGPUs, accelerator


# ficro_folder = "./rur_data" # Temporal
# # ficro_folder = "/data/Sanction/Firco/backup" # No Temporal
# firco_folder = ficro_folder
data_folder = "./data/"
LANGUAGE_EMBEDDING_CACHE_DIR = os.path.join(data_folder, "LanguageEmbeddings/distilbert")
YelpchiLineGraphDataFolder = os.path.join(data_folder, "Yelpchi/rur_data")
YelpchiEdgeToNodeDataFolder = os.path.join(data_folder, "Yelpchi/edge_to_node_data")
TransactionEdgeToNodeDataFolder = os.path.join(data_folder, 'UDS_data/24k_1239k_onecomponent_remove_duplicate')
FircoEdgeToNodeDataFolder = os.path.join(data_folder, 'Firco')
BitcoinGraphDataFolder = "/data/Bitcoin/bitcoin_bigquery_transasctions/2016_biweek/output_graphs"
BitcoinHeterogeneousGraphDataFolder = os.path.join(BitcoinGraphDataFolder, "hetero")
BitcoinHomogeneousGraphDataFolder = os.path.join(BitcoinGraphDataFolder, "homo")
BitcoinHeterogeneousWithnoneGraphDataFolder = os.path.join(BitcoinGraphDataFolder, "hetero_withnone")
BitcoinHomogeneousWithnoneGraphDataFolder = os.path.join(BitcoinGraphDataFolder, "homo_withnone")
# os.chdir("/home/hewwang/autoie/Sanction/graph/Yelpchi")
AmazonReviewCountVectorizerDataFolder = os.path.join(data_folder, "AmazonReview/data_pos_ratio=0.3/data_count_vectorizer")
AmazonBipartiteTextDataFolder = os.path.join(data_folder, "AmazonReview/text_split_by_edge")
GoogleLocalBipartiteTextDataFolder = os.path.join(data_folder, "GoogleLocal/text_split_by_edge")
AMinerCountVectorizerDataFolder = os.path.join(data_folder, "AMiner/data_count_vectorizer_split_by_node")
AMinerBipartiteTextDataFolder = os.path.join(data_folder, "AMiner/text_split_by_edge")
DBLPCountVectorizerDataFolder = os.path.join(data_folder, "DBLP/data_count_vectorizer_split_by_node")
DBLPBipartiteTextDataFolder = os.path.join(data_folder, "DBLP/text_split_by_edge")
MOOCCountVectorizerDataFolder = os.path.join(data_folder, "MOOC/data_count_vectorizer_split_by_node")
RedditCountVectorizerDataFolder = os.path.join(data_folder, "Reddit/data_count_vectorizer_split_by_node")
OAGCountVectorizerDataFolder = os.path.join(data_folder, "OAG2.1/data_count_vectorizer_split_by_node")
OAGBipartiteTextDataFolder = os.path.join(data_folder, "OAG2.1/text_split_by_edge")
MINDCountVectorizerDataFolder = os.path.join(data_folder, "MIND/data_count_vectorizer_split_by_node")
MINDBipartiteCountVectorizerDataFolder = os.path.join(data_folder, "MIND/bipartite_data_count_vectorizer_split_by_node")

MAX_SEQ_LEN = 512
INPUT_EMBEDDING_DIMENSION = 128

# max_epochs = 1000
max_epochs = 300
# max_epochs = 200
# max_epochs = 10  # Used for large graphs
learning_rate = 0.001
# learning_rate = 0.01  # Used for large graphs
# df = pd.read_csv(os.path.join(ficro_folder, 'Yelpchi_train.csv'),
#                 usecols=["RECEIVER_id", "SENDER_id"], dtype=str)
print("max_epochs2:", max_epochs)
log_directory = "logs_20231102"
