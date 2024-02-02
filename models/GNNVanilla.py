import dgl
import scipy.linalg
import scipy.sparse
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, f1_score, \
    average_precision_score
import scipy.sparse as sp
import torch
import numpy as np

def dot_product(x, y, dense=False, **kwargs):
    if x.dtype == "float64":
        x = x.astype("float32", copy=False)
    if y.dtype == "float64":
        y = y.astype("float32", copy=False)
    # result = dot_product_mkl(x, y, cast=True, dense=dense)
    # return result
    z = x @ y
    return z.A if dense and scipy.sparse.issparse(z) else z


def _torch_sparse_tensor_to_sparse_mx(sparse_tensor: torch.sparse.FloatTensor):
    CoalesceTensor = sparse_tensor.cpu().coalesce()
    TensorIndices = CoalesceTensor.indices().numpy()
    TensorValues = CoalesceTensor.values().numpy()
    out = sp.coo_matrix((TensorValues, (TensorIndices[0], TensorIndices[1])), shape=sparse_tensor.shape,
                        dtype=TensorValues.dtype)
    return out


def graph_to_homogeneous(graph: dgl.DGLGraph):
    """
    Convert a bipartite graph to homogeneous graph, if possible.
    :param graph:
    :return:
    """
    if graph.is_unibipartite:
        print("handling bipartite graph…")
        ndata = [key for key in graph.ndata.keys() if not key.startswith("_")]
        edata = [key for key in graph.edata.keys() if not key.startswith("_")]
        graph = dgl.to_homogeneous(graph, ndata=ndata, edata=edata)
    return graph


from models.Base import BaseModel

# USED_CLASS = GNN
# USED_CLASS = GNNEdgeAttributed
USED_CLASS = BaseModel


def evaluate(labels, scores, multilabel=False):
    #     scores = scores.cpu()
    #     labels = labels.cpu()
    if multilabel:
        label_column_mask = np.array(labels.sum(0) > 1).reshape(-1)  # Exclude class with too few samples
        labels = labels[:, label_column_mask]
        scores = scores[:, label_column_mask]
        average_precision = average_precision_score(labels, scores, average="macro")
        roc_auc = roc_auc_score(labels, scores, average="macro")
        macro_f1 = f1_score(labels, scores > 0.5, average="macro")
        micro_f1 = f1_score(labels, scores > 0.5, average="micro")
        return None, roc_auc, average_precision, macro_f1, micro_f1, int(labels.sum())
    else:
        predictions_test = np.array(scores)
        precision, recall, thresholds = precision_recall_curve(labels, predictions_test)
        f1_scores = 2 * recall * precision / (recall + precision)
        threshold = thresholds[np.nanargmax(f1_scores)]
        best_anomaly_f1 = np.nanmax(f1_scores)
        best_macro_f1 = f1_score(labels, predictions_test > threshold, average="macro")
        best_micro_f1 = f1_score(labels, predictions_test > threshold, average="micro")
        roc_auc = roc_auc_score(labels, scores, average="weighted")
        average_precision = average_precision_score(labels, scores)
        print("Threshold:", threshold)
        print(classification_report(labels, predictions_test > threshold))
        return best_anomaly_f1, roc_auc, average_precision, best_macro_f1, best_micro_f1, int(labels.sum())


