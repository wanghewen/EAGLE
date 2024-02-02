import torchmetrics
import pytorch_lightning as pl
import torch
from torchmetrics import PrecisionRecallCurve
from torchmetrics.classification import BinaryPrecisionRecallCurve, MulticlassPrecisionRecallCurve
from torchmetrics.functional import f1_score
from torchmetrics.utilities.data import dim_zero_cat

from global_parameters import learning_rate


# TODO: Change to MulticlassPrecisionRecallCurve and use average=None and get class 1 F1's
class BestBinaryF1(BinaryPrecisionRecallCurve):
    def compute(self):
        precision, recall, thresholds = super().compute()
        f1_scores = 2 * recall * precision / (recall + precision)
        f1_scores[~torch.isfinite(f1_scores)] = -1
        best_f1 = torch.max(f1_scores)
        return best_f1


class BestBinaryF1Macro(BinaryPrecisionRecallCurve):
    def compute(self):
        precision, recall, thresholds = super().compute()
        f1_scores = 2 * recall * precision / (recall + precision)
        f1_scores[~torch.isfinite(f1_scores)] = -1
        if thresholds.shape[0] == 1 and len(thresholds.shape) == 1:
            threshold = thresholds[0]
        else:
            threshold = thresholds[torch.argmax(f1_scores)]
        best_macro_f1 = f1_score(dim_zero_cat(self.preds), dim_zero_cat(self.target),
                                 task="multiclass",
                                 threshold=threshold, average="macro", num_classes=2)
        return best_macro_f1


# class BestF1Macro(MulticlassPrecisionRecallCurve):
#     def compute(self):
#         precision, recall, thresholds = super().compute()
#         precision, recall, thresholds = precision[1], recall[1], thresholds[1]
#         f1_scores = 2 * recall * precision / (recall + precision)
#         f1_scores[~torch.isfinite(f1_scores)] = -1
#         if thresholds.shape[0] == 1 and len(thresholds.shape) == 1:
#             threshold = thresholds[0]
#         else:
#             threshold = thresholds[torch.argmax(f1_scores)]
#         best_macro_f1 = f1_score(dim_zero_cat(self.preds), dim_zero_cat(self.target), task="multiclass",
#                                  threshold=threshold, average="macro", num_classes=2)
#         return best_macro_f1


def try_get_index1(l):
    # import ipdb; ipdb.set_trace()
    if len(l.shape) > 0 and l.shape[0] > 1:
        return l[1]
    else:
        return l

class BaseModel(pl.LightningModule):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        super().__init__()
        # self.tttt()

    # def on_fit_start(self):

        if self.n_classes == 1:
            n_classes = None
            task = "binary"
            top_k = 5
        else:
            n_classes = self.n_classes
            task = "multilabel"
            top_k = min(5, n_classes-1)

        self.train_acc = torchmetrics.Accuracy(task=task, num_labels=n_classes)  # Like micro-F1
        self.valid_acc = torchmetrics.Accuracy(task=task, num_labels=n_classes)
        self.test_acc = torchmetrics.Accuracy(task=task, num_labels=n_classes)
        self.train_auc = torchmetrics.AUROC(task=task, num_labels=n_classes)
        self.valid_auc = torchmetrics.AUROC(task=task, num_labels=n_classes)
        self.test_auc = torchmetrics.AUROC(task=task, num_labels=n_classes)
        self.train_ap = torchmetrics.AveragePrecision(task=task, num_labels=n_classes)
        self.valid_ap = torchmetrics.AveragePrecision(task=task, num_labels=n_classes)
        self.test_ap = torchmetrics.AveragePrecision(task=task, num_labels=n_classes)
        # self.train_best_micro_f1 = BestF1Macro(task=task, num_labels=n_classes)
        # self.valid_best_micro_f1 = BestF1Macro(task=task, num_labels=n_classes)
        # self.test_best_micro_f1 = BestF1Macro(task=task, num_labels=n_classes)
        # if task == "binary":
        self.train_best_anomaly_f1 = BestBinaryF1()
        self.valid_best_anomaly_f1 = BestBinaryF1()
        self.test_best_anomaly_f1 = BestBinaryF1()
        self.train_best_macro_f1 = BestBinaryF1Macro()
        self.valid_best_macro_f1 = BestBinaryF1Macro()
        self.test_best_macro_f1 = BestBinaryF1Macro()
        # self.train_best_macro_f1 = BestF1Macro(num_classes=2, validate_args=True)
        # self.valid_best_macro_f1 = BestF1Macro(num_classes=2, validate_args=True)
        # self.test_best_macro_f1 = BestF1Macro(num_classes=2, validate_args=True)

        self.train_topk_precision = torchmetrics.Precision(task=task, num_labels=n_classes, top_k=top_k)
        self.valid_topk_precision = torchmetrics.Precision(task=task, num_labels=n_classes, top_k=top_k)
        self.test_topk_precision = torchmetrics.Precision(task=task, num_labels=n_classes, top_k=top_k)
        self.train_macro_f1 = torchmetrics.F1Score(task=task, num_labels=n_classes, average="macro")
        self.valid_macro_f1 = torchmetrics.F1Score(task=task, num_labels=n_classes, average="macro")
        self.test_macro_f1 = torchmetrics.F1Score(task=task, num_labels=n_classes, average="macro")
        self.train_micro_f1 = torchmetrics.F1Score(task=task, num_labels=n_classes, average="micro")
        self.valid_micro_f1 = torchmetrics.F1Score(task=task, num_labels=n_classes, average="micro")
        self.test_micro_f1 = torchmetrics.F1Score(task=task, num_labels=n_classes, average="micro")
        self.train_f1_anomaly = torchmetrics.F1Score(task=task, num_labels=n_classes, average=None)
        self.valid_f1_anomaly = torchmetrics.F1Score(task=task, num_labels=n_classes, average=None)
        self.test_f1_anomaly = torchmetrics.F1Score(task=task, num_labels=n_classes, average=None)

        self.train_total_labels = torchmetrics.aggregation.SumMetric()
        self.valid_total_labels = torchmetrics.aggregation.SumMetric()
        self.test_total_labels = torchmetrics.aggregation.SumMetric()


    def log_result(self, preds, pred_labels, labels, loss, dataset=None, logs_to_logger=True):
        assert dataset in ["train", "val", "test", "test_run"]  # test_run is for running evaluation during training
        # preds, pred_labels, labels, loss = preds.detach().cpu(), pred_labels.detach().cpu(), labels.detach().cpu(), loss.detach().cpu()
        if dataset == "train":
            # print("!!!!!!!self.train_acc", self.train_acc(pred_labels, labels))
            if (preds is None) or (labels is None):
                metrics_dict = {
                    'train/loss_epoch': loss,
                    'step': self.current_epoch
                }
            else:
                metrics_dict = {
                    'train/loss_epoch': loss,
                    'train/accuracy_epoch': self.train_acc(pred_labels, labels),
                    'train/auc_epoch': self.train_auc(preds, labels),
                    'train/macro_f1_epoch': self.train_macro_f1(pred_labels, labels),
                    'train/micro_f1_epoch': self.train_micro_f1(pred_labels, labels),
                    'train/f1_anomaly_epoch': try_get_index1(self.train_f1_anomaly(pred_labels, labels)),
                    'train/ap_epoch': self.train_ap(preds, labels),
                    'train/total_labels': self.train_total_labels(labels),
                    'step': self.current_epoch
                }
                if len(preds.shape) == 1 or preds.shape[1] == 1:
                    metrics_dict["train/topk_precision_epoch"] = self.train_topk_precision(preds.reshape(1, -1), labels.reshape(1, -1))
                    metrics_dict["train/best_anomaly_f1_epoch"] = self.train_best_anomaly_f1(preds, labels)
                    # metrics_dict["train/best_macro_f1_epoch"] = self.train_best_macro_f1(torch.cat([1-preds.reshape(-1, 1), preds.reshape(-1, 1)], axis=1), labels.reshape(-1))
                    metrics_dict["train/best_macro_f1_epoch"] = self.train_best_macro_f1(preds, labels)
                    # metrics_dict["train/best_micro_f1_epoch"] = self.train_best_micro_f1(preds, labels)
                else:
                    metrics_dict["train/topk_precision_epoch"] = self.train_topk_precision(preds, labels)
                for metric in [self.train_acc, self.train_auc, self.train_macro_f1, self.train_f1_anomaly,
                               self.train_topk_precision, self.train_ap, self.train_best_anomaly_f1,
                               self.train_best_macro_f1, self.train_total_labels]:
                    metric.reset()
        elif dataset == "val":
            # print("!!!!!!!self.val_acc", self.valid_acc(pred_labels, labels))
            # import ipdb; ipdb.set_trace()
            metrics_dict = {
                'val/loss_epoch': loss,
                'val/accuracy_epoch': self.valid_acc(pred_labels, labels),
                'val/auc_epoch': self.valid_auc(preds, labels),
                'val/macro_f1_epoch': self.valid_macro_f1(pred_labels, labels),
                'val/micro_f1_epoch': self.valid_micro_f1(pred_labels, labels),
                'val/f1_anomaly_epoch': try_get_index1(self.valid_f1_anomaly(pred_labels, labels)),
                'val/ap_epoch': self.valid_ap(preds, labels),
                'val/total_labels': self.valid_total_labels(labels),
                'step': self.current_epoch
            }
            if len(preds.shape) == 1 or preds.shape[1] == 1:
                metrics_dict["val/topk_precision_epoch"] = self.valid_topk_precision(preds.reshape(1, -1), labels.reshape(1, -1))
                metrics_dict["val/best_anomaly_f1_epoch"] = self.valid_best_anomaly_f1(preds, labels)
                # metrics_dict["val/best_macro_f1_epoch"] = self.valid_best_macro_f1(torch.cat([1-preds.reshape(-1, 1), preds.reshape(-1, 1)], axis=1), labels.reshape(-1))
                metrics_dict["val/best_macro_f1_epoch"] = self.valid_best_macro_f1(preds, labels)
                # metrics_dict["val/best_micro_f1_epoch"] = self.valid_best_micro_f1(preds, labels)
            else:
                metrics_dict["val/topk_precision_epoch"] = self.valid_topk_precision(preds, labels)
            for metric in [self.valid_acc, self.valid_auc, self.valid_macro_f1, self.valid_f1_anomaly, 
                           self.valid_topk_precision, self.valid_ap, self.valid_best_anomaly_f1,
                           self.valid_best_macro_f1, self.valid_total_labels]:
                metric.reset()
            # print(self.valid_acc._get_final_stats())
        elif dataset in ["test_run", "test"]:
            metrics_dict = {
                f'{dataset}/loss_epoch': loss,
                f'{dataset}/accuracy_epoch': self.test_acc(pred_labels, labels),
                f'{dataset}/auc_epoch': self.test_auc(preds, labels),
                f'{dataset}/macro_f1_epoch': self.test_macro_f1(pred_labels, labels),
                f'{dataset}/micro_f1_epoch': self.test_micro_f1(pred_labels, labels),
                f'{dataset}/f1_anomaly_epoch': try_get_index1(self.test_f1_anomaly(pred_labels, labels)),
                f'{dataset}/ap_epoch': self.test_ap(preds, labels),
                f'{dataset}/total_labels': self.test_total_labels(labels),
                'step': self.current_epoch
            }
            if len(preds.shape) == 1 or preds.shape[1] == 1:
                metrics_dict["test/topk_precision_epoch"] = self.test_topk_precision(preds.reshape(1, -1), labels.reshape(1, -1))
                metrics_dict["test/best_anomaly_f1_epoch"] = self.test_best_anomaly_f1(preds, labels)
                # metrics_dict["test/best_macro_f1_epoch"] = self.test_best_macro_f1(torch.cat([1-preds.reshape(-1, 1), preds.reshape(-1, 1)], axis=1), labels.reshape(-1))
                metrics_dict["test/best_macro_f1_epoch"] = self.test_best_macro_f1(preds, labels)
                # metrics_dict["test/best_micro_f1_epoch"] = self.test_best_micro_f1(preds, labels)
            else:
                metrics_dict["test/topk_precision_epoch"] = self.test_topk_precision(preds, labels)
            for metric in [self.test_acc, self.test_auc, self.test_macro_f1, self.test_f1_anomaly, 
                           self.test_topk_precision, self.test_ap, self.test_best_anomaly_f1,
                           self.test_best_macro_f1, self.test_total_labels]:
                metric.reset()
        # import pdb; pdb.set_trace()
        # print(metrics_dict)
        # for key in metrics_dict:
        #     if hasattr(metrics_dict[key], "item"):
        #         # import ipdb; ipdb.set_trace()
        #         metrics_dict[key] = metrics_dict[key].item()
        self.log_dict(metrics_dict, logger=logs_to_logger)
        # for _, value in metrics_dict:
        #     value.reset()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer