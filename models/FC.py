from torch.nn import ModuleList, Linear, BatchNorm1d
from torch import nn
import torch
import torch.nn.functional as F
# pl.seed_everything(12)
from torch.utils.data import TensorDataset

from models.Base import BaseModel


class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()
        self.dropout = dropout

        self.bns = ModuleList([BatchNorm1d(hidden_channels)])
        self.lins = ModuleList([Linear(in_channels, hidden_channels)])
        for _ in range(num_layers - 1):
            self.bns.append(BatchNorm1d(hidden_channels))
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

    def forward(self, x):
        for lin, bn in zip(self.lins[:-1], self.bns):
            x = lin(x).relu_()
            x = bn(x)  # This will have better performance than put this above the activation function
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lins[-1](x)


class FCBaseline(BaseModel):

    def __init__(self, in_nfeats, labelsize, n_layers, n_hidden, dropout):
        super().__init__(n_classes=labelsize)
        #         self.fc = nn.Linear(embdim, labelsize)
        # self.train_dataset = TensorDataset(torch.FloatTensor(train_embeds), torch.LongTensor(train_labels))
        # self.val_dataset = TensorDataset(torch.FloatTensor(val_embeds), torch.LongTensor(val_labels))
        # self.test_dataset = TensorDataset(torch.FloatTensor(test_embeds), torch.LongTensor(test_labels))
        self.in_nfeats = in_nfeats
        self.labelsize = labelsize
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.fc = MLP(in_nfeats, labelsize, hidden_channels=n_hidden,
                      num_layers=n_layers, dropout=dropout)
        #         self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_outputs = []
        self.validation_outputs = []
        self.test_outputs = []



    def forward(self, batch):
        # in lightning, forward defines the prediction/inference actions
        x, labels = batch[0], batch[1]
        preds = self.fc(x)
        #         pred_labels = torch.argmax(preds, dim=1)
        # preds = torch.sigmoid(preds)
        # pred_labels = (preds > 0.5).float()
        return labels, preds

    # def _step(self, batch, batch_idx, dataset):
    def _step(self, batch):
        # training_step defined the train loop. It is independent of forward
        # # index = batch
        # # x, labels = dataset[index][0], dataset[index][1]
        # x, labels = batch[0], batch[1]
        # x, labels = x.to(self.device), labels.to(self.device)
        # # x = x.view(x.size(0), -1)
        # preds = self.fc(x)
        labels, preds = self(batch)
        loss = self.loss_fn(preds, labels.float())
        preds = torch.sigmoid(preds)
        #         pred_labels = torch.argmax(preds, dim=1)
        pred_labels = (preds > 0.5).float()
        # Note: In case GPU OOM for torch metrics, move them to CPU here or in log_result
        # return_dict = {'loss': loss.detach().cpu(), "preds": preds.detach().cpu(),
        #                "pred_labels": pred_labels.detach().cpu(),
        #                "labels": labels.detach().cpu()}
        return_dict = {'loss': loss, "preds": preds, "pred_labels": pred_labels, "labels": labels}
        return loss, return_dict


    def training_step(self, batch, batch_idx):
        # return self._step(batch, batch_idx, self.train_dataset)
        loss, step_dict = self._step(batch)
        self.train_outputs.append(step_dict)
        return loss

    def on_train_epoch_end(self):
        # log epoch metric
        output = self.train_outputs
        preds = torch.cat([x['preds'] for x in output]).detach()
        pred_labels = torch.cat([x['pred_labels'] for x in output]).detach()
        labels = torch.cat([x['labels'] for x in output]).detach()
        loss = torch.stack([x['loss'] for x in output]).mean().detach()

        self.log_result(preds, pred_labels, labels, loss, dataset="train")
        self.train_outputs.clear()

    def validation_step(self, batch, batch_idx):
        loss, step_dict = self._step(batch)
        self.validation_outputs.append(step_dict)
        return loss

    def test_step(self, batch, batch_idx):
        loss, step_dict = self._step(batch)
        self.test_outputs.append(step_dict)
        return loss

    def on_validation_epoch_end(self):
        # log epoch metric
        output = self.validation_outputs
        preds = torch.cat([x['preds'] for x in output]).detach()
        #         import pdb;pdb.set_trace()
        pred_labels = torch.cat([x['pred_labels'] for x in output]).detach()
        labels = torch.cat([x['labels'] for x in output]).detach()
        loss = torch.stack([x['loss'] for x in output]).mean().detach()
        self.log_result(preds, pred_labels, labels, loss, dataset="val")
        self.validation_outputs.clear()

    def on_test_epoch_end(self):
        output = self.test_outputs
        preds = torch.cat([x['preds'] for x in output]).detach()
        pred_labels = torch.cat([x['pred_labels'] for x in output]).detach()
        labels = torch.cat([x['labels'] for x in output]).detach()
        loss = torch.stack([x['loss'] for x in output]).mean().detach()

        self.log_result(preds, pred_labels, labels, loss, dataset="test")
        self.test_outputs.clear()
