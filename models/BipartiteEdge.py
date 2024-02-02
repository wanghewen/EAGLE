import datetime
import os

import scipy.linalg
import scipy.sparse
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
# from scipy.linalg import clarkson_woodruff_transform
from scipy.sparse.linalg import ArpackError
import scipy.sparse as sp
import torch
import numpy as np
import pytorch_lightning as pl
from torch.nn import BatchNorm1d, Linear

# from analysis.matrix_error import epp_vs_lra
from dataloaders.FC_dataloader import FCDataloaderInitializer
from global_parameters import convert_matrix_to_numpy, LANGUAGE_EMBEDDING_CACHE_DIR, log_directory
import CommonModules as CM

from models.Base import BaseModel
from models.FC import FCBaseline, MLP
from models.GNNVanilla import evaluate, dot_product, _torch_sparse_tensor_to_sparse_mx
from utils import export_numpy_array, import_numpy_array

USED_CLASS = BaseModel


def svd_with_error_handling(i, svd_dim):
    try:
        u, sigma, vT = scipy.sparse.linalg.svds(i, k=svd_dim,
                                                ncv=min(2 * svd_dim + 2, min(i.shape) - 1), random_state=12)
    except ArpackError:
        # traceback.print_exc()
        print("Current ncv failed, try using default ncv...")
        # u, sigma, vT = scipy.sparse.linalg.svds(i, k=svd_dim, ncv=3 * svd_dim)
        try:
            u, sigma, vT = scipy.sparse.linalg.svds(i, k=svd_dim, random_state=12)
        except ArpackError:
            print("Deafult ncv failed, try using more ncv (svd_dim * 3)...")
            try:
                u, sigma, vT = scipy.sparse.linalg.svds(i, k=svd_dim, ncv=svd_dim * 3, random_state=12)
            except ArpackError:
                print("Current ncv failed, try using more ncv (svd_dim * 4)...")
                try:
                    u, sigma, vT = scipy.sparse.linalg.svds(i, k=svd_dim, ncv=svd_dim * 4, random_state=12)
                except ArpackError:
                    print("Current ncv failed, try using more ncv (svd_dim * 5)...")
                    u, sigma, vT = scipy.sparse.linalg.svds(i, k=svd_dim, ncv=svd_dim * 5, random_state=12)
    return u, sigma, vT


class FC_BipartiteEdge(FCBaseline):
    def __init__(self, transform_x=False, separate_aer_embeddings=False, aggregator=None,
                 use_l2_norm=False, use_aer=True, svd_dim=None, alpha=None, beta=None, gamma=None,
                 sigma_u=None, sigma_v=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform_x = transform_x
        self.separate_aer_embeddings = separate_aer_embeddings
        # self.concat_embeddings = concat_embeddings
        self.aggregator = aggregator
        self.use_l2_norm = use_l2_norm
        self.use_aer = use_aer
        self.svd_dim = svd_dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma_u = sigma_u
        self.sigma_v = sigma_v
        self.batchnorm_start_1 = BatchNorm1d(self.in_nfeats)
        # self.batchnorm_start_2 = BatchNorm1d(self.in_nfeats)
        # if self.svd_dim is None and self.separate_aer_embeddings:
        if self.separate_aer_embeddings:
            self.transform_x_layer1 = Linear(self.in_nfeats//2, self.in_nfeats//4)
            if self.in_nfeats % 2 == 1:
                self.transform_x_layer2 = Linear(self.in_nfeats//2+1, self.in_nfeats//4)
            else:
                self.transform_x_layer2 = Linear(self.in_nfeats//2, self.in_nfeats//4)
        else:
            self.transform_x_layer1 = Linear(self.in_nfeats, self.in_nfeats//4)
            self.transform_x_layer2 = Linear(self.in_nfeats, self.in_nfeats//4)
        self.dropout_layer = torch.nn.Dropout(0.5)
        # self.transform_x_layer1 = Linear(self.in_nfeats, 256)
        # self.transform_x_layer2 = Linear(self.in_nfeats, 256)
        self.batchnorm1 = BatchNorm1d(self.in_nfeats//4)
        self.batchnorm2 = BatchNorm1d(self.in_nfeats//4)
        if self.separate_aer_embeddings and self.aggregator == "concat":
            # self.batchnorm = BatchNorm1d(self.n_hidden//2+self.n_hidden)
            self.fc = MLP(self.n_hidden//4*2+self.n_hidden, self.labelsize, hidden_channels=self.n_hidden,
                          num_layers=self.n_layers, dropout=self.dropout)
            # self.fc = Linear(self.n_hidden//2+self.n_hidden, self.labelsize)
        else:
            # self.batchnorm = BatchNorm1d(self.n_hidden)
            # self.fc = MLP(self.n_hidden, self.labelsize, hidden_channels=self.n_hidden,
            #               num_layers=self.n_layers, dropout=self.dropout)
            # self.batchnorm = BatchNorm1d(self.n_hidden+256)
            # self.fc = MLP(self.n_hidden+256, self.labelsize, hidden_channels=self.n_hidden,
            #               num_layers=self.n_layers, dropout=self.dropout)
            # self.batchnorm = BatchNorm1d(256)
            # self.fc = MLP(256, self.labelsize, hidden_channels=self.n_hidden,
            #               num_layers=self.n_layers, dropout=self.dropout)
            # self.batchnorm = BatchNorm1d(self.n_hidden//4 + self.n_hidden)
            # self.fc = Linear(self.n_hidden // 2 + self.n_hidden, self.labelsize)
            self.fc = MLP(self.n_hidden//4+self.n_hidden, self.labelsize, hidden_channels=self.n_hidden,
                          num_layers=self.n_layers, dropout=self.dropout)

    def forward(self, batch):
        # Should also refer to self._step
        (x, labels), (Qs, mask, features_numpy) = batch
        x, labels = x.squeeze(0), labels.squeeze(0)
        # x, labels = features_numpy, labels.squeeze(0)
        # CM.IO.ExportToPkl("./temp.pkl", x)
        assert (self.use_aer and ((Qs is not None) or (self.svd_dim is None))) or (not self.use_aer)
        # Make sure we use q to propagate features
        if Qs is not None:  # AER
            if self.svd_dim is None:
                if not self.separate_aer_embeddings:
                    # x = self.batchnorm_start_1(x)
                    x_prop = self.transform_x_layer1(x)
                    x_prop = torch.relu(x_prop)
                    x_prop = self.batchnorm1(x_prop)
                else:
                    # x = self.batchnorm_start_1(x)
                    x1 = x[:, :x.shape[1]//2]
                    x2 = x[:, x.shape[1]//2:]
                    if self.transform_x:
                        x1 = self.transform_x_layer1(x1)
                        x2 = self.transform_x_layer2(x2)
                        x1 = torch.relu(x1)
                        x2 = torch.relu(x2)
                        x1 = self.batchnorm1(x1)
                        x2 = self.batchnorm2(x2)
                        x1 = self.beta * x1
                        x2 = (1-self.beta) * x2
                        if self.aggregator == "concat":
                            x_prop = torch.cat([x1, x2], dim=1)
                        elif self.aggregator == "sum":
                            x_prop = x1 + x2
                        elif self.aggregator == "max":
                            x_prop = torch.max(torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2), dim=2)[0]
            else:
                if Qs.shape[0] == 1:  # Non separate embedding
                    q = Qs[0]
                    # q = torch.FloatTensor(q).to(self.device)
                    # x = torch.FloatTensor(x).to(self.device)
                    # x = self.batchnorm_start_1(x)
                    if self.transform_x:
                        x_prop = self.transform_x_layer1(x)
                        x_prop = torch.relu(x_prop)
                        x_prop = self.batchnorm1(x_prop)
                        # import ipdb; ipdb.set_trace()
                        x_prop = q @ (q.T @ x_prop)
                    else:
                        raise AssertionError
                        x_prop = q @ (q.T @ x)
                        x_prop = self.transform_x_layer1(x_prop)
                        x_prop = torch.relu(x_prop)
                        x_prop = self.batchnorm1(x_prop)
                    # x = torch.FloatTensor(x).to(self.device)
                    # x = x.float()
                    # (q.cpu().numpy() @ (q.T.cpu().numpy() @ x.cpu().numpy()))[0,0]
                elif Qs.shape[0] == 2:  # Separate embeddings
                    q1 = Qs[0]
                    q2 = Qs[1]
                    # if self.concat_embeddings:
                    #     x1 = x[:, :x.shape[1]]
                    #     x2 = x[:, x.shape[1]:]
                    # else:
                    # x = self.batchnorm_start_1(x)
                    x1 = x[:, :x.shape[1]//2]
                    x2 = x[:, x.shape[1]//2:]
                    if self.transform_x:
                        x1 = self.transform_x_layer1(x1)
                        x2 = self.transform_x_layer2(x2)
                        x1 = torch.relu(x1)
                        x2 = torch.relu(x2)
                        # import ipdb; ipdb.set_trace()
                            # q1 = torch.nn.functional.normalize(q1, dim=1, p=2)
                            # q2 = torch.nn.functional.normalize(q2, dim=1, p=2)
                        if self.gamma is not None:
                            BU = torch.ones(x1.shape).to(x1.device)
                            scalar_u = torch.sum(x1, dim=1, keepdim=True)
                            # scalar_u = torch.sum(x1.abs(), dim=1, keepdim=True)
                            BU = BU * scalar_u
                            BV = torch.ones(x2.shape).to(x2.device)
                            scalar_v = torch.sum(x2, dim=1, keepdim=True)
                            # scalar_v = torch.sum(x2.abs(), dim=1, keepdim=True)
                            BV = BV * scalar_v
                            TU = self.gamma / (1 - self.sigma_u[-2] ** 2)
                            TV = self.gamma / (1 - self.sigma_v[-2] ** 2)
                        x1 = q1 @ (q1.T @ x1)
                        x2 = q2 @ (q2.T @ x2)
                    else:
                        raise AssertionError
                        x1 = q1 @ (q1.T @ x1)
                        x2 = q2 @ (q2.T @ x2)
                        x1 = self.transform_x_layer1(x1)
                        x2 = self.transform_x_layer2(x2)
                        x1 = torch.relu(x1)
                        x2 = torch.relu(x2)
                        x1 = self.batchnorm1(x1)
                        x2 = self.batchnorm2(x2)
                    if self.gamma is not None:
                        if self.current_epoch == 0:
                            print("BU coefficient:", (self.alpha ** (TU + 1)) / (1-self.alpha),
                                  "BV coefficient:", (self.alpha ** (TV + 1)) / (1-self.alpha))
                        x1 = x1 - (self.alpha ** (TU + 1)) / (1-self.alpha) * BU
                        x2 = x2 - (self.alpha ** (TV + 1)) / (1-self.alpha) * BV
                    if self.use_l2_norm:
                        x1 = torch.nn.functional.normalize(x1, dim=1, p=2)
                        x2 = torch.nn.functional.normalize(x2, dim=1, p=2)
                    x1 = self.batchnorm1(x1)
                    x2 = self.batchnorm2(x2)
                    x1 = self.beta * x1
                    x2 = (1-self.beta) * x2
                    if self.aggregator == "concat":
                        x_prop = torch.cat([x1, x2], dim=1)
                    elif self.aggregator == "sum":
                        x_prop = x1 + x2
                    elif self.aggregator == "max":
                        x_prop = torch.max(torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2), dim=2)[0]
        else:  # MLP on raw features
            # if self.transform_x:
            x_prop = self.transform_x_layer1(x)
            x_prop = torch.relu(x_prop)
            x_prop = self.batchnorm1(x_prop)
        x = torch.cat([x_prop, x], dim=1)
        # if self.transform_x:
        #     x = torch.relu(x)
        # x = self.batchnorm(x)
        # import ipdb; ipdb.set_trace()
        x = self.dropout_layer(x)

        x = x[mask]
        labels = labels[mask]
        preds = self.fc(x)
        return labels, preds


class BipartiteEdge(USED_CLASS):
    def __init__(self,
                 svd_dim=256,  # None means not use SVD to reduce feature dim first
                 ppr_decay_factor=0.5,
                 beta=0.5,
                 gamma=None,
                 train_ratio=1.0,  # Reduce training data to save training time for large dataset
                 transform_x=True,
                 separate_aer_embeddings=False,
                 aggregator=None,
                 use_l2_norm=False,
                 use_aer=True,
                 use_gebe=False,
                 cache_folder_path=None,
                 force_redo_svd=False,
                 only_print_dataset_stats=False,
                 **kwargs):
        super(USED_CLASS, self).__init__()
        #         self.g = g
        self.svd_dim = svd_dim
        self.ppr_decay_factor = ppr_decay_factor
        self.beta = beta
        self.gamma = gamma
        print(f"ppr_decay_factor={ppr_decay_factor}")
        print(f"beta={beta}")
        print(f"gamma={gamma}")
        print(f"use l2 norm={use_l2_norm}")
        print(f"svd_dim={svd_dim}")
        self.train_ratio = train_ratio
        self.transform_x = transform_x
        self.separate_aer_embeddings = separate_aer_embeddings
        # self.concat_embeddings = concat_embeddings
        self.aggregator = aggregator
        assert self.aggregator in [None, "concat", "sum", "max"]
        self.use_l2_norm = use_l2_norm
        self.use_aer = use_aer
        self.use_gebe = use_gebe
        self.cache_folder_path = cache_folder_path
        self.force_redo_svd = force_redo_svd
        self.only_print_dataset_stats = only_print_dataset_stats
        if self.cache_folder_path is not None:
            os.makedirs(self.cache_folder_path, exist_ok=True)
        # self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased",
        #                                                cache_dir=LANGUAGE_EMBEDDING_CACHE_DIR)

    def _aer(self, edge_features, Eu, Ev, beta, separate_aer_embeddings):
        """

        :param edge_features:
        :param incidence_matrix: Transpose of the incidence matrix
        :return:
        """
        Zs = []
        Qs = []
        Eu_diags = Eu.sum(0).A.flatten()
        Ev_diags = Ev.sum(0).A.flatten()
        if separate_aer_embeddings:
            if self.svd_dim is None:
                print("Use power iterations to compute P/Z...")
                D_inverse = sp.diags(1 / Eu_diags, format="csr")
                # P = Eu @ D_inverse @ Eu.T
                Z = edge_features
                sum_term = edge_features
                for i in range(10):
                    print("current iteration:", i)
                    sum_term = self.ppr_decay_factor * (Eu @ (D_inverse @ (Eu.T @ sum_term)))
                    Z += sum_term
                Zs.append(Z)
                Qs.append(np.array([0]))

                D_inverse = sp.diags(1 / Ev_diags, format="csr")
                # P = Ev @ D_inverse @ Ev.T
                Z = edge_features
                sum_term = edge_features
                for i in range(10):
                    print("current iteration:", i)
                    sum_term = self.ppr_decay_factor * (Ev @ (D_inverse @ (Ev.T @ sum_term)))
                    Z += sum_term
                Zs.append(Z)
                Qs.append(np.array([0]))
                sigma_u = sigma_v = None
            else:
                D_inverse_u = sp.diags(1 / np.sqrt(Eu_diags), format="csr")
                i = dot_product(Eu, D_inverse_u)
                # F, m*k
                # u, original_sigma_ter, vT = randomized_svd(i_m, i_n_T, rep_dimension, random_state=12)  # m*k, k, k*k
                # import ipdb; ipdb.set_trace()
                print("svds matrix u nnz:", i.nnz, flush=True)
                svd_dim = min(self.svd_dim, min(Eu.shape) - 2, min(Ev.shape) - 2)
                # CM.IO.ExportToPkl("./temp.pkl", i)
                u, sigma_u, vT = svd_with_error_handling(i, svd_dim=svd_dim)
                sigma_u = sigma_u / max(sigma_u)
                # u, sigma, vT = randomized_svd(i, n_components=self.svd_dim)
                # u, sigma, vT = bksvd(i, k=self.svd_dim)
                print("largest singular value:", max(sigma_u))
                Sigma = np.diag(np.sqrt(1 / (1 - self.ppr_decay_factor * (sigma_u ** 2))))
                Q = u @ Sigma
                Z = Q @ (Q.T @ edge_features)
                Zs.append(Z)
                Qs.append(Q)

                D_inverse_v = sp.diags(1 / np.sqrt(Ev_diags), format="csr")
                i = dot_product(Ev, D_inverse_v)
                # F, m*k
                # u, original_sigma_ter, vT = randomized_svd(i_m, i_n_T, rep_dimension, random_state=12)  # m*k, k, k*k
                # import ipdb; ipdb.set_trace()
                print("svds matrix v nnz:", i.nnz, flush=True)
                u, sigma_v, vT = svd_with_error_handling(i, svd_dim=svd_dim)
                sigma_v = sigma_v / max(sigma_v)

                # u, sigma, vT = randomized_svd(i, n_components=self.svd_dim)
                # u, sigma, vT = bksvd(i, k=self.svd_dim)
                print("largest singular value:", max(sigma_v))
                Sigma = np.diag(np.sqrt(1 / (1 - self.ppr_decay_factor * (sigma_v ** 2))))
                Q = u @ Sigma
                Z = Q @ (Q.T @ edge_features)
                Zs.append(Z)
                Qs.append(Q)
        else:
            E = sp.hstack([np.sqrt(beta) * Eu, (1 - np.sqrt(beta)) * Ev])
            if self.svd_dim is None:
                print("Use power iterations to compute P/Z...")
                D_inverse = sp.diags(1 / np.concatenate([Eu_diags, Ev_diags]), format="csr")
                # P = E @ D_inverse @ E.T
                Z = edge_features
                sum_term = edge_features
                for i in range(10):
                    print("current iteration:", i)
                    sum_term = self.ppr_decay_factor * (E @ (D_inverse @ (E.T @ sum_term)))
                    Z += sum_term
                Zs = [Z]
                Qs = [np.array([0])]
                sigma_u = sigma_v = None
            else:
                D_inverse = sp.diags(1 / np.sqrt(np.concatenate([Eu_diags, Ev_diags])), format="csr")
                i = dot_product(E, D_inverse)
                # F, m*k
                # u, original_sigma_ter, vT = randomized_svd(i_m, i_n_T, rep_dimension, random_state=12)  # m*k, k, k*k
                # import ipdb; ipdb.set_trace()
                print("svds matrix nnz:", i.nnz, flush=True)
                svd_dim = min(self.svd_dim, min(E.shape) - 2)
                u, sigma, vT = svd_with_error_handling(i, svd_dim=svd_dim)
                sigma = sigma / max(sigma)
                sigma_u = sigma_v = sigma
                Sigma = np.diag(np.sqrt(1 / (1 - self.ppr_decay_factor*(sigma**2))))
                Q = u @ Sigma
                Z = Q @ (Q.T @ edge_features)
                Zs = [Z]
                Qs = [Q]
        if self.aggregator == "concat":
            Z = np.concatenate(Zs, axis=1)
        elif self.aggregator == "sum":
            Z = np.sum(Zs, axis=0)
        elif self.aggregator == "max":
            Z = np.max(Zs, axis=0)
        return Z, Qs, sigma_u, sigma_v

    def get_static_graph_features(self, dataloaders=None, gpus=None, **kwargs):
        graph_features = {"features_ter": None}
        if self.use_gebe:
            train_dataloader, val_dataloader, test_dataloader = dataloaders
            train_data = next(iter(train_dataloader))
            label_names = train_data["label_names"]
        else:
            u_ter, sigma_ter, i_m_i_n_T, original_sigma_ter = None, None, None, None
            train_dataloader, val_dataloader, test_dataloader = dataloaders
            train_data = next(iter(train_dataloader))
            val_data = next(iter(val_dataloader))
            test_data = next(iter(test_dataloader))

            mask_key = "edges_mask"
            feature_key = "efeats"
            label_names = train_data["label_names"]

            if isinstance(train_data[feature_key], torch.Tensor):
                train_mask = train_data[mask_key]
                val_mask = val_data[mask_key]
                test_mask = test_data[mask_key]
                labels = train_data["labels"].cpu().numpy()
            else:
                train_mask = train_data[mask_key].cpu().numpy()
                val_mask = val_data[mask_key].cpu().numpy()
                test_mask = test_data[mask_key].cpu().numpy()
                labels = train_data["labels"]
            # print("label distribution:", labels[train_mask | val_mask | test_mask].sum(0))
            # print("no label edges:", (labels[train_mask | val_mask | test_mask].sum(1)==0).sum())
            # return
            # import os
            # train_embeds, train_labels = CM.IO.ImportFromPkl(
            #     os.path.join("/data/PublicGraph/DBLP/data_count_vectorizer", "train_embeds_labels.pkl"))
            # test_embeds, test_labels = CM.IO.ImportFromPkl(os.path.join("/data/PublicGraph/DBLP/data_count_vectorizer_split_by_node", "test_embeds_labels.pkl"))
            if self.train_ratio < 1:
                train_mask[np.where(train_mask)[0][int(train_mask.sum() * self.train_ratio):]] = False

            if isinstance(train_data[feature_key], torch.Tensor):
                edge_features = train_data[feature_key].cpu().numpy().astype(np.float32)
            elif isinstance(train_data[feature_key], np.ndarray) or sp.issparse(train_data[feature_key]):
                edge_features = train_data[feature_key].astype(np.float32)
            else:
                edge_features = train_data[feature_key]
                # if gpus is not None:
                #     self.encoder = self.encoder.to(f"cuda:{gpus[0]}")
                #     with torch.no_grad():
                #         # for i in range(0, edge_features.input_ids.shape[0] - 500, 500):
                #         times = []
                #         for i in range(0, 10000, 500):
                #             CM.Utilities.TimeElapsed(Unit=False, LastTime=True)
                #             outputs = self.encoder(input_ids=edge_features.input_ids[i:i+500].to(f"cuda:{gpus[0]}"),
                #                                    attention_mask=edge_features.attention_mask[i:i+500].to(f"cuda:{gpus[0]}"))
                #             times.append(CM.Utilities.TimeElapsed(Unit=False, LastTime=True))
                #     print(times, sum(times)/10000)

            if isinstance(edge_features, scipy.sparse.csr_matrix):
                print("nnz(X):", edge_features.nnz)
            else:
                print("nnz(X):", np.count_nonzero(edge_features))

            ##############Do Propagation##############
            print("feature dimension:", edge_features.shape[1])
            print("incidence matrix shape:", train_data["g"].incidence_matrix("in").shape,
                  train_data["g"].incidence_matrix("out").shape)
            print("number of labels:", labels.shape[1])
            if self.only_print_dataset_stats:
                return {}
            # return None # Used to generate cached dataset
            # incidence_matrix = train_data["g"].incidence_matrix("in") + train_data["g"].incidence_matrix("out")
            # incidence_matrix = _torch_sparse_tensor_to_sparse_mx(incidence_matrix).T
            # features_ter = edge_features[:, :100]  # For debug purpose
            Eu = _torch_sparse_tensor_to_sparse_mx(train_data["g"].incidence_matrix("out")).T
            Ev = _torch_sparse_tensor_to_sparse_mx(train_data["g"].incidence_matrix("in")).T

        features = []
        start_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
        if self.use_aer:
            cache_folder_path = str(self.cache_folder_path)
            cache_features_epp_path = os.path.join(cache_folder_path, f"cache_features_epp_separate"
                                                                      f"={self.separate_aer_embeddings}_svd_dim="
                                                                      f"{self.svd_dim}.blp")
            cache_Q0_path = os.path.join(cache_folder_path, f"cache_Q0_separate"
                                                            f"={self.separate_aer_embeddings}_svd_dim="
                                                            f"{self.svd_dim}.blp")
            cache_Q1_path = os.path.join(cache_folder_path, f"cache_Q1_separate"
                                                            f"={self.separate_aer_embeddings}_svd_dim="
                                                            f"{self.svd_dim}.blp")
            cache_sigma_path = os.path.join(cache_folder_path, f"cache_sigma_separate"
                                                            f"={self.separate_aer_embeddings}_svd_dim="
                                                            f"{self.svd_dim}.pkl")
            if CM.IO.FileExist(cache_sigma_path) and (not self.force_redo_svd):
            # if False:
                print("import from _aer cache files at:", self.cache_folder_path)
                features_epp = import_numpy_array(cache_features_epp_path)
                # import ipdb; ipdb.set_trace()
                sigma_u, sigma_v = CM.IO.ImportFromPkl(cache_sigma_path)
                if self.separate_aer_embeddings:
                    Qs = [import_numpy_array(cache_Q0_path), import_numpy_array(cache_Q1_path)]
                else:
                    Qs = [import_numpy_array(cache_Q0_path)]
            else:
                print("export to _aer cache files at:", self.cache_folder_path)
                features_epp, Qs, sigma_u, sigma_v = self._aer(edge_features, Eu, Ev, beta=self.beta,
                                                               separate_aer_embeddings=self.separate_aer_embeddings)
                if cache_folder_path != "None":
                    CM.IO.ExportToPkl(cache_sigma_path, [sigma_u, sigma_v])
                    export_numpy_array(features_epp, cache_features_epp_path)
                    if self.separate_aer_embeddings:
                        export_numpy_array(Qs[0], cache_Q0_path)
                        export_numpy_array(Qs[1], cache_Q1_path)
                    else:
                        export_numpy_array(Qs[0], cache_Q0_path)
            # if self.transform_x:  # Need to transform in FC ourselves. Otherwise, we don't need to do any
                # transformations in FC.
                # import ipdb; ipdb.set_trace()
            if self.svd_dim is not None:
                Qs = torch.FloatTensor(np.concatenate([Q.reshape(1, *Q.shape) for Q in Qs], axis=0))
                # Qs = np.concatenate([Q.reshape(1, *Q.shape) for Q in Qs], axis=0)
                features_edges = edge_features #!!!!!!!!!!!!!!!!!!!Shouldn't remove this!
            else:
                features_edges = features_epp
        else:
            features_edges, Qs = edge_features, None

        end_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
        print(f"_aer_ppr time elapsed: {end_training_time - start_training_time} sec")
        features.append(features_edges)

        sparse_flag = False
        for feature in features:
            if scipy.sparse.isspmatrix(feature):
                sparse_flag = True
        if sparse_flag:
            features = scipy.sparse.hstack(features, format="csr")
        else:
            features = np.concatenate(features, axis=1)
        # features = normalize(np.concatenate([features_ter, features_epp], axis=1), norm='l2', axis=1)
        ##########################################

        print(f"sigma_u[-2]={sigma_u[-2]}" if sigma_u is not None else f"sigma_u[-2]=None")
        print(f"sigma_v[-2]={sigma_v[-2]}" if sigma_v is not None else f"sigma_u[-2]=None")
        print(f"sigma_u[-2]**2={sigma_u[-2]**2}" if sigma_u is not None else f"sigma_u[-2]**2=None")
        print(f"sigma_v[-2]**2={sigma_v[-2]**2}" if sigma_v is not None else f"sigma_u[-2]**2=None")
        print(f"1/(1-sigma_u[-2]**2)={1/(1-sigma_u[-2]**2)}" if sigma_u is not None else f"1/(1-sigma_u[-2]**2)=None")
        print(f"1/(1-sigma_v[-2]**2)={1/(1-sigma_v[-2]**2)}" if sigma_v is not None else f"1/(1-sigma_v[-2]**2)=None")
        print(f"sigma_u[0]={sigma_u[0]}" if sigma_u is not None else f"sigma_u[0]=None")
        print(f"sigma_v[0]={sigma_v[0]}" if sigma_v is not None else f"sigma_u[0]=None")
        print(f"1/(1-self.ppr_decay_factor*(sigma_u[0]**2))={1/(1-self.ppr_decay_factor*(sigma_u[0]**2))}" if sigma_u is not None else f"1/(1-self.ppr_decay_factor*(sigma_u[0]**2))=None")
        print(f"1/(1-self.ppr_decay_factor*(sigma_v[0]**2))={1/(1-self.ppr_decay_factor*(sigma_v[0]**2))}" if sigma_v is not None else f"1/(1-self.ppr_decay_factor*(sigma_v[0]**2))=None")
        graph_features |= {
            "features": features, "labels": labels, "train_mask": train_mask, "val_mask": val_mask,
            "test_mask": test_mask, "Qs": Qs, "label_names": label_names, "sigma_u" : sigma_u,
            "sigma_v": sigma_v
        }
        return graph_features

    def fit(self, graph_features=None, max_epochs=1000, gpus=None, logger=None, **kwargs):
        """
        Use Sklearn instead of Pytorch-lightning to fit the data

        :param dataloaders:
        :param max_epochs:
        :param gpus:
        :return:
        """
        if self.only_print_dataset_stats:
            return None
        features, labels, train_mask, val_mask, _, Qs, sigma_u, sigma_v = graph_features["features"], \
            graph_features["labels"], \
            graph_features["train_mask"], graph_features["val_mask"], \
            graph_features["test_mask"], graph_features["Qs"], graph_features["sigma_u"], graph_features["sigma_v"]

        if len(labels.shape) == 1 or labels.shape[1] == 1:
            label_size = 1
            labels = labels.reshape(-1, 1)
            self.multilabel = False
        else:
            label_size = labels.shape[1]
            self.multilabel = True

        in_nfeats = features.shape[1]
        model = FC_BipartiteEdge(in_nfeats=in_nfeats, labelsize=label_size, n_layers=1,
                                 n_hidden=features.shape[1],
                                 # n_hidden=128,
                                 dropout=0.5, transform_x=self.transform_x, aggregator=self.aggregator,
                                 separate_aer_embeddings=self.separate_aer_embeddings, use_l2_norm=self.use_l2_norm,
                                 use_aer=self.use_aer, svd_dim=self.svd_dim, alpha=self.ppr_decay_factor,
                                 beta=self.beta, gamma=self.gamma, sigma_u=sigma_u, sigma_v=sigma_v)
        # import ipdb;ipdb.set_trace()
        features_numpy = convert_matrix_to_numpy(features)
        features = torch.FloatTensor(features_numpy)
        if isinstance(labels, torch.Tensor):
            labels = labels.float()
        else:
            labels = torch.FloatTensor(labels)
        labels[labels > 0] = 1
        labels = labels.long()
        # bulk_data = None
        # if self.transform_x:
        bulk_data = Qs
        # train_data = [features[train_mask], labels[train_mask]]
        train_data = [features, labels]
        # val_data = [features[val_mask], labels[val_mask]]
        val_data = [features, labels]
        train_dataloader = FCDataloaderInitializer().build_dataloader_non_graph(*train_data,
                                                                                bulk_data=[bulk_data, train_mask, features_numpy],
                                                                                # batch_size=10000,
                                                                                batch_size=None,
                                                                                shuffle=False,
                                                                                train_val_test="train")
        val_dataloader = FCDataloaderInitializer().build_dataloader_non_graph(*val_data,
                                                                              bulk_data=[bulk_data, val_mask, features_numpy],
                                                                              # batch_size=10000,
                                                                              batch_size=None,
                                                                              shuffle=False,
                                                                              train_val_test="val")
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join("./trained_models", "MLPVanilla", str(datetime.datetime.now())),
            # dirpath=os.path.join("./trained_models", "MLPVanilla_nobatchnorm_fullbatch", str(datetime.datetime.now())),
            filename='epoch={epoch}-val_loss_epoch={'
                     'val/loss_epoch:.6f}-val_accuracy_epoch={'
                     'val/accuracy_epoch:.4f}-val_macro_f1_epoch={'
                     'val/macro_f1_epoch:.4f}',
            # monitor='val/best_macro_f1_epoch', mode="max", save_last=True,
            monitor='val/auc_epoch', mode="max", save_last=True,
            # monitor='val/macro_f1_epoch', mode="max", save_last=True,
            # monitor='val/ap_epoch', mode="max", save_last=True,
            # monitor='val/loss_epoch', mode="min", save_last=True,
            auto_insert_metric_name=False)
        earlystopping_callback = EarlyStopping(monitor='val/loss_epoch',
                                               min_delta=0.00,
                                               # patience=1,
                                               patience=30,  # Need to multiply check_val_every_n_epoch to obtain the
                                               # final val epochs
                                               verbose=True,
                                               mode='min')
        if gpus is not None:
            accelerator = "gpu"
        else:
            accelerator = "cpu"
            gpus = "auto"
        # profiler = PyTorchProfiler(dirpath="./logs_20230925", profile_memory=True, with_stack=False,
        #                            row_limit=50, sort_by_key="self_cuda_memory_usage")
        trainer = pl.Trainer(max_epochs=max_epochs, accelerator=accelerator, devices=gpus,
                             callbacks=[checkpoint_callback, earlystopping_callback],
                             check_val_every_n_epoch=10,
                             profiler=False,
                             precision=32, logger=logger)
        start_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
        trainer.fit(model, train_dataloader, [val_dataloader])
        end_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
        print(f"Training time elapsed: {end_training_time - start_training_time} sec")
        print(checkpoint_callback.best_model_path)
        return trainer

    def test(self, trainer, graph_features=None):
        if self.only_print_dataset_stats:
            return None
        features, labels, train_mask, val_mask, test_mask, Qs, label_names = (graph_features["features"],
                                                                              graph_features["labels"], \
            graph_features["train_mask"], graph_features["val_mask"], \
            graph_features["test_mask"], graph_features["Qs"], graph_features["label_names"])

        trainer: pl.Trainer
        features_numpy = convert_matrix_to_numpy(features)
        features = torch.FloatTensor(features_numpy)
        if isinstance(labels, torch.Tensor):
            labels = labels.float()
        else:
            labels = torch.FloatTensor(labels)
        labels[labels > 0] = 1
        labels = labels.long()
        if train_mask is not None:
            # bulk_data = None
            # if self.transform_x:
            bulk_data = Qs
            # train_data = [features[train_mask], labels[train_mask]]
            train_data = [features, labels]
            train_dataloader = FCDataloaderInitializer().build_dataloader_non_graph(*train_data,
                                                                                    bulk_data=[bulk_data, train_mask, features_numpy],
                                                                                    # batch_size=10000,
                                                                                    batch_size=None,
                                                                                    shuffle=False,
                                                                                    train_val_test="train")
            train_labels, train_prediction = zip(*trainer.predict(dataloaders=train_dataloader,
                                                                  ckpt_path=trainer.checkpoint_callback.best_model_path))
            train_labels = torch.cat(train_labels)
            train_prediction = torch.cat(train_prediction)
            print("train anomaly F1, roc_auc_score, AP, macro_f1, total_labels",
                  evaluate(train_labels,
                           train_prediction,
                           multilabel=self.multilabel))
        if val_mask is not None:
            # if self.transform_x:
            bulk_data = Qs
            # val_data = [features[val_mask], labels[val_mask]]
            val_data = [features, labels]
            val_dataloader = FCDataloaderInitializer().build_dataloader_non_graph(*val_data,
                                                                                  bulk_data=[bulk_data, val_mask, features_numpy],
                                                                                  # batch_size=10000,
                                                                                  batch_size=None,
                                                                                  shuffle=False,
                                                                                  train_val_test="val")
            val_labels, val_prediction = zip(*trainer.predict(dataloaders=val_dataloader,
                                                              ckpt_path=trainer.checkpoint_callback.best_model_path))
            val_labels = torch.cat(val_labels)
            val_prediction = torch.cat(val_prediction)
            print("val anomaly F1, roc_auc_score, AP, macro_f1, total_labels",
                  evaluate(val_labels,
                           val_prediction,
                           multilabel=self.multilabel))
        if test_mask is not None:
            # if self.transform_x:
            bulk_data = Qs
            # test_data = [features[test_mask], labels[test_mask]]
            test_data = [features, labels]
            test_dataloader = FCDataloaderInitializer().build_dataloader_non_graph(*test_data,
                                                                                   bulk_data=[bulk_data, test_mask, features_numpy],
                                                                                   # batch_size=10000,
                                                                                   batch_size=None,
                                                                                   shuffle=False,
                                                                                   train_val_test="test")
            start_testing_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
            test_labels, test_prediction = zip(*trainer.predict(dataloaders=test_dataloader,
                                                                ckpt_path=trainer.checkpoint_callback.best_model_path))
            end_testing_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
            print(f"Testing time elapsed: {end_testing_time - start_testing_time} sec")

            # if len(labels.shape) > 1 and labels.shape[1] > 1:
            test_labels = torch.cat(test_labels)
            test_prediction = torch.cat(test_prediction)
            print("test anomaly F1, roc_auc_score, AP, macro_f1, micro_f1, total_labels",
                  test_results := evaluate(test_labels,
                                           test_prediction,
                                           multilabel=self.multilabel), flush=True)

        return test_results
