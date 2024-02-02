import dgl
import dgl.function as fn
import scipy.sparse
from dgl.data import DGLDataset
import torch
import CommonModules as CM
import os
import collections
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import DistilBertTokenizerFast, DistilBertModel
from sentence_transformers import SentenceTransformer

from global_parameters import TransactionEdgeToNodeDataFolder, MAX_SEQ_LEN, LANGUAGE_EMBEDDING_CACHE_DIR, find_gpus
from icecream import ic


# Refer to https://docs.dgl.ai/en/latest/new-tutorial/6_load_data.html
# And https://github.com/minhduc0711/gcn/blob/2d09dd2f5c5400e4b53f74e2b35a5a612904ae48/src/data/datasets.py
class EdgeAttributed(DGLDataset):
    def __init__(self, name, data_folder, subset=None, no_node_features=False,
                 initialized_instance=None, add_self_loop=True,
                 randomize_train_test=False, randomize_by_node=False, is_bipartite=False,
                 process_text=False, use_text_cache=True, num_classes=None, sample_way=None, sample_size=None,
                 reduced_dim=None, use_linegraph=False, as_directed=True, as_dynamic=False, test_portion_index=9,
                 edge_to_node=False, force_binary_label_mapper=None, **kwargs):
        """

        :param name:
        :param data_folder:
        :param subset:
        :param no_node_features:
        :param initialized_instance:
        :param add_self_loop:
        :param randomize_train_test:
        :param randomize_by_node:
        :param is_bipartite:
        :param process_text:
        :param use_text_cache:
        :param num_classes: Only select the first num_class columns of the labels (in _citation.py, the columns are
        sorted using column sum. So the first num_class columns have the most label entries)
        :param sample_way:
        :param sample_size:
        :param reduced_dim:
        :param use_linegraph:
        :param as_directed:
        :param as_dynamic:
        :param test_portion_index:
        :param force_binary_label_mapper: A dict to indicate how to convert multiclass label into binary class label. E.g, {"-1": 0, "0": 0, "1": 1, "other_key": 0}. You must have a label vectorizer in the dataset to do this.
        """
        # This will introduce zero attributed edges, which will affect training process like batch normalization
        self.add_self_loop = add_self_loop
        self.randomize_train_test = randomize_train_test
        self.randomize_by_node = randomize_by_node
        self.is_bipartite = is_bipartite
        self.process_text = process_text
        self.use_text_cache = use_text_cache
        self.num_classes = num_classes
        self.use_linegraph = use_linegraph
        self.label_names = None
        self.as_directed = as_directed
        self.edge_to_node = edge_to_node
        self.end_node_same_type = True  # Used in edge to node graph
        self.bidirectional = True  # Used in edge to node graph
        self.force_binary_label_mapper = force_binary_label_mapper
        self.as_dynamic = as_dynamic  # Split the dataset into 10 portions as 1 train/1 val/8 test
        # When as_dynamic, generate the graph using the test_portion_number
        # E.g., When test_portion_index==2, there will be 1 train/1 val/1 test
        # When test_portion_index==9, there will be 1 train/1 val/8 test
        if self.as_dynamic:
            assert 2 <= test_portion_index <= 9
            assert not self.randomize_train_test
            assert not self.randomize_by_node
        self.test_portion_index = test_portion_index
        if add_self_loop and is_bipartite:
            # warnings.warn("Parameters add_self_loop and is_bipartite should not be true at the same time!")
            raise RuntimeError("Parameters add_self_loop and is_bipartite should not be true at the same time!")
        if initialized_instance is None:
            if reduced_dim:
                data_folder = os.path.join(data_folder, f"reduce_dim_{reduced_dim}")
            else:
                data_folder = os.path.join(data_folder, "base")
            if sample_size:
                data_folder = data_folder + f"_{sample_way}_sampled_size={sample_size}"
            else:
                data_folder = data_folder
            self.df_train = pd.read_csv(os.path.join(data_folder, "train_edges.csv"))
            self.df_val = pd.read_csv(os.path.join(data_folder, "val_edges.csv"))
            self.df_test = pd.read_csv(os.path.join(data_folder, "test_edges.csv"))
            if CM.IO.FileExist(os.path.join(data_folder, "label_vectorizer.pkl")):
                label_vectorizer = CM.IO.ImportFromPkl(os.path.join(data_folder, "label_vectorizer.pkl"))
                if label_vectorizer is not None:
                    self.label_names = [str(name) for name in label_vectorizer.classes]
                    if self.num_classes is not None:
                        self.label_names = self.label_names[:self.num_classes]
            elif force_binary_label_mapper is not None:
                raise NotImplementedError
            if force_binary_label_mapper is not None:
                assert (sum(force_binary_label_mapper.values()) == 1) and (1 in force_binary_label_mapper.values())
                self.num_classes = None
                for key in force_binary_label_mapper:
                    if force_binary_label_mapper[key] == 1:
                        self._force_binary_label_index = self.label_names.index(str(key))
                        self.label_names = [self.label_names[self._force_binary_label_index]]
            else:
                self._force_binary_label_index = None
            if self.process_text:
                if self.use_text_cache and CM.IO.FileExist(os.path.join(data_folder, "train_embeds_text_labels.pkl")):
                    self.train_embeds, self.train_labels = CM.IO.ImportFromPkl(
                        os.path.join(data_folder, "train_embeds_text_labels.pkl"))
                    self.val_embeds, self.val_labels = CM.IO.ImportFromPkl(
                        os.path.join(data_folder, "val_embeds_text_labels.pkl"))
                    self.test_embeds, self.test_labels = CM.IO.ImportFromPkl(
                        os.path.join(data_folder, "test_embeds_text_labels.pkl"))
            if (not self.process_text) or (not self.use_text_cache) or \
                    (not CM.IO.FileExist(os.path.join(data_folder, "train_embeds_text_labels.pkl"))):
                self.train_embeds, self.train_labels = CM.IO.ImportFromPkl(
                    os.path.join(data_folder, "train_embeds_labels.pkl"))
                self.val_embeds, self.val_labels = CM.IO.ImportFromPkl(
                    os.path.join(data_folder, "val_embeds_labels.pkl"))
                self.test_embeds, self.test_labels = CM.IO.ImportFromPkl(
                    os.path.join(data_folder, "test_embeds_labels.pkl"))
                if (not self.use_text_cache and self.process_text) or \
                    (not CM.IO.FileExist(os.path.join(data_folder, "train_embeds_text_labels.pkl")) and self.process_text):
                    self.train_embeds, self.val_embeds, self.test_embeds = self._process_text()  # Process text using
                    # LLM
                    CM.IO.ExportToPkl(os.path.join(data_folder, "train_embeds_text_labels.pkl"), [self.train_embeds, self.train_labels])
                    CM.IO.ExportToPkl(os.path.join(data_folder, "val_embeds_text_labels.pkl"), [self.val_embeds, self.val_labels])
                    CM.IO.ExportToPkl(os.path.join(data_folder, "test_embeds_text_labels.pkl"), [self.test_embeds, self.test_labels])
            if len(self.train_labels.shape) > 1:
                if self.force_binary_label_mapper:
                    self.train_labels = self.train_labels[:, self._force_binary_label_index]
                    self.val_labels = self.val_labels[:, self._force_binary_label_index]
                    self.test_labels = self.test_labels[:, self._force_binary_label_index]
                else:
                    label_column_mask = np.array(self.train_labels.sum(0) > 10).reshape(-1)  # Exclude class with too
                    # few samples
                    self.train_labels = self.train_labels[:, label_column_mask]  # Exclude class with too few samples
                    self.val_labels = self.val_labels[:, label_column_mask]  # Exclude class with too few samples
                    self.test_labels = self.test_labels[:, label_column_mask]  # Exclude class with too few samples
            if self.edge_to_node:
                name = name.replace("EdgeAttributed", "EdgeToNode")
            super().__init__(name=name)
        else:
            self = initialized_instance
        if self.edge_to_node:
            if subset == "train":
                self.nodes_mask = self.graph.ndata["train_mask"]
            elif subset == "val":
                self.nodes_mask = self.graph.ndata["val_mask"]
            elif subset == "test":
                self.nodes_mask = self.graph.ndata["test_mask"]
            else:
                self.nodes_mask = None
            if no_node_features:  # Exclude any node features
                self.nfeats = torch.eye(self.graph.num_nodes())
            else:
                self.nfeats = self.graph.ndata["nfeat"]
        else:
            if subset == "train":
                self.edges_mask = self.graph.edata["train_mask"]
            elif subset == "val":
                self.edges_mask = self.graph.edata["val_mask"]
            elif subset == "test":
                self.edges_mask = self.graph.edata["test_mask"]
            else:
                self.edges_mask = None
            if no_node_features:  # Exclude any node features
                self.nfeats = None
            self.nodes_mask = None

    def _concatenate_dataset(self):
        df = pd.concat([self.df_train, self.df_val, self.df_test])
        if isinstance(self.train_embeds, (np.ndarray, torch.Tensor)):
            embeds = np.concatenate([self.train_embeds, self.val_embeds, self.test_embeds], axis=0)
        elif scipy.sparse.issparse(self.train_embeds):
            embeds = scipy.sparse.vstack([self.train_embeds, self.val_embeds, self.test_embeds])
        # elif isinstance(self.train_embeds, list):  # Python list
        #     embeds = self.train_embeds + self.val_embeds + self.test_embeds
        else:
            raise TypeError
        if isinstance(self.train_labels, (np.ndarray, torch.Tensor)):
            labels = np.concatenate([self.train_labels, self.val_labels, self.test_labels], axis=0)
        else:
            labels = scipy.sparse.vstack([self.train_labels, self.val_labels, self.test_labels])
        return df, embeds, labels

    def _process_text(self):
        # tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased",
        #                                                      do_lower_case=True,
        #                                                      cache_dir=LANGUAGE_EMBEDDING_CACHE_DIR)
        # encoder = DistilBertModel.from_pretrained("distilbert-base-uncased",
        #                                           cache_dir=LANGUAGE_EMBEDDING_CACHE_DIR)
        # gpus, accelerator = find_gpus(3)
        # # device = torch.device(f"cuda:{','.join([str(gpu) for gpu in sorted(gpus)])}")
        # device = torch.device(f"cuda:{gpus[0]}")
        # encoder = torch.nn.DataParallel(encoder, device_ids=gpus)
        # encoder.to(device)
        # # encoder = encoder.to(f"cuda:{gpus[0]}")
        # encoder = SentenceTransformer('all-mpnet-base-v2')
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        batch_size = 500
        encoded_embeds = []
        for texts in [self.train_embeds, self.val_embeds, self.test_embeds]:
            texts = texts.tolist()
            print("Example text for _process_text:", texts[0])
            sentence_embeddings = encoder.encode(texts, batch_size=batch_size, show_progress_bar=True,
                                                 convert_to_numpy=True)
            encoded_embeds.append(sentence_embeddings)
        #     sorted_original_index = np.array(sorted(list(range(len(texts))), key=lambda x: len(texts[x])))
        #     texts = sorted(texts, key=lambda x: len(x))
        #     outputs = []
        #     with torch.no_grad():
        #         for i in tqdm(range(0, len(texts), batch_size)):
        #         # for i in range(0, 10000, 500):
        #             text_encoded_dict = tokenizer.batch_encode_plus(
        #                 texts[i:i + batch_size],  # Sentence to encode.
        #                 add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        #                 max_length=MAX_SEQ_LEN,  # Pad & truncate all sentences.
        #                 padding='longest',
        #                 # padding=True,
        #                 truncation=True,
        #                 return_attention_mask=True,  # Construct attn. masks.
        #                 return_tensors='pt',  # Return pytorch tensors.
        #             )
        #             output = encoder(input_ids=text_encoded_dict.input_ids.to(device),
        #              attention_mask=text_encoded_dict.attention_mask.to(device))
        #             output = output.last_hidden_state[:, 0, :]
        #             outputs.append(output.cpu())
        #     concatenated_outputs = torch.cat(outputs).numpy()[sorted_original_index]
        #     encoded_embeds.append(concatenated_outputs)
        # del encoder, tokenizer
        del encoder
        torch.cuda.empty_cache()
        return encoded_embeds


    def _process_edge_to_node_graph(self, df, edge_features, edge_labels):
        edges_src = torch.from_numpy(df['SENDER_id'].to_numpy())
        edges_dst = torch.from_numpy(df['RECEIVER_id'].to_numpy())
        if len(edge_labels.shape) == 1:
            edge_labels = edge_labels.reshape(-1, 1)
        if self.end_node_same_type:
            sender_key = receiver_key = "user"
        else:
            edges_src = edges_src.unique(return_inverse=True)[1]  # Re-encode SENDER_id to start from 1
            edges_dst = edges_dst.unique(return_inverse=True)[1]  # Re-encode RECEIVER_id to start from 1
            sender_key = "sender"
            receiver_key = "receiver"
        # self.edge_start_index = max(max(edges_src), max(edges_dst)) + 1
        edge_node_index = torch.arange(edge_features.shape[0])
        #         if self.homo:
        self.graph: dgl.DGLGraph
        if self.bidirectional:
            self.graph = dgl.heterograph({(sender_key, "send", "transaction"): (edges_src, edge_node_index),
                                          ("transaction", "sent-by", sender_key): (edge_node_index, edges_src),
                                          (receiver_key, "receive", "transaction"): (edges_dst, edge_node_index),
                                          ("transaction", "received-by", receiver_key): (edge_node_index, edges_dst)
                                          })
        else:
            self.graph = dgl.heterograph({(sender_key, "send", "transaction"): (edges_src, edge_node_index),
                                          ("transaction", "received-by", receiver_key): (edge_node_index, edges_dst)
                                          })
        #         self.graph = dgl.graph((edge_node_index, edges_dst))
        assert (self.graph.ntypes == ["receiver", "sender", "transaction"]) or (
                    self.graph.ntypes == ["transaction", "user"])
        print(edge_features.shape, edges_src.shape, edge_node_index.shape)
        if self.end_node_same_type:
            self.edge_start_index = max(max(edges_src), max(edges_dst)) + 1
            self.graph.nodes[sender_key].data['nfeat'] = torch.zeros(self.edge_start_index, self.train_embeds.shape[1])
            self.graph.nodes["transaction"].data['nfeat'] = edge_features
            edge_label_shape = list(edge_labels.shape)
            edge_label_shape[0] = self.edge_start_index
            self.graph.nodes[sender_key].data['label'] = torch.zeros(edge_label_shape, dtype=torch.long)
            self.graph.nodes["transaction"].data['label'] = edge_labels
        else:
            self.graph.nodes[sender_key].data['nfeat'] = torch.zeros(edges_src.max() + 1, self.train_embeds.shape[1])
            self.graph.nodes[receiver_key].data['nfeat'] = torch.zeros(edges_dst.max() + 1, self.train_embeds.shape[1])
            self.graph.nodes["transaction"].data['nfeat'] = edge_features
            edge_label_shape = list(edge_labels.shape)
            edge_label_shape[0] = edges_src.max() + 1
            self.graph.nodes[sender_key].data['label'] = torch.zeros(edge_label_shape, dtype=torch.long)
            edge_label_shape = list(edge_labels.shape)
            edge_label_shape[0] = edges_dst.max() + 1
            self.graph.nodes[receiver_key].data['label'] = torch.zeros(edge_label_shape, dtype=torch.long)
            self.graph.nodes["transaction"].data['label'] = edge_labels
        #         self.graph.ndata[dgl.NTYPE] = node_types
        #         self.graph.edata['efeat'] = edge_features
        #         self.graph.edata['label'] = edge_labels
        #         self.graph.update_all(fn.copy_e('efeat', 'm'), fn.mean('m', 'nfeat')) # i.e. self.graph.ndata["nfeat"]
        #         self.graph = dgl.add_self_loop(self.graph)  # only 1 graph in dataset
        self.edges_id = torch.arange(self.graph.number_of_edges())
        #         self.edges_src, self.edges_dst = self.graph.find_edges(self.edges_id)

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        #         m_edges = self.graph.number_of_edges()
        n_nodes_transaction = self.graph.number_of_nodes(ntype="transaction")
        n_train = self.train_embeds.shape[0]
        n_val = self.val_embeds.shape[0]
        n_test = self.test_embeds.shape[0]
        train_mask = torch.zeros(n_nodes_transaction, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes_transaction, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes_transaction, dtype=torch.bool)
        edge_start_index = 0
        train_mask[edge_start_index:edge_start_index + n_train] = True
        val_mask[edge_start_index + n_train:edge_start_index + n_train + n_val] = True
        test_mask[edge_start_index + n_train + n_val:edge_start_index + n_train + n_val + n_test] = True
        self.graph.nodes["transaction"].data['train_mask'] = train_mask
        self.graph.nodes["transaction"].data['val_mask'] = val_mask
        self.graph.nodes["transaction"].data['test_mask'] = test_mask
        eid = torch.arange(n_nodes_transaction, dtype=torch.int)
        self.graph.nodes["transaction"].data['eid'] = eid
        for ntype in [sender_key, receiver_key]:
            n_nodes_ntype = self.graph.number_of_nodes(ntype=ntype)
            self.graph.nodes[ntype].data['train_mask'] = torch.zeros(n_nodes_ntype, dtype=torch.bool)
            self.graph.nodes[ntype].data['val_mask'] = torch.zeros(n_nodes_ntype, dtype=torch.bool)
            self.graph.nodes[ntype].data['test_mask'] = torch.zeros(n_nodes_ntype, dtype=torch.bool)
            eid = torch.zeros(n_nodes_ntype, dtype=torch.int)
            eid[:] = -1
            self.graph.nodes[ntype].data['eid'] = eid
        # if self.homo:
        self.graph = dgl.to_homogeneous(self.graph, ndata=["nfeat", "label", 'train_mask', 'val_mask',
                                                           'test_mask', "eid"])
        # if self.bidirectional:
        self.graph = self.graph.add_self_loop()
        l = self.graph.ndata["eid"][:n_nodes_transaction]
        # Make sure in edge to node graph, the eids for transactions are sorted
        assert all(l[i] <= l[i + 1] for i in range(len(l) - 1))
        # else:
        #     pass

    def _process_edge_attributed_graph(self, df, edge_features, edge_labels):
        edges_src = torch.from_numpy(df['SENDER_id'].to_numpy())
        edges_dst = torch.from_numpy(df['RECEIVER_id'].to_numpy())

        self.graph = dgl.graph((edges_src, edges_dst))
        ic(self.graph.number_of_nodes(), edge_features.shape, edges_src.shape, edges_dst.shape, edge_labels.shape)
        #         self.graph.ndata[dgl.NTYPE] = node_types
        self.efeats = edge_features
        if isinstance(edge_features, (np.ndarray, torch.Tensor)) and self.train_embeds.dtype != object:
            self.graph.edata['efeat'] = edge_features
            self.graph.update_all(fn.copy_e('efeat', 'm'), fn.mean('m', 'nfeat'))  # i.e. self.graph.ndata["nfeat"]
            self.nfeats = self.graph.ndata["nfeat"]
        else:
            self.nfeats = self.efeats  # Dummy nfeats! Should never be used!
        if len(edge_labels.shape) == 1:
            edge_labels = edge_labels.reshape(-1, 1)
        if isinstance(edge_labels, (np.ndarray, torch.Tensor)):
            self.graph.edata['label'] = edge_labels
        self.labels = edge_labels
        if self.add_self_loop:
            self.graph = dgl.add_self_loop(self.graph)  # only 1 graph in dataset
            self.labels = self.graph.edata["label"]
            self.efeats = self.graph.edata["efeat"]
        self.edges_id = torch.arange(self.graph.number_of_edges())
        # self.edges_src, self.edges_dst = self.graph.find_edges(self.edges_id)

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        #         m_edges = self.graph.number_of_edges()
        n_edges = self.graph.number_of_edges()
        n_train = self.train_embeds.shape[0]
        n_val = self.val_embeds.shape[0]
        n_test = self.test_embeds.shape[0]
        train_mask = torch.zeros(n_edges, dtype=torch.bool)
        val_mask = torch.zeros(n_edges, dtype=torch.bool)
        test_mask = torch.zeros(n_edges, dtype=torch.bool)
        edge_start_index = 0
        train_mask[edge_start_index:edge_start_index + n_train] = True
        val_mask[edge_start_index + n_train:edge_start_index + n_train + n_val] = True
        test_mask[edge_start_index + n_train + n_val:edge_start_index + n_train + n_val + n_test] = True
        self.graph.edata['train_mask'] = train_mask
        self.graph.edata['val_mask'] = val_mask
        self.graph.edata['test_mask'] = test_mask
        if self.is_bipartite:
            assert not self.add_self_loop
            assert len(set(edges_src.tolist()) & set(edges_dst.tolist())) == 0  # Ensure no overlap between src and dst
            self.graph.ndata["_bipartite_indicator"] = torch.zeros(self.graph.number_of_nodes(), dtype=int)
            self.graph.ndata["_bipartite_indicator"][edges_dst] = 1
            self.graph.edata["_bipartite_indicator"] = torch.zeros(self.graph.number_of_edges(), dtype=int)
            self.graph = dgl.to_heterogeneous(self.graph, ["sender", "receiver"], ["bipartite_edge"],
                                              "_bipartite_indicator", "_bipartite_indicator")

    def process(self):
        print("Number of edges for original graph:", self.df_train.shape[0] + self.df_val.shape[0] + self.df_test.shape[0])
        print("Number of Nodes for original graph:",
              len(set(self.df_train['SENDER_id']) | set(self.df_train['RECEIVER_id']) |
                 set(self.df_val['SENDER_id']) | set(self.df_val['RECEIVER_id']) |
                  set(self.df_test['SENDER_id']) | set(self.df_test['RECEIVER_id'])))
        if self.randomize_train_test and not self.randomize_by_node:
            print("Randomly split dataset by edges...")
            df, embeds, labels = self._concatenate_dataset()
            if self.num_classes:
                labels = labels[:, :self.num_classes]
            if scipy.sparse.issparse(labels):
                labels = labels.A
            self.labels = labels
            index = np.arange(df.shape[0])
            train_index, test_index = train_test_split(index, test_size=0.2, shuffle=True, random_state=12)
            train_index, val_index = train_test_split(train_index, test_size=0.2, shuffle=True, random_state=12)
            df_train, df_val, df_test = df.iloc[train_index], df.iloc[val_index], df.iloc[test_index]
            self.train_embeds, self.val_embeds, self.test_embeds = embeds[train_index], embeds[val_index], embeds[test_index]
            self.train_labels, self.val_labels, self.test_labels = labels[train_index], labels[val_index], labels[test_index]
        elif self.randomize_by_node:
            print("Randomly split dataset by nodes...")
            df, embeds, labels = self._concatenate_dataset()
            if self.num_classes:
                labels = labels[:, :self.num_classes]
            if scipy.sparse.issparse(labels):
                labels = labels.A
            self.labels = labels
            # ###########DEBUG############
            # df = df.drop_duplicates(subset=['SENDER_id', 'RECEIVER_id'], ignore_index=True)
            # ############################
            all_node_list = np.array(list(set(df["SENDER_id"]) | set(df["RECEIVER_id"])))
            index = np.arange(len(all_node_list))
            node_train_index, node_test_index = train_test_split(index, test_size=0.2, shuffle=True, random_state=12)
            node_train_index, node_val_index = train_test_split(node_train_index, test_size=0.2, shuffle=True, random_state=12)
            train_index = (df["SENDER_id"].isin(all_node_list[node_train_index]) | df["RECEIVER_id"].isin(
                all_node_list[node_train_index])).to_numpy()
            val_index = ((df["SENDER_id"].isin(all_node_list[node_val_index]) | df["RECEIVER_id"].isin(
                all_node_list[node_val_index])) & ~train_index).to_numpy()
            test_index = ((df["SENDER_id"].isin(all_node_list[node_test_index]) | df["RECEIVER_id"].isin(
                all_node_list[node_test_index])) & ~train_index).to_numpy()
            df_train, df_val, df_test = df.iloc[train_index], df.iloc[val_index], df.iloc[test_index]
            self.train_embeds, self.val_embeds, self.test_embeds = embeds[train_index], embeds[val_index], embeds[test_index]
            self.train_labels, self.val_labels, self.test_labels = labels[train_index], labels[val_index], labels[test_index]
        elif self.as_dynamic:
            print("Split using dynamic graph logic...")
            df, embeds, labels = self._concatenate_dataset()
            if self.num_classes:
                labels = labels[:, :self.num_classes]
            if scipy.sparse.issparse(labels):
                labels = labels.A
            self.labels = labels
            index = np.arange(df.shape[0])
            train_index, test_index = train_test_split(index, test_size=0.2, shuffle=False)
            train_index, val_index = train_test_split(train_index, test_size=0.5, shuffle=False)
            if self.test_portion_index >= 9:
                pass
            else:
                test_index, _ = train_test_split(test_index, train_size=(self.test_portion_index-1)/8, shuffle=False)
            df_train, df_val, df_test = df.iloc[train_index], df.iloc[val_index], df.iloc[test_index]
            self.train_embeds, self.val_embeds, self.test_embeds = embeds[train_index], embeds[val_index], embeds[test_index]
            self.train_labels, self.val_labels, self.test_labels = labels[train_index], labels[val_index], labels[test_index]
        else:
            print("Not randomly split dataset...")
            df_train, df_val, df_test = self.df_train, self.df_val, self.df_test
            if self.num_classes:
                self.train_labels = self.train_labels[:, :self.num_classes]
                self.val_labels = self.val_labels[:, :self.num_classes]
                self.test_labels = self.test_labels[:, :self.num_classes]
            if scipy.sparse.issparse(self.train_labels):
                self.train_labels = self.train_labels.A
                self.val_labels = self.val_labels.A
                self.test_labels = self.test_labels.A
            self.train_labels, self.val_labels, self.test_labels = self.train_labels, self.val_labels, self.test_labels


        if isinstance(self.train_labels, (np.ndarray, torch.Tensor)) and len(self.train_labels.shape) > 1 and self.train_labels.shape[1] > 1:
            self.train_labels[self.train_labels > 0] = 1
            self.val_labels[self.val_labels > 0] = 1
            self.test_labels[self.test_labels > 0] = 1

            self.train_labels[self.train_labels < 0] = 0
            self.val_labels[self.val_labels < 0] = 0
            self.test_labels[self.test_labels < 0] = 0

        df = pd.concat([df_train, df_val, df_test], axis=0)
        if isinstance(self.train_embeds, np.ndarray) and self.train_embeds.dtype == object:
            edge_features = np.concatenate([self.train_embeds, self.val_embeds, self.test_embeds], axis=0)
        else:
            self.train_embeds = self.train_embeds.astype(np.float32)
            self.val_embeds = self.val_embeds.astype(np.float32)
            self.test_embeds = self.test_embeds.astype(np.float32)
            if isinstance(self.train_embeds, (np.ndarray, torch.Tensor)):
                self.train_embeds = torch.from_numpy(self.train_embeds)
                self.val_embeds = torch.from_numpy(self.val_embeds)
                self.test_embeds = torch.from_numpy(self.test_embeds)
                edge_features = torch.cat([self.train_embeds, self.val_embeds, self.test_embeds])
            else:
                edge_features = scipy.sparse.vstack([self.train_embeds, self.val_embeds, self.test_embeds])
        if isinstance(self.train_labels, (np.ndarray, torch.Tensor)):
            self.train_labels = torch.from_numpy(self.train_labels).long()
            self.val_labels = torch.from_numpy(self.val_labels).long()
            self.test_labels = torch.from_numpy(self.test_labels).long()
            edge_labels = torch.cat([self.train_labels, self.val_labels, self.test_labels])
        else:
            edge_labels = scipy.sparse.vstack([self.train_labels, self.val_labels, self.test_labels])
        if self.edge_to_node:
            self._process_edge_to_node_graph(df, edge_features, edge_labels)
        else:
            self._process_edge_attributed_graph(df, edge_features, edge_labels)


    def __getitem__(self, i):
        if self.edge_to_node:
            return {
                "g": self.graph,
                #             "efeats": self.efeats,
                "nfeats": self.nfeats,
                "labels": self.graph.ndata["label"],
                "nodes_mask": self.nodes_mask,
                "edges_id": self.edges_id,
                #             "edges_src": self.edges_src,
                #             "edges_dst": self.edges_dst
                "end_node_same_type": self.end_node_same_type,
            }
        else:
            return {
                "g": self.graph,
                "efeats": self.efeats,
                "nfeats": self.nfeats,  # Dummy nfeats! Should never be used!
                "labels": self.labels,
                "edges_mask": self.edges_mask,
                "edges_id": self.edges_id,
                "nodes_mask": self.nodes_mask,
                # "edges_src": self.edges_src,
                # "edges_dst": self.edges_dst,
                "label_names": self.label_names
            }

    def __len__(self):
        return 1
