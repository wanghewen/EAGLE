from functools import partial
from torch.utils.data import DataLoader, TensorDataset, Dataset, default_collate
from icecream import ic


class ConcatDataset(Dataset):
    # Refer to https://discuss.pytorch.org/t/combine-two-dataloaders/156164/7
    # Used to concat an iterative dataset with a bulk dataset
    def __init__(self, iterative_dataset, fixed_dataset=None):
        self.iterative_dataset = iterative_dataset
        self.fixed_dataset = fixed_dataset

    def __getitem__(self, index):
        a = self.iterative_dataset[index % len(self.iterative_dataset)]
        b = self.fixed_dataset[:]
        return a, b

    def __len__(self):
        return len(self.iterative_dataset)


def collate_fn(batch, bulk_data=None):
    batch = default_collate(batch)
    if bulk_data is not None:
        batch = (batch, bulk_data)
    return batch


class FCDataloaderInitializer():
    @staticmethod
    def build_dataloader(dataset, batch_size=None, train_val_test=None, device="cpu"):
        dataset = dataset[0]
        embeds, labels, nodes_mask = dataset["nfeats"], dataset["labels"], dataset["nodes_mask"]
        # train_embeds, train_labels = CM.IO.ImportFromPkl(os.path.join(data_folder, "train_embeds_labels.pkl"))
        # val_embeds, val_labels = CM.IO.ImportFromPkl(os.path.join(data_folder, "val_embeds_labels.pkl"))
        # test_embeds, test_labels = CM.IO.ImportFromPkl(os.path.join(data_folder, "test_embeds_labels.pkl"))
        # embeds, labels = embeds[nodes_mask].to(device), labels[nodes_mask].to(device)
        embeds, labels = embeds[nodes_mask], labels[nodes_mask]
        ic(embeds.shape, labels.shape)
        # embeds = embeds.astype(np.float32)
        labels[labels > 0] = 1
        dataset = TensorDataset(embeds, labels)

        if batch_size is None:
            batch_size = len(dataset)
            if train_val_test == "train":
                num_workers = 10
                persistent_workers = True
            else:
                num_workers = 0
                persistent_workers = False
            dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True,
                                    num_workers=num_workers, shuffle=False, persistent_workers=persistent_workers)

        return dataloader


    @staticmethod
    def build_dataloader_non_graph(*data, bulk_data=None, batch_size=None, shuffle=False, train_val_test=None):
        data_shapes = [element.shape for element in data]
        ic(data_shapes)
        # embeds = embeds.astype(np.float32)
        # if bulk_data is not None:
        #     # Refer to https://pytorch.org/docs/stable/data.html#torch.utils.data.default_collate
        #     bulk_dataset = TensorDataset(bulk_data)
        #     dataset = ConcatDataset(dataset, fixed_dataset=bulk_dataset)

        if batch_size is None:
            # batch size = 1, To increase data loader's efficiency, as it's (list comprehension) heavily depend on the
            data = [element.unsqueeze(0) for element in data]
            pass
        dataset = TensorDataset(*data)
        batch_size = len(dataset)

        if train_val_test == "train" and batch_size > 1:
            num_workers = 20
            prefetch_factor = 2
            persistent_workers = True
            dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, prefetch_factor=prefetch_factor,
                                    num_workers=num_workers, shuffle=shuffle, persistent_workers=persistent_workers,
                                    collate_fn=partial(collate_fn, bulk_data=bulk_data))
        else:
            num_workers = 0
            prefetch_factor = None
            persistent_workers = False
            dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True,
                                    num_workers=num_workers, shuffle=shuffle, persistent_workers=persistent_workers,
                                    collate_fn=partial(collate_fn, bulk_data=bulk_data))

        return dataloader

