import datetime
from copy import copy

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import os
import pytorch_lightning as pl

from global_parameters import max_epochs, print_result_for_pl, INPUT_EMBEDDING_DIMENSION, find_gpus, log_directory

import CommonModules as CM

csv_logger = CSVLogger(".", name='log', version=None, prefix='')


def train(dataset_class, dataloader_class, model_class, model_parameter_dict: dict, logger, checkpoint_callback,
          use_pl, training_phase=None, previous_phase_checkpoint=None, initialize_dataset=True):
    model_parameter_dict["training_phase"] = training_phase
    if initialize_dataset:
        dataset = dataset_class()
        train_dataset, val_dataset, test_dataset = copy(dataset), copy(dataset), copy(dataset)
        # Will modify initialized_instance in __init__
        dataset_class(subset="train", initialized_instance=train_dataset)
        dataset_class(subset="val", initialized_instance=val_dataset)
        dataset_class(subset="test", initialized_instance=test_dataset)
        if isinstance(train_dataset[0]["nfeats"], dict):
            nfeats = train_dataset[0]["nfeats"]["SENDER"]
        else:
            nfeats = train_dataset[0]["nfeats"]
        if len(nfeats.shape) > 1:
            in_nfeats = nfeats.shape[1]
        else:
            in_nfeats = INPUT_EMBEDDING_DIMENSION
    else:
        train_dataset, val_dataset, test_dataset = None, None, None  # The model will use dataset class to get instances
        in_nfeats = None

    # gpus = -1
    gpus, accelerator = find_gpus(1, model_parameter_dict)
    # import ipdb; ipdb.set_trace()

    if dataloader_class is not None:  # In case you don't use dataloaders in your fitting code
        dataloader_intializing_instance = dataloader_class(device=gpus)
        train_dataloader = dataloader_intializing_instance.build_dataloader(train_dataset, train_val_test="train")
        val_dataloader = dataloader_intializing_instance.build_dataloader(
            val_dataset, train_val_test="val",
            existing_dataloader_intializing_instance=dataloader_intializing_instance)
        test_dataloader = dataloader_intializing_instance.build_dataloader(
            test_dataset, train_val_test="test",
            existing_dataloader_intializing_instance=dataloader_intializing_instance)
    else:
        train_dataloader, val_dataloader, test_dataloader = None, None, None
    if model_parameter_dict.get("n_classes", None) == "auto":  # Infer n_classes during run time
        if len(train_dataset.train_labels.shape) > 1:
            n_classes = train_dataset.train_labels.shape[1]
        else:
            n_classes = 1
        model_parameter_dict["n_classes"] = n_classes  # Slightly conflict with labelsize. Should fix in the future
    try:  # For edge attributed models
        if previous_phase_checkpoint:
            model = model_class.load_from_checkpoint(previous_phase_checkpoint, in_nfeats=in_nfeats,
                                                     in_efeats=in_nfeats, labelsize=1,
                                                     **model_parameter_dict)
        else:
            model = model_class(in_nfeats=in_nfeats, in_efeats=in_nfeats, labelsize=1, **model_parameter_dict)
    except TypeError:
        if previous_phase_checkpoint:
            model = model_class.load_from_checkpoint(previous_phase_checkpoint, in_nfeats=in_nfeats,
                                                     labelsize=1, **model_parameter_dict)
        else:
            model = model_class(in_nfeats=in_nfeats, labelsize=1, **model_parameter_dict)
    # if previous_phase_checkpoint:
    #     model: BaseModel
    #     model = model.load_from_checkpoint(previous_phase_checkpoint)
    if use_pl:
        if training_phase != "discrimination":
            earlystopping_callback = EarlyStopping(monitor='val/loss_epoch',
                                                   min_delta=0.00,
                                                   # patience=30,
                                                   patience=30,
                                                   verbose=True,
                                                   mode='min')
            callbacks = [checkpoint_callback, earlystopping_callback]
        else:
            callbacks = [checkpoint_callback]
        trainer = pl.Trainer(max_epochs=max_epochs, accelerator=accelerator, devices=gpus,
                             callbacks=callbacks,
                             logger=[logger, csv_logger], check_val_every_n_epoch=10, profiler=False,
                             precision=32)
        # Because there may be another call inside fit, we define these variables to record time
        start_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
        # if model_parameter_dict["structure"] == "GAT":
        #     import ipdb; ipdb.set_trace()
        trainer.fit(model, train_dataloader, [val_dataloader, test_dataloader])
        end_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
        if training_phase:
            print(f"Training time elapsed for phase {training_phase}: {end_training_time - start_training_time} sec")
        else:
            print(f"Training time elapsed: {end_training_time - start_training_time} sec")
        print(checkpoint_callback.best_model_path)
        graph_dataloaders = train_dataloader, val_dataloader, test_dataloader
        return trainer, checkpoint_callback, graph_dataloaders, None, model
    else:  # Let the model handle all the training
        graph_features = model.get_static_graph_features([train_dataloader, val_dataloader, test_dataloader], gpus=gpus)
        # return None, None, None, None, None  # Used to generate cached dataset
        # Sometimes the trainer is inside fit. If not, trainer should be None
        trainer = model.fit(graph_features, max_epochs=max_epochs,
                            gpus=gpus, dataset_class=dataset_class, logger=[logger, csv_logger])
        return trainer, None, None, graph_features, model


def test(trainer, checkpoint_callback, graph_dataloaders, graph_features, model, use_pl):
    if use_pl:
        train_dataloader, val_dataloader, test_dataloader = graph_dataloaders
        test_results = print_result_for_pl(trainer, checkpoint_callback, train_dataloader, val_dataloader,
                                           test_dataloader)
    else:
        test_results = model.test(trainer, graph_features)
    return test_results


def main(logger_name, dataset_class, dataloader_class, model_class, model_parameter_dict, use_pl=True,
         initialize_dataset=True, as_dynamic=False):
    """

    :param logger_name:
    :param dataset_class:
    :param dataloader_class:
    :param model_class:
    :param model_parameter_dict:
    :param use_pl:
    :param initialize_dataset: Whether automatically initialize dataset or let the model handle the dataset class
    initialization
    :return:
    """
    pl.seed_everything(12)
    logger = TensorBoardLogger(log_directory, name=logger_name, version=None)
    # logger = TensorBoardLogger("logs_20220831", name=logger_name, version=None)
    previous_phase_checkpoint = None
    if "training_phases" in model_parameter_dict:
        training_phases = model_parameter_dict["training_phases"]  # e.g. ['discrimination', 'classification']
    else:
        training_phases = [None]
    for training_phase_index, training_phase in enumerate(training_phases):
        if training_phase_index >= 1:
            model_parameter_dict["training_phase"] = training_phase
            previous_phase_checkpoint = checkpoint_callback.best_model_path
            # model_class_instance = model_class_instance.load_from_checkpoint(checkpoint_callback.best_model_path,
            #                                                                  **model_parameter_dict)
        if training_phase == "discrimination":
            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join("./trained_models", logger.name, str(datetime.datetime.now())),
                filename='epoch={epoch}-train_loss_epoch={'
                         'train/loss_epoch:.6f}',
                # monitor='val/best_macro_f1_epoch', mode="max", save_last=True,
                # monitor='val/auc_epoch', mode="max", save_last=True,
                # monitor='val/macro_f1_epoch', mode="max", save_last=True,
                # monitor='val/ap_epoch', mode="max", save_last=True,
                monitor='train/loss_epoch', mode="min", save_last=True,
                auto_insert_metric_name=False)
        else:
            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join("./trained_models", logger.name, str(datetime.datetime.now())),
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
        # if as_dynamic:
        #     dataset_class = wrapped_partial(dataset_class, as_dynamic=True, test_portion_index=2)
        trainer, checkpoint_callback, graph_dataloaders, graph_features, model_class_instance = \
            train(dataset_class, dataloader_class, model_class, model_parameter_dict, logger, checkpoint_callback,
                  use_pl, training_phase=training_phase, previous_phase_checkpoint=previous_phase_checkpoint,
                  initialize_dataset=initialize_dataset)
    # return # Used to generate cached dataset
    if as_dynamic:
        if graph_features:
            old_graph_features = graph_features
        else:
            _, _, old_test_dataloader = graph_dataloaders
        test_results = test(trainer, checkpoint_callback, graph_dataloaders, graph_features, model_class_instance, use_pl)
        for test_portion_index in range(2, 10):
            print("Current test_portion_index:", test_portion_index)
            # as_dynamic should already be True for dataset_class
            test_dataset = dataset_class(subset="test", test_portion_index=test_portion_index)
            gpus, _ = find_gpus(1, model_parameter_dict)
            new_test_dataloader = dataloader_class(device=gpus).build_dataloader(test_dataset, train_val_test="test")
            if graph_features:
                start_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
                new_graph_features = model_class_instance.get_dynamic_graph_features(
                    new_test_dataloader, old_graph_features)
                end_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
                print(f"get_dynamic_graph_features time elapsed: {end_time - start_time} sec")
                graph_features |= new_graph_features
                # import ipdb; ipdb.set_trace()
                start_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
                test_results = test(trainer, checkpoint_callback, graph_dataloaders, graph_features, model_class_instance, use_pl)
                end_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
                print(f"test_dynamic_graph time elapsed: {end_time - start_time} sec")
                old_graph_features = graph_features
            else:
                graph_dataloaders = None, None, new_test_dataloader
                start_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
                test_results = test(trainer, checkpoint_callback, graph_dataloaders, graph_features,
                                    model_class_instance, use_pl)
                end_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
                print(f"test_dynamic_graph time elapsed: {end_time - start_time} sec")
    else:
        test_results = test(trainer, checkpoint_callback, graph_dataloaders, graph_features, model_class_instance, use_pl)

    return test_results


