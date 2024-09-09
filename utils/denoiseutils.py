# -*- coding: utf-8 -*-
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
import recbole_gnn.recbole_gnn.utils as gnnutils
from recbole_gnn.recbole_gnn.config import Config as GNNConfig
from trainer import *


def load_config(args):
    if args.backbone not in ["SGL", "NCL"]:
        config = Config(
            model=args.backbone,
            dataset=args.dataset,
            config_file_list=[
                'config/base.yaml',                        # basic settings
                f'config/{args.dataset}.yaml',             # dataset settings (we put our prompt template here)
                f'config/{args.backbone.lower()}.yaml',    # backbone settings (backbone hyperparameters)
                f'config/llmhd.yaml'                       # our model setting
            ]
        )
    elif args.backbone in ["SGL", "NCL"]:
        config = GNNConfig(
            model=args.backbone,
            dataset=args.dataset,
            config_file_list=[
                'config/base.yaml',                         # basic settings
                f'config/{args.dataset}.yaml',              # dataset settings (we put our prompt template here)
                f'config/{args.backbone.lower()}.yaml',     # backbone settings (backbone hyperparameters)
                f'config/llmhd.yaml'                        # our model setting
            ]
        )
    else:
        raise ValueError("Backbone model must be ['NGCF', 'LightGCN', 'SGL', 'NCL']")
    return config


def load_dataset(args, config):
    if args.backbone not in ["SGL", "NCL"]:
        dataset = create_dataset(config)
    elif args.backbone in ["SGL", "NCL"]:
        dataset = gnnutils.create_dataset(config)
    else:
        raise ValueError("Backbone model must be ['NGCF', 'LightGCN' 'SGL', 'NCL']")
    return dataset

def load_dataloader(args, config, dataset):
    if args.backbone not in ["SGL", "NCL"]:
        train_data, valid_data, test_data = data_preparation(config, dataset)
    elif args.backbone in ["SGL", "NCL"]:
        train_data, valid_data, test_data = gnnutils.data_preparation(config, dataset)
    else:
        raise ValueError("Backbone model must be ['NGCF', 'LightGCN' 'SGL', 'NCL']")
    return train_data, valid_data, test_data

def load_trainer(args):
    if args.denoise not in ["BOD", "DeCA"]:
        trainer_class = DenoiseTrainer
    elif args.denoise == "BOD":
        trainer_class = BODTrainer
    else:
        raise NotImplementedError(f"The trainer of denoise method '{args.denoise}' is not implemented!")
    return trainer_class