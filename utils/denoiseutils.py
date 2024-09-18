# -*- coding: utf-8 -*-
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
import recbole_gnn.recbole_gnn.utils as gnnutils
from recbole_gnn.recbole_gnn.config import Config as GNNConfig
from utils.trainer import *
from model.base import *
from model.PairDenoise import *


def load_config(args):
    if args.denoise.lower() not in ["bpr", "bce"]:
        config_file_list = [
            'config/base.yaml',                        # basic settings
            f'config/{args.dataset}.yaml',             # dataset settings (we put our prompt template here)
            f'config/{args.backbone.lower()}.yaml',    # backbone settings (backbone hyperparameters)
            f'config/{args.denoise.lower()}.yaml'      # our model setting
        ]
    else:
        config_file_list = [
            'config/base.yaml',  # basic settings
            f'config/{args.dataset}.yaml',  # dataset settings (we put our prompt template here)
            f'config/{args.backbone.lower()}.yaml',  # backbone settings (backbone hyperparameters)
        ]
    if args.backbone not in ["SGL", "NCL"]:
        config = Config(
            model=args.backbone,
            dataset=args.dataset,
            config_file_list=config_file_list
        )
    elif args.backbone in ["SGL", "NCL"]:
        config = GNNConfig(
            model=args.backbone,
            dataset=args.dataset,
            config_file_list=config_file_list
        )
    else:
        raise ValueError("Backbone model must be one of ['NGCF', 'LightGCN', 'SGL', 'NCL']")
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


def load_model(args, config, dataset, backbone):
    if args.denoise == "BPR":
        model = BasePairDenoiseCF(config, dataset, backbone)
    elif args.denoise == "BCE":
        model = BasePointDenoiseCF(config, dataset, backbone)
    elif args.denoise == "TCE":
        model = TCEPairDenoise(config, dataset, backbone)
    elif args.denoise == "RCE":
        model = RCEPairDenoise(config, dataset, backbone)
    else:
        raise ValueError("Denoise Training Approach Not Implemented!")
    return model


def load_trainer(args):
    if args.denoise not in ["BOD", "DeCA"]:
        trainer_class = DenoiseTrainer
    elif args.denoise == "BOD":
        trainer_class = BODTrainer
    else:
        raise NotImplementedError(f"The trainer of denoise method '{args.denoise}' is not implemented!")
    return trainer_class