# -*- coding: utf-8 -*-
import os
import sys
import time
import argparse
from logging import getLogger
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    calculate_valid_score,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_model
)
from utils.denoiseutils import *

def run_denoise_cf(args):
    if args.denoise not in ["BPR", "BCE", "TCE", "RCE", "BOD", "DCF", "DDRM"]:
        raise ValueError("'denoise' must be one of [BPR, BCE, T-CE, R-CE, BOD, DCF].")

    config = load_config(args)              # load config

    # fix all seed
    config['device'] = torch.device(f"cuda:{config['gpu_id'][-1]}")
    init_seed(config["seed"], config["reproducibility"])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    # log info
    logger.info(sys.argv)
    logger.info(config)

    # load dataset
    dataset = load_dataset(args, config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = load_dataloader(args, config, dataset)     # test_data.uid2history_item -- used

    # Load Model
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    if args.backbone not in ["SGL", "NCL"]:
        backbone = get_model(args.backbone)(config, train_data._dataset).to(config["device"])
    elif args.backbone in ["SGL", "NCL"]:
        backbone = gnnutils.get_model(args.backbone)(config, train_data.dataset).to(config["device"])
    else:
        raise ValueError("Backbone model must be ['NGCF', 'LightGCN' 'SGL', 'NCL']")
    model = load_model(args, config, train_data.dataset, backbone)
    logger.info(model)
    # Log FLOPs
    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = load_trainer(args)(config, model)

    # model training
z    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=True, show_progress=config["show_progress"]
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='LightGCN', help='loading backbone models [NGCF / LightGCN / SGL / NCL]')
    parser.add_argument('--dataset', type=str, default='yelp', help='dataset to be used [yelp]')
    parser.add_argument('--denoise', type=str, default='RCE', help='the denoise training mode of backbone [BPR, TCE, RCE]')
    args, _ = parser.parse_known_args()

    run_denoise_cf(args)


    # MODEL | LightGCN  | NGCF  | SGL   | NCL
    # BPR   |    OK     |       |       |
    # TCE-P |    OK     |       |       |
    # RCE-P |           |       |       |
    # BOD   |           |       |       |
    # DDRM  |           |       |       |
