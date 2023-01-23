import logging
import torch
from typing import Dict
from collections import MutableMapping
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def get_optimizer(model: torch.nn.Module, config: Dict):
    logging.info(f"using {config['optimizer']} optimizer.")
    if config["optimizer"] == "Adam":
        adam_config = config["Adam"]
        return torch.optim.Adam(model.parameters(), lr=adam_config["lr"])
    elif config["optimizer"] == "SGD":
        sgd_config = config["SGD"]
        return torch.optim.SGD(
            model.parameters(), lr=sgd_config["lr"], momentum=sgd_config["momentum"]
        )
    elif config["optimizer"] == "ASGD":
        asgd_config = config["ASGD"]
        sgd_config = config["SGD"]
        return torch.optim.ASGD(
            model.parameters(),
            lr=asgd_config["lr"],
            weight_decay=asgd_config["weight_decay"],
            t0=asgd_config["t0"],
            lambd=asgd_config["lambd"],
        ), torch.optim.SGD(
            model.parameters(), lr=sgd_config["lr"], momentum=sgd_config["momentum"]
        )
    else:
        logging.error(f"{config['optimizer']} not implemented.")
        exit(-1)


def get_criterion(config: Dict):
    if config["criterion"] == "CrossEntropy":
        logging.info(f"using {config['criterion']} criterion.")
        cross_entropy_config = config["CrossEntropy"]
        # TODO change 0 to pad value
        return torch.nn.CrossEntropyLoss(ignore_index=0)
    else:
        logging.error(f"{config['criterion']} not implemented.")
        exit(-1)


def get_early_stopper(config: Dict):
    if config.get("early_stopper", False):
        logging.info("using an early stopper.")
        return EarlyStopping(
            monitor="loss/val",
            mode="min",
            patience=10,
            verbose=False,
        )
    else:
        return None


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
