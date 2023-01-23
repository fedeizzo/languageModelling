import pdb
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import json
import yaml
import argparse
import logging
import torch
import torch.nn.functional as F
import pandas as pd

from pytorch_lightning import Trainer
from data_processing.dataset import generate_dataset_precomputed, PeenTreeBankDataset
from models.lstm import PeenTreeBankLSTM, LSTMTruncatedBP
from models.gru import PeenTreeBankGRU
from typing import Dict
from utils.utils import get_early_stopper
from tqdm import tqdm
from copy import deepcopy


def get_model(config: Dict, vocabulary_len: int) -> pl.LightningModule:
    if config["model"] == "LSTM" and config.get("use_bptt", False):
        model = LSTMTruncatedBP(config["LSTM"], vocabulary_len, config["bptt_steps"])
    elif config["model"] == "LSTM":
        model = PeenTreeBankLSTM(config["LSTM"], vocabulary_len)
    elif config["model"] == "MogrifyLSTM":
        model = PeenTreeBankLSTM(
            config["MogrifyLSTM"],
            vocabulary_len,
            mogrifier=True,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )
    elif config["model"] == "GRU":
        model = PeenTreeBankGRU(config["gru"], vocabulary_len)
    else:
        logging.error(f"{config['model']} model not implemented")
        exit(1)
    return model


def get_model_test(
    checkpoint_path: str, config: Dict, vocabulary_len: int
) -> pl.LightningModule:
    if config["model"] == "LSTM" and config.get("use_bptt", False):
        model = LSTMTruncatedBP.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=config["LSTM"],
            vocabulary_length=vocabulary_len,
            bptt_steps=config["bptt_steps"],
            is_test=True,
        )
    elif config["model"] == "LSTM":
        model = PeenTreeBankLSTM.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=config["LSTM"],
            vocabulary_lenght=vocabulary_len,
        )
    elif config["model"] == "MogrifyLSTM":
        model = PeenTreeBankLSTM.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=config["MogrifyLSTM"],
            vocabulary_lenght=vocabulary_len,
            mogrifier=True,
            device="cpu",
        )
    elif config["model"] == "GRU":
        model = PeenTreeBankGRU.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=config["gru"],
            vocabulary_lenght=vocabulary_len,
        )
    else:
        logging.error(f"{config['model']} model not implemented")
        exit(1)
    return model


def test_model(model, test_loader):
    model.eval()
    rows = []
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for batch in test_loader:
                x, x_lens, y = batch
                y = y.reshape(-1)
                outputs = model(
                    x,
                    x_lens,
                )

                outputs = outputs.reshape(outputs.shape[0] * outputs.shape[1], -1)
                ce_loss = F.cross_entropy(outputs, y, ignore_index=0)
                # deep copy solves a problem that may happen in some cases with the dataloader
                rows.append(
                    [
                        deepcopy(x).numpy(),
                        deepcopy(x_lens).numpy(),
                        deepcopy(outputs).numpy(),
                        deepcopy(y).numpy(),
                        deepcopy(ce_loss).numpy(),
                    ]
                )
                pbar.update(1)
    with open("./test_embedding_maps.json", "w") as outfile:
        json.dump(test_loader.dataset.embedding, outfile)
    df = pd.DataFrame(rows, columns=["x", "x_lens", "outputs", "y", "ce_loss"])
    df.to_pickle("./test_results.pkl")


def main(config):
    if config["task"] == "generate_precomputed":
        logging.info("starting generate precomputed task.")
        generate_dataset_precomputed(
            config["train_filename"],
            config["validation_filename"],
            config["test_filename"],
            config["precomputed_filename"],
        )
    elif config["task"] == "train":
        torch.manual_seed(config["seed"])
        logging.info("starting train task.")
        train_loader, val_loader, test_loader = PeenTreeBankDataset.get_loaders(
            config["precomputed_filename"], config[config["model"]]["batch_size"]
        )
        model = get_model(config, len(train_loader.dataset.embedding) + 1)
        logging.info(f"{config['model']} model chosen for the train task.")
        tbl = TensorBoardLogger(save_dir="lightning_logs")
        tbl.log_hyperparams(config)
        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else None,
            devices=[0] if torch.cuda.is_available() else None,
            max_epochs=config[config["model"]]["epochs"],
            logger=tbl,
            callbacks=[get_early_stopper(config)],
            gradient_clip_val=config["gradient_clipping_val"]
            if config[config["model"]]["optimizer"] != "ASGD"
            else None,
        )
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        model.is_test = True
        model.batch_size = 1
        del train_loader, val_loader
        _, _, test_loader = PeenTreeBankDataset.get_loaders(
            config["precomputed_filename"], 1
        )
        model = model.cpu()
        test_model(model, test_loader)
    elif config["task"] == "test":
        torch.manual_seed(config["seed"])
        logging.info("starting train task.")
        train_loader, _, test_loader = PeenTreeBankDataset.get_loaders(
            config["precomputed_filename"],
            config[config["model"]]["batch_size"],
            is_test=True,
        )
        model = get_model_test(
            config["model_checkpoint"], config, len(train_loader.dataset.embedding) + 1
        )
        test_model(model, test_loader)
    else:
        logging.error(f"{config['task']} task not implemented.")
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Configuration file for the execution.",
    )
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.Loader)
    logging.basicConfig(
        level=config.get("logging_level", logging.INFO),
        format="%(asctime)s :: %(levelname)-8s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main(config)
