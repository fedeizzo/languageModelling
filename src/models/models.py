import torch
import math
import torch.nn as nn
import pytorch_lightning as pl

from typing import Any, Dict
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils.utils import get_optimizer, get_criterion, flatten_dict


class LSTM(nn.Module):
    def __init__(
        self,
        lstm_size: int,
        embedding_dim: int,
        num_layers: int,
        dropout: float,
        tie_weights: bool,
        vocabulary_lenght: int,
        batch_size: int,
    ):
        super(LSTM, self).__init__()
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_lenght,
            embedding_dim=embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=lstm_size,
            hidden_size=lstm_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(lstm_size, vocabulary_lenght)
        if tie_weights:
            assert embedding_dim == self.lstm_size, "cannot tie, check dims"
            self.embedding.weight = self.fc.weight
        self.init_weights()

    def forward(self, x, x_lens):
        h0, c0 = self.init_hidden(self.batch_size)
        embed = self.embedding(x)
        packed = pack_padded_sequence(
            embed, x_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        output_packed, (h, c) = self.lstm(packed, (h0, c0))
        output_padded, _ = pad_packed_sequence(output_packed, batch_first=True)
        outputs = self.fc(output_padded)
        return outputs, (h, c)

    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1 / math.sqrt(self.lstm_size)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_emb)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(
                self.embedding_dim, self.lstm_size
            ).uniform_(-init_range_other, init_range_other)
            self.lstm.all_weights[i][1] = torch.FloatTensor(
                self.lstm_size, self.lstm_size
            ).uniform_(-init_range_other, init_range_other)

    # init layer uniformely in the range -0.1 and 0.1 as stated in the following paper
    # https://arxiv.org/abs/1708.02182
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.lstm_size)
        cell = torch.zeros(self.num_layers, batch_size, self.lstm_size)
        if torch.cuda.is_available():
            hidden, cell = hidden.to("cuda"), cell.to("cuda")

        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell


class PeenTreeBankLSTM(pl.LightningModule):
    def __init__(self, config: Dict[str, Any], vocabulary_lenght: int):
        super().__init__()
        self.config = config
        self.model = LSTM(
            config["lstm_size"],
            config["embedding_dim"],
            config["num_layers"],
            config["dropout"],
            config["tie_embedding_weights"],
            vocabulary_lenght,
            config["batch_size"],
        )
        self.criterion = self.configure_criterion()
        self.epoch_loss = {"train": 0, "val": 0, "train_count": 0, "val_count": 0}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, x_lens, target = batch
        target = target.reshape(-1)
        outputs, hidden = self.model(x, x_lens)
        outputs = outputs.reshape(outputs.shape[0] * outputs.shape[1], -1)

        loss = self.criterion(outputs, target)
        self.log("loss/train", loss.item(), on_epoch=True, on_step=False)
        self.log(
            "perplexity/train", torch.exp(loss).item(), on_epoch=True, on_step=False
        )
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        x, x_lens, target = batch
        target = target.reshape(-1)
        outputs, hidden = self.model(x, x_lens)
        outputs = outputs.reshape(outputs.shape[0] * outputs.shape[1], -1)

        loss = self.criterion(outputs, target)
        self.log("loss/val", loss.item(), on_epoch=True, on_step=False)
        self.log("perplexity/val", torch.exp(loss).item(), on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        return get_optimizer(self, self.config)

    def configure_criterion(self):
        return get_criterion(self.config)
