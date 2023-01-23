import torch
import math
import torch.nn as nn
import pytorch_lightning as pl

from typing import Any, Dict, List
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import asgd, optimizer
from models.regularization import EmbeddingDropout, WeightDrop

from utils.utils import get_optimizer, get_criterion, flatten_dict
from torchnlp.nn import LockedDropout


class LSTM(nn.Module):
    def __init__(
        self,
        lstm_size: int,
        embedding_dim: int,
        num_layers: int,
        dropout: float,
        tie_weights: bool,
        vocabulary_length: int,
        batch_size: int,
    ):
        super(LSTM, self).__init__()
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_length,
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
        self.fc = nn.Linear(lstm_size, vocabulary_length)
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


class MogrifierLSTMCell(nn.Module):
    def __init__(self, lstm_size: int, hidden_size: int, mogrify_steps: int):
        super(MogrifierLSTMCell, self).__init__()
        self.mogrify_steps = mogrify_steps
        self.lstm = nn.LSTMCell(lstm_size, hidden_size)
        self.mogrifier_list = nn.ModuleList(
            [nn.Linear(hidden_size, lstm_size)]
        )  # start with q
        for i in range(1, mogrify_steps):
            if i % 2 == 0:
                self.mogrifier_list.extend([nn.Linear(hidden_size, lstm_size)])  # q
            else:
                self.mogrifier_list.extend([nn.Linear(lstm_size, hidden_size)])  # r

    def mogrify(self, x, h):
        for i in range(self.mogrify_steps):
            if (i + 1) % 2 == 0:
                h = (2 * torch.sigmoid(self.mogrifier_list[i](x))) * h
            else:
                x = (2 * torch.sigmoid(self.mogrifier_list[i](h))) * x
        return x, h

    def forward(self, x, states):
        ht, ct = states
        x, ht = self.mogrify(x, ht)
        ht, ct = self.lstm(x, (ht, ct))
        return ht, ct

    def init_weights(self, mat):
        for m in mat.modules():
            if type(m) in [nn.LSTMCell]:
                for name, param in m.named_parameters():
                    # https://discuss.pytorch.org/t/initializing-rnn-gru-and-lstm-correctly/23605/3
                    if "weight_ih" in name:
                        for idx in range(4):
                            mul = param.shape[0] // 4
                            torch.nn.init.xavier_uniform_(
                                param[idx * mul : (idx + 1) * mul]
                            )
                    elif "weight_hh" in name:
                        for idx in range(4):
                            mul = param.shape[0] // 4
                            torch.nn.init.xavier_uniform_(
                                param[idx * mul : (idx + 1) * mul]
                            )
                    elif "bias" in name:
                        param.data.fill_(0)
            elif type(m) in [nn.Linear]:
                if m.bias != None:
                    m.bias.data.fill_(0.1)


class MogrifierLSTM(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_size: int,
        mogrify_steps: int,
        vocabulary_length: int,
        tie_weights: bool,
        dropout: float,
        device: torch.device,
        regularization: List[str],
        embedding_dropout=None,
        input_output_dropout=None,
    ):
        super(MogrifierLSTM, self).__init__()
        pad_value = 0
        self.hidden_size = hidden_size
        self.device = device
        self.regularization = regularization
        if "EmbeddingDropout" in self.regularization:
            self.embedding = EmbeddingDropout(
                vocabulary_length, embedding_dim, pad_value, dropout=embedding_dropout
            )
        else:
            self.embedding = nn.Embedding(vocabulary_length, embedding_dim)
        self.mogrifier_lstm_layer1 = MogrifierLSTMCell(
            embedding_dim, hidden_size, mogrify_steps
        )
        self.mogrifier_lstm_layer2 = MogrifierLSTMCell(
            hidden_size, hidden_size, mogrify_steps
        )

        self.fc = nn.Linear(hidden_size, vocabulary_length)
        if "LockedDropout" in self.regularization:
            self.in_lock_dropout = LockedDropout(p=input_output_dropout)
            self.out_lock_dropout = LockedDropout(p=input_output_dropout)
        self.drop = nn.Dropout(dropout)
        if tie_weights:
            self.fc.weight = self.embedding.weight
        if "WeightsInitialization" in self.regularization:
            self.apply(self.init_weights)

    def init_weights(self, mat):
        for m in mat.modules():
            if type(m) in [nn.LSTMCell]:
                for name, param in m.named_parameters():
                    # https://discuss.pytorch.org/t/initializing-rnn-gru-and-lstm-correctly/23605/3
                    if "weight_ih" in name:
                        for idx in range(4):
                            mul = param.shape[0] // 4
                            torch.nn.init.xavier_uniform_(
                                param[idx * mul : (idx + 1) * mul]
                            )
                    elif "weight_hh" in name:
                        for idx in range(4):
                            mul = param.shape[0] // 4
                            torch.nn.init.xavier_uniform_(
                                param[idx * mul : (idx + 1) * mul]
                            )
                    elif "bias" in name:
                        param.data.fill_(0)
            elif type(m) in [MogrifierLSTMCell]:
                m.apply(m.init_weights)
            elif type(m) in [nn.Embedding, EmbeddingDropout]:
                torch.nn.init.uniform_(m.weight, -0.1, 0.1)
            elif type(m) in [nn.Linear]:
                if m.bias != None:
                    m.bias.data.fill_(0.1)

    def forward(self, seq, max_len=10):
        embed = self.embedding(seq)

        if "LockedDropout" in self.regularization:
            embed = self.in_lock_dropout(embed)

        batch_size = seq.shape[0]
        h1, c1 = [
            torch.zeros(batch_size, self.hidden_size).to(self.device),
            torch.zeros(batch_size, self.hidden_size).to(self.device),
        ]
        h2, c2 = [
            torch.zeros(batch_size, self.hidden_size).to(self.device),
            torch.zeros(batch_size, self.hidden_size).to(self.device),
        ]
        hidden_states = []
        outputs = []
        for step in range(max_len):
            x = self.drop(embed[:, step])
            h1, c1 = self.mogrifier_lstm_layer1(x, (h1, c1))
            h2, c2 = self.mogrifier_lstm_layer2(h1, (h2, c2))
            # if "LockedDropout" in self.regularization:
            #     try:
            #         h2 = self.out_lock_dropout(h2)
            #     except Exception:
            #         import pdb

            #         pdb.set_trace()
            out = self.fc(self.drop(h2))
            hidden_states.append(h2.unsqueeze(1))
            outputs.append(out.unsqueeze(1))

        hidden_states = torch.cat(
            hidden_states, dim=1
        )  # (batch_size, max_len, hidden_size)
        outputs = torch.cat(outputs, dim=1)  # (batch_size, max_len, vocab_size)

        return outputs, hidden_states


class PeenTreeBankLSTM(pl.LightningModule):
    def __init__(
        self,
        config: Dict[str, Any],
        vocabulary_lenght: int,
        mogrifier: bool = False,
        device=None,
    ):
        super().__init__()
        self.config = config
        if not mogrifier:
            self.model = LSTM(
                config["lstm_size"],
                config["embedding_dim"],
                config["num_layers"],
                config["dropout"],
                config["tie_embedding_weights"],
                vocabulary_lenght,
                config["batch_size"],
            )
            self.is_mogrifier = False
        else:
            self.model = MogrifierLSTM(
                config["embedding_dim"],
                config["lstm_size"],
                config["mogrify_steps"],
                vocabulary_lenght,
                config["tie_embedding_weights"],
                config["dropout"],
                device,
                config.get("regularization", []),
                config.get("embedding_dropout", None),
                config.get("input_output_dropout", None),
            )
            self.is_mogrifier = True
        self.criterion = self.configure_criterion()
        self.epoch_loss = {"train": 0, "val": 0, "train_count": 0, "val_count": 0}

    def forward(self, x, x_lens):
        return self.model(x, torch.max(x_lens) if self.is_mogrifier else x_lens)

    def training_step(self, batch, batch_idx):
        x, x_lens, target = batch
        target = target.reshape(-1)
        outputs, hidden = self.model(
            x, torch.max(x_lens) if self.is_mogrifier else x_lens
        )
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
        outputs, hidden = self.model(
            x, torch.max(x_lens) if self.is_mogrifier else x_lens
        )
        outputs = outputs.reshape(outputs.shape[0] * outputs.shape[1], -1)

        loss = self.criterion(outputs, target)
        self.log("loss/val", loss.item(), on_epoch=True, on_step=False)
        self.log("perplexity/val", torch.exp(loss).item(), on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        if self.config["use_lr_scheduler"]:
            optimizer = get_optimizer(self, self.config)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3, verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "loss/val",
                    "interval": "epoch",
                },
            }
        else:
            return get_optimizer(self, self.config)

    def configure_criterion(self):
        return get_criterion(self.config)


class LSTMTruncatedBP(pl.LightningModule):
    def __init__(
        self,
        config: Dict[str, Any],
        vocabulary_length: int,
        bptt_steps: int,
        is_test: bool = False,
    ):
        super().__init__()
        self.config = config
        # self.truncated_bptt_steps = bptt_steps
        self.regularization = config.get("regularization", [])
        self.batch_size = config["batch_size"]
        self.num_layers = config["num_layers"]
        self.lstm_size = config["lstm_size"]
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_length,
            embedding_dim=config["embedding_dim"],
        )
        self.is_test = is_test

        if "EmbeddingDropout" in self.regularization:
            pad_value = 0
            self.embedding = EmbeddingDropout(
                vocabulary_length,
                config["embedding_dim"],
                pad_value,
                dropout=config["embedding_dropout"],
            )
        else:
            self.embedding = nn.Embedding(
                config["vocabulary_length"], config["embedding_dim"]
            )
        self.lstm = nn.LSTM(
            input_size=config["lstm_size"],
            hidden_size=config["lstm_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            batch_first=True,
            bidirectional=False,
        )
        if "WeightDrop" in self.regularization:
            self.old_lstm = self.lstm
            weights = [f"weight_hh_l{i}" for i in range(self.num_layers)]
            self.lstm = WeightDrop(
                self.old_lstm, weights, dropout=config["hidden_dropout"]
            )

        self.fc = nn.Linear(config["lstm_size"], vocabulary_length)
        if config["tie_embedding_weights"]:
            assert (
                config["embedding_dim"] == config["lstm_size"]
            ), "cannot tie, check dims"
            self.embedding.weight = self.fc.weight

        if "LockedDropout" in self.regularization:
            self.in_lock_dropout = LockedDropout(p=config["input_output_dropout"])
            self.out_lock_dropout = LockedDropout(p=config["input_output_dropout"])

        if "WeightsInitialization" in self.regularization:
            self.apply(self.init_weights)
        self.criterion = self.configure_criterion()
        if self.config["optimizer"] == "ASGD":
            self.automatic_optimization = False
            self.use_asgd = False
            self.validation_losses_epoch = []
            self.validation_losses_batch = []

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.lstm_size)
        cell = torch.zeros(self.num_layers, batch_size, self.lstm_size)
        if torch.cuda.is_available() and not self.is_test:
            hidden, cell = hidden.to("cuda"), cell.to("cuda")

        return hidden, cell

    def init_weights(self, mat):
        for m in mat.modules():
            if type(m) in [nn.LSTM]:
                for name, param in m.named_parameters():
                    # https://discuss.pytorch.org/t/initializing-rnn-gru-and-lstm-correctly/23605/3
                    if "weight_ih" in name:
                        for idx in range(4):
                            mul = param.shape[0] // 4
                            torch.nn.init.xavier_uniform_(
                                param[idx * mul : (idx + 1) * mul]
                            )
                    elif "weight_hh" in name:
                        for idx in range(4):
                            mul = param.shape[0] // 4
                            torch.nn.init.xavier_uniform_(
                                param[idx * mul : (idx + 1) * mul]
                            )
                    elif "bias" in name:
                        param.data.fill_(0)
            elif type(m) in [nn.Linear]:
                if m.bias != None:
                    m.bias.data.fill_(0.1)

    def configure_optimizers(self):
        if self.config["optimizer"] == "ASGD":
            asgd, sgd = get_optimizer(self, self.config)
            return asgd, sgd
        elif self.config["use_lr_scheduler"]:
            optimizer = get_optimizer(self, self.config)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3, verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "loss/val",
                    "interval": "epoch",
                },
            }
        else:
            return get_optimizer(self, self.config)

    def configure_criterion(self):
        return get_criterion(self.config)

    def forward(self, x, x_lens):
        # Model forward
        h0, c0 = self.init_hidden(self.batch_size)
        embed = self.embedding(x)

        if "LockedDropout" in self.regularization:
            embed = self.in_lock_dropout(embed)
        packed = pack_padded_sequence(
            embed, x_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        output_packed, (h, c) = self.lstm(packed, (h0, c0))
        output_padded, _ = pad_packed_sequence(output_packed, batch_first=True)
        if "LockedDropout" in self.regularization:
            output_padded = self.out_lock_dropout(output_padded)
        outputs = self.fc(output_padded)

        return outputs

    # def training_step(self, batch, batch_idx, hiddens):
    def training_step(self, batch, batch_idx):
        if self.config["optimizer"] == "ASGD":
            asgd, sgd = self.optimizers()
        # Model preparation
        x, x_lens, target = batch
        target = target.reshape(-1)

        # Model forward
        hiddens = self.init_hidden(self.batch_size)
        embed = self.embedding(x)

        if "LockedDropout" in self.regularization:
            embed = self.in_lock_dropout(embed)
        packed = pack_padded_sequence(
            embed, x_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        output_packed, hiddens = self.lstm(packed, hiddens)
        output_padded, _ = pad_packed_sequence(output_packed, batch_first=True)
        if "LockedDropout" in self.regularization:
            output_padded = self.out_lock_dropout(output_padded)
        outputs = self.fc(output_padded)

        # Output reshape
        outputs = outputs.reshape(outputs.shape[0] * outputs.shape[1], -1)

        loss = self.criterion(outputs, target)
        if self.config["optimizer"] == "ASGD":
            if not self.use_asgd:
                sgd.zero_grad()
                self.manual_backward(loss)
                sgd.step()
            else:
                asgd.zero_grad()
                self.manual_backward(loss)
                asgd.step()
        self.log("loss/train", loss.item(), on_epoch=True, on_step=False)
        self.log(
            "perplexity/train", torch.exp(loss).item(), on_epoch=True, on_step=False
        )
        # return {"loss": loss, "hiddens": hiddens}
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        # Model preparation
        x, x_lens, target = batch
        target = target.reshape(-1)

        # Model forward
        h0, c0 = self.init_hidden(self.batch_size)
        embed = self.embedding(x)

        if "LockedDropout" in self.regularization:
            embed = self.in_lock_dropout(embed)
        packed = pack_padded_sequence(
            embed, x_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        output_packed, (h, c) = self.lstm(packed, (h0, c0))
        output_padded, _ = pad_packed_sequence(output_packed, batch_first=True)
        if "LockedDropout" in self.regularization:
            output_padded = self.out_lock_dropout(output_padded)
        outputs = self.fc(output_padded)

        # Output reshape
        outputs = outputs.reshape(outputs.shape[0] * outputs.shape[1], -1)

        loss = self.criterion(outputs, target)
        if self.config["optimizer"] == "ASGD":
            self.validation_losses_batch.append(loss.item())
        self.log("loss/val", loss.item(), on_epoch=True, on_step=False)
        self.log("perplexity/val", torch.exp(loss).item(), on_epoch=True, on_step=False)
        return loss

    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        if self.config["optimizer"] == "ASGD":
            self.validation_losses_epoch.append(
                sum(self.validation_losses_batch) / len(self.validation_losses_batch)
            )
            self.validation_losses_epoch = self.validation_losses_epoch[
                -(self.config["ASGD"]["patience"] + 1) :
            ]
            self.validation_losses_batch = []
            if len(self.validation_losses_epoch) > self.config["ASGD"]["patience"]:
                use_asgd_last = self.use_asgd
                sentinel = self.validation_losses_epoch[-1]
                for i in self.validation_losses_epoch[:-1]:
                    self.use_asgd = (i < sentinel) or self.use_asgd
                if use_asgd_last == False and self.use_asgd:
                    self.use_asgd = True
                    print("Switching to ASGD.")
