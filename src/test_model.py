import torch
import torch.nn as nn
import numpy as np
import logging

from torch.utils.data import DataLoader
from typing import Union, Any

from models.models import LSTM


def test_model(
    model: Union[LSTM, Any],
    device: torch.device,
    test_loader: DataLoader,
    criterion,
):
    model.eval()
    hidden = model.init_hidden(batch_size, device)

    total_loss = 0
    for x, x_lens, target in test_loader:
        x = x.to(device)
        x_lens = x_lens.to(device)
        target = target.reshape(-1).to(device)

        hidden = model.detach_hidden(hidden)

        prediction, hidden = model(x, x_lens, hidden)
        prediction = prediction.reshape(prediction.shape[0] * prediction.shape[1], -1)
        total_loss += criterion(prediction, target)

    total_loss /= len(test_loader)
    pp = torch.exp(torch.tensor(total_loss))
    logging.info(f"testing with loss {total_loss} and pp {pp}")
