from typing import Optional

import torch
import torch.nn as nn

from bert_fold.dto.outputs import DecoderOutput
from bert_fold.dto.targets import PairwiseTargets


# noinspection PyAbstractClass
class PairwiseDistanceDecoder(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.linear = nn.Linear(n_in, n_in)

    def forward(self, x, targets: Optional[PairwiseTargets] = None) -> DecoderOutput:
        x = x.matmul(self.linear(x).transpose(-2, -1))

        y_hat = x[:, 1:-1, 1:-1]
        loss = None

        if targets is not None:
            idx_0, idx_1, idx_2 = targets.indices
            y_hat_ij = y_hat[idx_0, idx_1, idx_2]
            y_hat_ji = y_hat[idx_0, idx_2, idx_1]
            y_hat_all = torch.cat((y_hat_ij, y_hat_ji))
            # float is needed in case of apex O2.
            y_hat_all = y_hat_all.float()

            y = torch.log(targets.values)
            y = torch.cat((y, y))

            criterion = nn.MSELoss()
            # criterion = nn.L1Loss()
            # criterion = LogCoshLoss()
            loss = criterion(y_hat_all, y.type_as(y_hat_all))

        return DecoderOutput(
            loss=loss,
            y_hat=torch.exp((y_hat + y_hat.transpose(2, 1)) / 2.),
        )
