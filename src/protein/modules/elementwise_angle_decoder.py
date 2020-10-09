from typing import Optional

import torch
import torch.nn as nn

from protein.dto.outputs import DecoderOutput
from protein.dto.targets import ElementwiseTargets


# noinspection PyAbstractClass
class ElementwiseAngleDecoder(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)

    def forward(self, x, targets: Optional[ElementwiseTargets] = None) -> DecoderOutput:
        x = self.linear(x)
        # float is needed in case of apex O2.
        y_hat = x[:, 1:-1].float()
        loss = None

        if targets is not None:
            y = targets.values
            y = torch.stack((torch.cos(y), torch.sin(y)), dim=-1).type_as(y_hat)

            cos = nn.CosineSimilarity(dim=1)
            loss = (1. - cos(y_hat[targets.indices], y)).mean()

        return DecoderOutput(
            loss=loss,
            y_hat=y_hat,
        )
