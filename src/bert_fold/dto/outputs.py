from dataclasses import dataclass
from typing import Optional
from typing import Tuple, Sequence

import torch
from transformers.file_utils import ModelOutput

ValueAndWeight = Tuple[float, float]


@dataclass(frozen=True)
class DecoderOutput(ModelOutput):
    y_hat: torch.Tensor
    loss: Optional[torch.FloatTensor] = None

    @property
    def loss_and_cnt(self) -> ValueAndWeight:
        return self.loss.detach().item(), len(self.y_hat)


@dataclass(frozen=True)
class BertFoldOutput(ModelOutput):
    y_hat: Sequence[torch.Tensor]
    loss: Optional[torch.Tensor] = None
    loss_dist: Optional[ValueAndWeight] = None
    loss_phi: Optional[ValueAndWeight] = None
    loss_psi: Optional[ValueAndWeight] = None
    mae_l_8: Optional[ValueAndWeight] = None
    # top_l5_precision: Optional[ValueAndWeight] = None
