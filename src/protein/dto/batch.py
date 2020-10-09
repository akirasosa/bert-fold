from typing import TypedDict

import torch

from protein.dto import Indices2D, Indices3D


class ProteinNetBatch(TypedDict):
    input_ids: torch.LongTensor
    attention_mask: torch.Tensor
    coords: torch.FloatTensor
    phi: torch.FloatTensor
    psi: torch.FloatTensor
    seq_indices: Indices2D
    pair_indices: Indices3D
