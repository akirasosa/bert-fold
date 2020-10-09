from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from protein.dto import Indices2D, Indices3D

T = TypeVar('T')


@dataclass
class Targets(Generic[T]):
    indices: T
    values: torch.Tensor


ElementwiseTargets = Targets[Indices2D]
PairwiseTargets = Targets[Indices3D]


@dataclass
class BertFoldTargets:
    dist: PairwiseTargets
    phi: ElementwiseTargets
    psi: ElementwiseTargets
