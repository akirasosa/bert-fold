from dataclasses import dataclass

import torch
from torch_scatter import scatter_mean

from mylib.torch.nn.functional import scatter_sort
from protein.dto import Indices3D


@dataclass
class MAEForSeq:
    range_min: int = 24
    range_max: int = 999999999
    contact_thre: float = 8.

    def __call__(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            indices: Indices3D,
    ) -> torch.Tensor:
        # batch, i, j
        idx_0, idx_1, idx_2 = indices

        gap = torch.abs(idx_1 - idx_2)
        mask_seq_range = (self.range_min <= gap) & (gap <= self.range_max)
        mask_y_true = (targets <= self.contact_thre)
        mask = mask_seq_range & mask_y_true

        err = torch.abs(targets[mask].type_as(inputs) - inputs[mask])
        maes = scatter_mean(err, idx_0[mask])

        return maes


@dataclass
class TopLNPrecision:
    def __init__(self, n: int = 5, contact_thre: float = 8.):
        self.n = n
        self.contact_thre = contact_thre

    def __call__(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            indices: Indices3D,
            seq_lens: torch.LongTensor,
    ) -> torch.Tensor:
        # batch, i, j
        idx_0, idx_1, idx_2 = indices

        _, y_hat_sort_index = scatter_sort(inputs, idx_0)
        _, n_pairs = torch.unique(idx_0, return_counts=True)

        ln_mask = torch.cat([
            torch.cat((
                torch.ones(ln, device=n.device),
                # FIXME
                # L/5 can be greater than n_pairs, when there are many disordered residues.
                # Then, how should I take care for it?
                torch.zeros(n - ln, device=n.device),
            ))
            for ln, n in zip(seq_lens // self.n, n_pairs)
        ]).bool()

        ln_indices = y_hat_sort_index[ln_mask]
        results = scatter_mean(
            (targets[ln_indices] < self.contact_thre).float(),
            idx_0[ln_mask],
        )

        return results
