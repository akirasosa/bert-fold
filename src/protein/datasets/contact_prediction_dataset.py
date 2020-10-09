from typing import List, Any

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from mylib.torch.data.dataset import PandasDataset
from mylib.torch.functional import calculate_distances
from protein.dto import Indices3D
from protein.dto.batch import ProteinNetBatch
from protein.dto.targets import BertFoldTargets, Targets, PairwiseTargets
from protein.tokenizers import ProtBertTokenizer


# %%
class ProteinNetDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            tokenizer: ProtBertTokenizer = ProtBertTokenizer()
    ):
        self.tokenizer = tokenizer
        self.data = PandasDataset(df)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]

        input_ids = torch.tensor(self.tokenizer.encode(item['primary']))
        attention_mask = torch.ones_like(input_ids)

        R = item['tertiary_ca'].reshape(-1, 3)
        R = torch.from_numpy(R.copy())

        valid_mask = item['valid_mask'].astype(bool)

        seq_mask = valid_mask[:-1] & valid_mask[1:]
        seq_indices = np.nonzero(seq_mask)[0]
        phi = item['phi'][seq_mask]
        psi = item['psi'][seq_mask]

        idx_0, idx_1 = np.triu_indices(len(valid_mask))
        pair_mask = valid_mask[idx_0] & valid_mask[idx_1]
        pair_mask &= np.abs(idx_0 - idx_1) >= 6
        pair_indices = (idx_0[pair_mask], idx_1[pair_mask])

        return input_ids, attention_mask, R, seq_indices, phi, psi, pair_indices, item['id']

    @staticmethod
    def collate(batch: List[Any]) -> ProteinNetBatch:
        input_ids, attention_mask, R, seq_indices, phi, psi, pair_indices, _ = tuple(zip(*batch))

        input_ids = pad_sequence(input_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        # B, N, 3 (xyz)
        R = pad_sequence(R, batch_first=True)

        seq_batch = np.concatenate([np.full(len(x), n) for n, x in enumerate(seq_indices)])
        seq_indices = np.concatenate(seq_indices)
        seq_indices = (
            torch.from_numpy(seq_batch).long(),
            torch.from_numpy(seq_indices).long(),
        )

        phi = torch.from_numpy(np.concatenate(phi)).float()
        psi = torch.from_numpy(np.concatenate(psi)).float()

        pair_batch = np.concatenate([np.full(len(i), n) for n, (i, j) in enumerate(pair_indices)])
        pair_i = np.concatenate([i for i, j in pair_indices])
        pair_j = np.concatenate([j for i, j in pair_indices])
        pair_indices = (
            torch.from_numpy(pair_batch).long(),
            torch.from_numpy(pair_i).long(),
            torch.from_numpy(pair_j).long(),
        )

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'coords': R,
            'phi': phi,
            'psi': psi,
            'seq_indices': seq_indices,
            'pair_indices': pair_indices,
        }


def _prepare_dist_targets(coords: torch.FloatTensor, pair_indices: Indices3D) -> PairwiseTargets:
    R0 = coords[pair_indices[0], pair_indices[1]]
    R1 = coords[pair_indices[0], pair_indices[2]]
    D = calculate_distances(R0, R1)

    return Targets(
        indices=pair_indices,
        values=D,
    )


def prepare_targets(raw_targets: ProteinNetBatch) -> BertFoldTargets:
    dist = _prepare_dist_targets(raw_targets['coords'], raw_targets['pair_indices'])

    return BertFoldTargets(
        dist=dist,
        phi=Targets(
            indices=raw_targets['seq_indices'],
            # TODO
            # indices=(raw_targets['seq_indices'][0], raw_targets['seq_indices'][1] + 1),
            values=raw_targets['phi'],
        ),
        psi=Targets(
            indices=raw_targets['seq_indices'],
            values=raw_targets['psi'],
        ),
    )


# %%
if __name__ == '__main__':
    # %%
    from const import DATA_PROTEIN_NET_DIR
    import matplotlib.pyplot as plt

    # %%
    df = pd.read_parquet(DATA_PROTEIN_NET_DIR / f'casp12/validation.pqt')
    ds = ProteinNetDataset(df)
    loader = DataLoader(
        ds,
        batch_size=6,
        collate_fn=ProteinNetDataset.collate,
        shuffle=True,
    )

    # %%
    batch: ProteinNetBatch = next(iter(loader))

    # %%
    targets = prepare_targets(batch)
    mask = targets.dist.indices[0] == 0
    seq_len = batch['attention_mask'][0].sum() - 2
    idx_i, idx_j = targets.dist.indices[1][mask], targets.dist.indices[2][mask]

    D = np.full((seq_len, seq_len), np.nan)
    D[idx_i, idx_j] = targets.dist.values[mask].numpy()

    plt.matshow(D)
    plt.show()

    # %%
    mask = np.abs(targets.dist.indices[1] - targets.dist.indices[2]) >= 2
    print(mask.sum())

    plt.hist(np.log(targets.dist.values[mask].numpy().reshape(-1)), bins=100)
    plt.show()

