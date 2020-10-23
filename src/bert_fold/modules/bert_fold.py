from typing import Optional

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

from bert_fold.dto.outputs import BertFoldOutput
from bert_fold.dto.targets import BertFoldTargets
from bert_fold.metrics import MAEForSeq
from bert_fold.modules.pairwise_distance_decoder import PairwiseDistanceDecoder
from bert_fold.dto.batch import ProteinNetBatch


def init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, nn.LayerNorm):
        module.weight.data.fill_(1.0)
        module.bias.data.zero_()


# noinspection PyAbstractClass
class BertFold(nn.Module):
    def __init__(
            self,
            pretrained: bool = True,
            gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if pretrained:
            self.bert = BertModel.from_pretrained(
                'Rostlab/prot_bert_bfd',
                gradient_checkpointing=gradient_checkpointing,
            )
        else:
            conf = BertConfig.from_pretrained('Rostlab/prot_bert_bfd')
            self.bert = BertModel(conf)

        # noinspection PyUnresolvedReferences
        dim = self.bert.config.hidden_size

        self.evo_linear = nn.Linear(21, dim)

        self.decoder_dist = PairwiseDistanceDecoder(dim)
        # self.decoder_phi = ElementwiseAngleDecoder(dim, 2)
        # self.decoder_psi = ElementwiseAngleDecoder(dim, 2)

        self.evo_linear.apply(init_weights)
        self.decoder_dist.apply(init_weights)
        # self.decoder_phi.apply(init_weights)
        # self.decoder_psi.apply(init_weights)

    def forward(
            self,
            inputs: ProteinNetBatch,
            targets: Optional[BertFoldTargets] = None,
    ) -> BertFoldOutput:
        x_emb = self.bert.embeddings(inputs['input_ids'])
        x_evo = self.evo_linear(inputs['evo'].type_as(x_emb))
        x = x_emb + x_evo
        extended_attention_mask = self.bert.get_extended_attention_mask(
            inputs['attention_mask'],
            inputs['input_ids'].shape,
            inputs['input_ids'].device,
        )
        x = self.bert.encoder.forward(x, attention_mask=extended_attention_mask)[0]

        # x = self.bert.forward(
        #     inputs['input_ids'],
        #     attention_mask=inputs['attention_mask'],
        # )[0]
        # x = torch.cat((
        #     x,
        #     inputs['evo'].type_as(x),
        # ), dim=-1)

        targets_dist = None if targets is None else targets.dist
        # targets_phi = None if targets is None else targets.phi
        # targets_psi = None if targets is None else targets.psi

        outs = [
            self.decoder_dist.forward(x, targets_dist),
            # self.decoder_phi.forward(x, targets_phi),
            # self.decoder_psi.forward(x, targets_psi),
        ]

        y_hat = tuple(x.y_hat for x in outs)

        if targets is None:
            return BertFoldOutput(
                y_hat=y_hat,
            )

        loss = torch.stack([x.loss for x in outs]).sum()

        # Collect metrics
        with torch.no_grad():
            # Long range MAE metrics
            mae_l8_fn = MAEForSeq(contact_thre=8.)
            results = mae_l8_fn(
                inputs=y_hat[0][targets.dist.indices],
                targets=targets.dist.values,
                indices=targets.dist.indices,
            )
            if len(results) > 0:
                mae_l_8 = (results.mean().detach().item(), len(results))
            else:
                mae_l_8 = (0, 0)

            # Top L/5 precision metrics
            # top_l5_precision_fn = TopLNPrecision(n=5, contact_thre=8.)
            # results = top_l5_precision_fn(
            #     inputs=out_dist.y_hat[targets.dist.indices],
            #     targets=targets.dist.values,
            #     indices=targets.dist.indices,
            #     seq_lens=attention_mask.sum(-1) - 2,
            # )
            # if len(results) > 0:
            #     top_l5_precision = (results.mean().detach().item(), len(results))
            # else:
            #     top_l5_precision = (0, 0)

        return BertFoldOutput(
            y_hat=y_hat,
            loss=loss,
            loss_dist=outs[0].loss_and_cnt,
            # loss_phi=outs[1].loss_and_cnt,
            # loss_psi=outs[2].loss_and_cnt,
            mae_l_8=mae_l_8,
        )


# %%
if __name__ == '__main__':
    # %%
    from torch.utils.data import DataLoader
    from bert_fold.dataset import ProteinNetDataset, prepare_targets
    from bert_fold.dto.batch import ProteinNetBatch
    from const import DATA_PROTEIN_NET_DIR
    import pandas as pd

    # %%
    loader = DataLoader(
        ProteinNetDataset(
            pd.read_parquet(DATA_PROTEIN_NET_DIR / f'casp12/validation.pqt')
        ),
        batch_size=2,
        collate_fn=ProteinNetDataset.collate,
        shuffle=False
    )
    # %%
    model = BertFold(pretrained=False)

    # %%
    batch: ProteinNetBatch = next(iter(loader))
    targets = prepare_targets(batch)
    out = model.forward(batch, targets=targets)
    pass

    # %%
    for k, v in model.named_parameters():
        print(k)
