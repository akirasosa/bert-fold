from collections import OrderedDict
from functools import cached_property, partial
from logging import getLogger, FileHandler
from multiprocessing import cpu_count
from pathlib import Path
from time import time
from typing import Optional, Union, Mapping, Sequence, TypedDict, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch_optimizer import RAdam

from bert_fold.dataset import ProteinNetDataset, prepare_targets
from bert_fold.dto.batch import ProteinNetBatch
from bert_fold.modules.bert_fold import BertFold
from bert_fold.params import ModuleParams, DataParams, Params
from bert_fold.tokenizers import ProtBertTokenizer
from const import DATA_PROTEIN_NET_DIR
from mylib.pytorch_lightning.base_module import PLBaseModule
from mylib.pytorch_lightning.logging import configure_logging
from mylib.torch.ensemble.ema import create_ema
from mylib.torch.optim.AdaBelief import AdaBelief
from mylib.torch.optim.sched import flat_cos


def load_bert_fold(params: ModuleParams) -> BertFold:
    if params.pretrained_ckpt_path is None:
        return BertFold(
            pretrained=True,
            gradient_checkpointing=params.gradient_checkpointing,
        )

    model = BertFold(pretrained=False)
    ckpt = torch.load(params.pretrained_ckpt_path)

    if any(k.startswith('ema_model') for k in ckpt['state_dict'].keys()):
        prefix = 'ema_model'
    else:
        prefix = 'model'

    new_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        if not k.startswith(f'{prefix}.'):
            continue
        new_dict[k[len(f'{prefix}.'):]] = v

    info = model.load_state_dict(new_dict, strict=False)

    logger = getLogger('lightning')
    logger.info(info)

    return model


# noinspection PyAbstractClass
class DataModule(pl.LightningDataModule):
    def __init__(self, params: DataParams):
        super().__init__()
        self.params = params
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.tokenizer = ProtBertTokenizer()

    def setup(self, stage: Optional[str] = None):
        def filter_df(df: pd.DataFrame):
            mask = df['primary'].apply(len) <= self.params.seq_len
            return df[mask]

        df_train = pd.read_parquet(DATA_PROTEIN_NET_DIR / 'casp12/train_all.pqt')
        df_val = pd.read_parquet(DATA_PROTEIN_NET_DIR / 'casp12/validation.pqt')
        df_test = pd.read_parquet(DATA_PROTEIN_NET_DIR / 'casp12/testing.pqt')

        self.train_dataset = ProteinNetDataset(
            filter_df(df_train),
            tokenizer=self.tokenizer,
        )
        self.val_dataset = ProteinNetDataset(
            filter_df(df_val),
            tokenizer=self.tokenizer,
        )
        self.test_dataset = ProteinNetDataset(
            filter_df(df_test),
            tokenizer=self.tokenizer,
        )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            collate_fn=ProteinNetDataset.collate,
            num_workers=cpu_count(),
            pin_memory=True,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, Sequence[DataLoader]]:
        return [
            DataLoader(
                ds,
                batch_size=self.params.batch_size,
                collate_fn=ProteinNetDataset.collate,
                shuffle=False,
                num_workers=cpu_count(),
                pin_memory=True,
            )
            for ds in [self.val_dataset, self.test_dataset]
        ]


ValueAndWeight = Tuple[float, float]


class StepResult(TypedDict):
    loss: torch.Tensor
    n_processed: int
    loss_dist: ValueAndWeight
    # loss_phi: ValueAndWeight
    # loss_psi: ValueAndWeight
    mae_l_8: ValueAndWeight
    # top_l5_precision: ValueAndWeight


def calc_weighted_mean(items: Sequence[Tuple[float, float]]) -> float:
    items = np.array(items)
    values, weights = items[:, 0], items[:, 1]
    result = (values * weights).sum() / weights.sum()
    return result


# noinspection PyAbstractClass
class PLModule(PLBaseModule[BertFold]):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.hparams = hparams
        self.model = load_bert_fold(self.hp)

        if self.hp.use_ema:
            self.ema_model = create_ema(self.model)

    def configure_optimizers(self):
        opt = AdaBelief(
            self.model.parameters(),
            lr=self.hp.lr,
            weight_decay=self.hp.weight_decay,
        )

        return [opt]

    def step(self, model: BertFold, batch: ProteinNetBatch) -> StepResult:
        targets = prepare_targets(batch)
        out = model.forward(batch['input_ids'], batch['attention_mask'], targets=targets)
        result = StepResult(n_processed=len(batch['input_ids']), **{
            k: v for k, v in out.items()
            if not k == 'y_hat'
        })

        return result

    def collect_metrics(self, outputs: Sequence[StepResult]) -> Mapping:
        base_metric = super(PLModule, self).collect_metrics(outputs)

        keys = StepResult.__annotations__.keys()
        keys = filter(lambda x: x not in ['loss', 'n_processed'], keys)
        # noinspection PyTypedDict
        other_metrics = {
            k: calc_weighted_mean([x[k] for x in outputs])
            for k in keys
        }

        return {
            **base_metric,
            **other_metrics,
        }

    @cached_property
    def hp(self) -> ModuleParams:
        return ModuleParams(**self.hparams)


def train(params: Params):
    seed_everything(params.seed)

    experiment_id = str(int(time()))

    tb_logger = TensorBoardLogger(
        params.t.save_dir,
        name='bert_fold',
        version=experiment_id,
    )

    log_dir = Path(tb_logger.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = getLogger('lightning')
    logger.addHandler(FileHandler(log_dir / 'train.log'))
    logger.info(params.pretty())

    trainer = pl.Trainer(
        max_epochs=params.t.epochs,
        gpus=params.t.gpus,
        tpu_cores=params.t.num_tpu_cores,
        logger=tb_logger,
        precision=16 if params.t.use_16bit else 32,
        amp_level=params.t.amp_level,
        amp_backend='apex',
        resume_from_checkpoint=params.t.resume_from_checkpoint,
        weights_save_path=params.t.weights_save_path,
        # early_stop_callback=EarlyStopping(
        #     monitor='ema_0_loss' if params.m.use_ema else 'val_0_loss',
        #     patience=30,
        #     mode='min'
        # ),
        checkpoint_callback=ModelCheckpoint(
            monitor='ema_0_loss' if params.m.use_ema else 'val_0_loss',
            save_last=True,
            verbose=True,
        ),
        # distributed_backend=params.t.distributed_backend,
        # num_nodes=params.t.num_nodes,
        accumulate_grad_batches=params.t.accumulate_grad_batches,
        terminate_on_nan=True,
        deterministic=True,
        benchmark=True,
        val_check_interval=1 / 3,
    )
    dm = DataModule(params.d)
    net = PLModule(params.m.dict_config())

    logger.info(net.model)

    trainer.fit(net, datamodule=dm)


if __name__ == '__main__':
    configure_logging()
    params = Params.load()
    train(params)
