import dataclasses
from typing import Optional

from const import EXP_DIR
from mylib.params import ParamsMixIn


@dataclasses.dataclass(frozen=True)
class DataParams(ParamsMixIn):
    batch_size: int = 256
    seq_len: int = 512


@dataclasses.dataclass(frozen=True)
class ModuleParams(ParamsMixIn):
    optim: str = 'radam'
    lr: float = 1e-5
    lr_bert: float = 3e-6
    weight_decay: float = 0.

    ema_decay: Optional[float] = None
    ema_eval_freq: int = 1

    pretrained_ckpt_path: Optional[str] = None

    @property
    def use_ema(self) -> bool:
        return self.ema_decay is not None


@dataclasses.dataclass(frozen=True)
class TrainerParams(ParamsMixIn):
    num_tpu_cores: Optional[int] = None
    gpus: Optional[int] = None
    epochs: int = 100
    amp_level: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    save_dir: str = str(EXP_DIR)
    distributed_backend: Optional[str] = None
    num_nodes: int = 1
    accumulate_grad_batches: int = 1
    weights_save_path: Optional[str] = None

    @property
    def use_16bit(self) -> bool:
        return self.amp_level is not None


@dataclasses.dataclass(frozen=True)
class Params(ParamsMixIn):
    module_params: ModuleParams
    trainer_params: TrainerParams
    data_params: DataParams
    note: str = ''
    seed: int = 0

    @property
    def m(self) -> ModuleParams:
        return self.module_params

    @property
    def t(self) -> TrainerParams:
        return self.trainer_params

    @property
    def d(self) -> DataParams:
        return self.data_params


# %%
if __name__ == '__main__':
    # %%
    p = Params.load('params/chem-mlm/001.yaml')
    print(p.pretty())
