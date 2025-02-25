"""itwinai integration of MLPF"""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)
from pathlib import Path

import ray.train
import ray.train.torch
import ray.tune
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import yaml
from ray.train import Checkpoint, DataConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchConfig
from ray.train.torch import TorchTrainer as RayTorchTrainer
from ray.tune import TuneConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from itwinai.torch.config import TrainingConfiguration
from itwinai.torch.type import Batch, Metric
from itwinai.loggers import Logger
from itwinai.torch.distributed import RayTorchDistributedStrategy

import torch

from mlpf.model.mlpf import MLPF
from mlpf.model.PFDataset import get_interleaved_dataloaders  # , Collater, PFDataset
from mlpf.model.utils import (
    # unpack_predictions,
    # unpack_target,
    # get_model_state_dict,
    # load_checkpoint,
    # save_checkpoint,
    CLASS_LABELS,
    X_FEATURES,
    ELEM_TYPES_NONZERO,
    # save_HPs,
    get_lr_schedule,
    count_parameters,
)

from .training import configure_model_trainable, train_all_epochs

from itwinai.torch.trainer import TorchTrainer as ItwinaiTorchTrainer
# from itwinai.components import DataGetter as ItwinaiDataGetter


if TYPE_CHECKING:
    from ray.train.horovod import HorovodConfig


# class PFDatasetGetter(ItwinaiDataGetter):
#     """Instantiate and return a dataset."""


class MLPFTrainer(ItwinaiTorchTrainer):
    """itwinai trainer class for MLPF

    Args:
        config (Union[Dict, TrainingConfiguration]): training configuration
            containing hyperparameters.
        epochs (int): number of training epochs.
        model (Optional[Union[nn.Module, str]], optional): pytorch model to
            train or a string identifier. Defaults to None.
        strategy (Literal['ddp', 'deepspeed', 'horovod'], optional):
            distributed strategy. Defaults to 'ddp'.
        test_every (Optional[int], optional): run a test epoch
            every ``test_every`` epochs. Disabled if None. Defaults to None.
        random_seed (Optional[int], optional): set random seed for
            reproducibility. If None, the seed is not set. Defaults to None.
        logger (Optional[Logger], optional): logger for ML tracking.
            Defaults to None.
        metrics (Optional[Dict[str, Metric]], optional): map of torchmetrics
            metrics. Defaults to None.
        checkpoints_location (str): path to checkpoints directory.
            Defaults to "checkpoints".
        checkpoint_every (Optional[int]): save a checkpoint every
            ``checkpoint_every`` epochs. Disabled if None. Defaults to None.
        disable_tqdm (bool): whether to disable tqdm progress bar(s).
        name (Optional[str], optional): trainer custom name. Defaults to None.
        profiling_wait_epochs (int): how many epochs to wait before starting
            the profiler.
        profiling_warmup_epochs (int): length of the profiler warmup phase in terms of
            number of epochs.
        ray_scaling_config (ScalingConfig, optional): scaling config for Ray Trainer.
            Defaults to None,
        ray_tune_config (TuneConfig, optional): tune config for Ray Tuner.
            Defaults to None.
        ray_run_config (RunConfig, optional): run config for Ray Trainer.
            Defaults to None.
        ray_search_space (Dict[str, Any], optional): search space for Ray Tuner.
            Defaults to None.
        ray_torch_config (TorchConfig, optional): torch configuration for Ray's TorchTrainer.
            Defaults to None.
        ray_data_config (DataConfig, optional): dataset configuration for Ray.
            Defaults to None.
        ray_horovod_config (HorovodConfig, optional): horovod configuration for Ray's
            HorovodTrainer. Defaults to None.
        from_checkpoint (str | Path, optional): path to checkpoint directory. Defaults to None.
    """

    def __init__(
        self,
        config: Union[Dict, TrainingConfiguration],
        epochs: int,
        model: Optional[Union[nn.Module, str]] = None,
        strategy: Optional[Literal["ddp", "deepspeed", "horovod"]] = "ddp",
        test_every: Optional[int] = None,
        random_seed: Optional[int] = None,
        logger: Optional[Logger] = None,
        metrics: Optional[Dict[str, Metric]] = None,
        checkpoints_location: str | Path = "checkpoints",
        checkpoint_every: Optional[int] = None,
        disable_tqdm: bool = False,
        name: Optional[str] = None,
        profiling_wait_epochs: int = 1,
        profiling_warmup_epochs: int = 2,
        ray_scaling_config: ScalingConfig | None = None,
        ray_tune_config: TuneConfig | None = None,
        ray_run_config: RunConfig | None = None,
        ray_search_space: Dict[str, Any] | None = None,
        ray_torch_config: TorchConfig | None = None,
        ray_data_config: DataConfig | None = None,
        ray_horovod_config: Optional["HorovodConfig"] = None,
        from_checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__(
            config=config,
            epochs=epochs,
            model=model,
            strategy=strategy,
            test_every=test_every,
            random_seed=random_seed,
            logger=logger,
            metrics=metrics,
            checkpoints_location=checkpoints_location,
            checkpoint_every=checkpoint_every,
            disable_tqdm=disable_tqdm,
            name=name,
            profiling_wait_epochs=profiling_wait_epochs,
            profiling_warmup_epochs=profiling_warmup_epochs,
            ray_scaling_config=ray_scaling_config,
            ray_tune_config=ray_tune_config,
            ray_run_config=ray_run_config,
            ray_search_space=ray_search_space,
            ray_torch_config=ray_torch_config,
            ray_data_config=ray_data_config,
            ray_horovod_config=ray_horovod_config,
            from_checkpoint=from_checkpoint,
        )

        # TODO: remove as it is already in the trainer
        if self.checkpoints_location:
            Path(self.checkpoints_location).mkdir(exist_ok=True, parents=True)

        self.ray_run_config = ray.train.RunConfig(
            name=Path(self.config.outdir).name,
            storage_path=self.config.storage_path,
            log_to_file=False,
            failure_config=ray.train.FailureConfig(max_failures=2),
            checkpoint_config=ray.train.CheckpointConfig(
                num_to_keep=1
            ),  # keep only latest checkpoint
            sync_config=ray.train.SyncConfig(sync_artifacts=True),
        )

        self.ray_scaling_config = ray.train.ScalingConfig(
            num_workers=self.config.gpus if self.config.gpus > 0 else 1,
            use_gpu=self.config.gpus > 0,
            resources_per_worker={
                "CPU": max(1, self.config.ray_cpus // self.config.num_workers - 1),
                "GPU": int(self.config.gpus > 0),
            },  # -1 to avoid blocking
        )

    def create_dataloaders(
        self,
        train_dataset=None,
        validation_dataset=None,
        test_dataset=None,
    ) -> None:
        """
        Create train, validation and test dataloaders using the
        configuration provided in the Trainer constructor.
        Generally a user-defined method.

        Args:
            train_dataset (Dataset): training dataset object.
            validation_dataset (Optional[Dataset]): validation dataset object.
                Default None.
            test_dataset (Optional[Dataset]): test dataset object.
                Default None.
        """
        loaders = get_interleaved_dataloaders(
            world_size=self.strategy.global_world_size(),
            rank=self.strategy.global_rank(),
            config=self.config.model_dump(),
            use_cuda=True,
            use_ray=False,
        )
        self.train_dataloader = loaders["train"]
        self.validation_dataloader = loaders["valid"]

    def create_model_loss_optimizer(self) -> None:
        config = self.config.model_dump()
        model_kwargs = {
            "input_dim": len(X_FEATURES[config["dataset"]]),
            "num_classes": len(CLASS_LABELS[config["dataset"]]),
            "input_encoding": config["model"]["input_encoding"],
            "pt_mode": config["model"]["pt_mode"],
            "eta_mode": config["model"]["eta_mode"],
            "sin_phi_mode": config["model"]["sin_phi_mode"],
            "cos_phi_mode": config["model"]["cos_phi_mode"],
            "energy_mode": config["model"]["energy_mode"],
            "elemtypes_nonzero": ELEM_TYPES_NONZERO[config["dataset"]],
            "learned_representation_mode": config["model"][
                "learned_representation_mode"
            ],
            **config["model"][config["conv_type"]],
        }
        print(model_kwargs)
        self.model = MLPF(**model_kwargs)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["lr"])
        self.lr_scheduler = get_lr_schedule(
            config=config,
            opt=self.optimizer,
            epochs=config["num_epochs"],
            steps_per_epoch=len(self.train_dataloader),
            last_epoch=False,
        )

        trainable_params, nontrainable_params, _ = count_parameters(self.model)
        self.log(trainable_params, "trainable_params", kind="param")
        self.log(nontrainable_params, "nontrainable_params", kind="param")
        self.log(trainable_params + nontrainable_params, "total_params", kind="param")

        if self.strategy.is_distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        distribute_kwargs = self.get_default_distributed_kwargs()

        # Distributed model, optimizer, and scheduler
        (self.model, self.optimizer, self.lr_scheduler) = self.strategy.distributed(
            self.model, self.optimizer, self.lr_scheduler, **distribute_kwargs
        )

    def execute(self):
        super().execute(train_dataset=None)

    def train(self) -> None:
        config = self.config.model_dump()

        # TODO: define dynamically
        dtype = torch.float32
        comet_experiment = None

        import logging

        logging.warning(
            f"Passing rank: {self.strategy.local_rank()}. Device was {self.strategy.device()}"
        )
        # return
        rank = "cpu" if self.strategy.device() == "cpu" else self.strategy.local_rank()

        train_all_epochs(
            # The rank is misused by train_all_epochs, which uses it also as a device, making training
            # fail when the itwinai TorchTrainer detects Ray but Ray only uses one node without GPU.
            rank=rank,
            world_size=self.strategy.global_world_size(),
            model=self.model,
            optimizer=self.optimizer,
            train_loader=self.train_dataloader,
            valid_loader=self.validation_dataloader,
            num_epochs=config["num_epochs"],
            patience=config["patience"],
            outdir=config["outdir"],
            config=config,
            trainable=config["model"]["trainable"],
            dtype=dtype,
            start_epoch=1,  # Epochs must start from 1
            lr_schedule=self.lr_scheduler,
            use_ray=isinstance(self.strategy, RayTorchDistributedStrategy),
            checkpoint_freq=config["checkpoint_freq"],
            comet_experiment=comet_experiment,
            comet_step_freq=config["comet_step_freq"],
            val_freq=config["val_freq"],
            save_attention=config["save_attention"],
            checkpoint_dir=self.checkpoints_location,
        )
