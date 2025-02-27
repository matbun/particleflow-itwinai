"""itwinai integration of MLPF"""

import glob
import json
import uuid
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Literal,
    Optional,
    Union,
)
from timeit import default_timer as timer

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray.train
import ray.train.torch
import ray.tune
import sklearn
import sklearn.metrics
import torch
import torch.nn as nn
from itwinai.loggers import EpochTimeTracker, Logger
from itwinai.torch.config import TrainingConfiguration
from itwinai.torch.monitoring.monitoring import measure_gpu_utilization
from itwinai.torch.profiling.profiler import profile_torch_trainer
from itwinai.torch.trainer import TorchTrainer as ItwinaiTorchTrainer
from itwinai.torch.type import Batch, Metric
from ray.train import DataConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchConfig
from ray.tune import TuneConfig
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from mlpf.model.losses import mlpf_loss
from mlpf.model.mlpf import MLPF, set_save_attention
from mlpf.model.PFDataset import (
    Collater,
    InterleavedIterator,
    PFDataset,
    set_worker_sharing_strategy,
)
from mlpf.model.utils import (
    CLASS_LABELS,
    ELEM_TYPES_NONZERO,
    X_FEATURES,
    count_parameters,
    get_lr_schedule,
    unpack_predictions,
    unpack_target,
)

# from itwinai.components import DataGetter as ItwinaiDataGetter


if TYPE_CHECKING:
    from ray.train.horovod import HorovodConfig

# Disable GUI
matplotlib.use("agg")

# class PFDatasetGetter(ItwinaiDataGetter):
#     """Instantiate and return a dataset."""


def get_histogram_figure(tensor: torch.Tensor, bins="auto") -> plt.Figure:
    """Generates a Matplotlib figure for a histogram of the given tensor.

    Args:
        tensor (torch.Tensor): The input tensor for which the histogram is created.
        bins (str or int): Number of bins (default: "auto" for automatic bin selection).

    Returns:
        plt.Figure: The generated histogram figure.
    """
    tensor = tensor.detach().cpu().numpy()  # Convert tensor to NumPy array

    # Compute histogram
    counts, bin_edges = np.histogram(tensor, bins=bins)

    # Create figure and axis
    fig, ax = plt.subplots()
    ax.bar(
        bin_edges[:-1],
        counts,
        width=np.diff(bin_edges),
        align="edge",
        color="blue",
        edgecolor="black",
        alpha=0.75,
    )

    # Labels and title
    ax.set_title("Histogram")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

    plt.close(fig)

    return fig


def visualize_confusion_matrix(
    matrix,
    title="Confusion Matrix",
    row_label="True",
    column_label="Predicted",
    epoch=None,
    cmap=plt.cm.Blues,
    normalize=False,
    class_names=None,
    figsize=(10, 8),
    include_values=True,
    value_format=None,
    colorbar=True,
    rotate_xticks=45,
):
    """Generate a figure for the confusion matrix with enhanced customization options.

    Args:
        matrix: Confusion matrix (numpy array)
        title: Title of the confusion matrix plot
        row_label: Label for the y-axis (true labels)
        column_label: Label for the x-axis (predicted labels)
        epoch: Optional epoch number to include in the title
        cmap: Colormap to use (default: plt.cm.Blues)
        normalize: Whether to normalize the confusion matrix (default: False)
        class_names: Optional list of class names (default: None, will use indices 0-N)
        figsize: Figure size as (width, height) tuple (default: (10, 8))
        include_values: Whether to display values in cells (default: True)
        value_format: Format string for cell values
            (default: 'd' for integers, '.2f' for floats)
        colorbar: Whether to include a colorbar (default: True)
        rotate_xticks: Rotation angle for x-tick labels (default: 45)

    Returns:
        Figure object of the confusion matrix
    """
    # Create a copy of the matrix to avoid modifying the original
    cm = matrix.copy()

    # Normalize if requested
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + np.finfo(float).eps)
        if value_format is None:
            value_format = ".2f"
    elif value_format is None:
        value_format = "d"

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Display the matrix
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

    # Create title, potentially including epoch
    full_title = title
    if epoch is not None:
        full_title += f" (Epoch {epoch})"
    ax.set_title(full_title, fontsize=14, fontweight="bold")

    # Add colorbar if requested
    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        if normalize:
            cbar.set_label("Percentage", rotation=270, labelpad=15)
        else:
            cbar.set_label("Count", rotation=270, labelpad=15)

    # Set class names
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    # Set ticks and labels
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Rotate x-tick labels if specified
    plt.setp(
        ax.get_xticklabels(), rotation=rotate_xticks, ha="right", rotation_mode="anchor"
    )

    # Set axis labels
    ax.set_ylabel(row_label, fontsize=12)
    ax.set_xlabel(column_label, fontsize=12)

    # Loop over data dimensions and create text annotations
    if include_values:
        thresh = cm.max() / 2.0
        for i, j in np.ndindex(cm.shape):
            if value_format == "d":
                cell_text = format(int(cm[i, j]), value_format)
            else:
                cell_text = format(cm[i, j], value_format)

            ax.text(
                j,
                i,
                cell_text,
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    plt.tight_layout()
    plt.close(fig)
    return fig


class MLPFTrainer2(ItwinaiTorchTrainer):
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

        logging.info(self.config.model_dump())

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

        use_gpu = self.config.gpus > 0
        num_workers = self.config.gpus if use_gpu else 1
        resources = {
            "CPU": max(1, self.config.ray_cpus // num_workers - 1),
            "GPU": int(use_gpu),
        }  # -1 to avoid blocking
        print(f"RAY_RESOURCES_PER_WORKER: {resources}")
        self.ray_scaling_config = ray.train.ScalingConfig(
            num_workers=num_workers,
            use_gpu=use_gpu,
            resources_per_worker=resources,
        )

        # Initial epoch: here the convention is to start from 1
        self.epoch = 1

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
        loaders = self.get_interleaved_dataloaders()
        self.train_dataloader = loaders["train"]
        self.validation_dataloader = loaders["valid"]

    def get_interleaved_dataloaders(self):
        loaders = {}
        config = self.config.model_dump()
        for split in ["train", "valid"]:  # build train, valid dataset and dataloaders
            loaders[split] = []
            for type_ in config[f"{split}_dataset"][config["dataset"]]:
                dataset = []
                for sample in config[f"{split}_dataset"][config["dataset"]][type_][
                    "samples"
                ]:
                    version = config[f"{split}_dataset"][config["dataset"]][type_][
                        "samples"
                    ][sample]["version"]
                    split_configs = config[f"{split}_dataset"][config["dataset"]][
                        type_
                    ]["samples"][sample]["splits"]
                    print("split_configs", split_configs)

                    nevents = None
                    if config[f"n{split}"] is not None:
                        nevents = config[f"n{split}"] // len(split_configs)

                    for split_config in split_configs:
                        ds = PFDataset(
                            config["data_dir"],
                            f"{sample}/{split_config}:{version}",
                            split,
                            num_samples=nevents,
                            sort=config["sort_data"],
                        ).ds

                        if self.strategy.is_main_worker:
                            logging.info(f"{split}_dataset: {sample}, {len(ds)}")

                        dataset.append(ds)
                dataset = torch.utils.data.ConcatDataset(dataset)

                # build dataloaders
                batch_size = (
                    config[f"{split}_dataset"][config["dataset"]][type_]["batch_size"]
                    * config["gpu_batch_multiplier"]
                )
                loader = self.strategy.create_dataloader(
                    dataset,
                    batch_size=batch_size,
                    collate_fn=Collater(
                        [
                            "X",
                            "ytarget",
                            "ytarget_pt_orig",
                            "ytarget_e_orig",
                            "genjets",
                            "targetjets",
                        ],
                        ["genmet"],
                    ),
                    num_workers=config["num_workers"],
                    prefetch_factor=config["prefetch_factor"],
                    shuffle=split == "train",
                    generator=self.torch_rng,
                    # pin_memory=use_cuda,
                    # pin_memory_device="cuda:{}".format(rank) if use_cuda else "",
                    drop_last=True,
                    worker_init_fn=set_worker_sharing_strategy,
                )

                loaders[split].append(loader)

            loaders[split] = InterleavedIterator(loaders[split])
        return loaders

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

    def _set_epoch_dataloaders(self, epoch: int):
        """Sets epoch in the distributed sampler of a dataloader when using it.
        Conside that now the dataloaders are of type InterleavedIterator.
        """
        if self.strategy.is_distributed:
            for loader in self.train_dataloader.data_loaders:
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(epoch)
            if self.validation_dataloader is not None:
                for loader in self.validation_dataloader.data_loaders:
                    if isinstance(loader.sampler, DistributedSampler):
                        loader.sampler.set_epoch(epoch)
            if self.test_dataloader is not None:
                for loader in self.test_dataloader.data_loaders:
                    if isinstance(loader.sampler, DistributedSampler):
                        loader.sampler.set_epoch(epoch)

    def set_epoch(self) -> None:
        """Set current epoch at the beginning of training."""
        if self.profiler is not None and self.epoch > 1:
            # We don't want to start stepping until after the first epoch
            self.profiler.step()

        # The sheduler is already stepping at each batch
        # if self.lr_scheduler:
        #     self.lr_scheduler.step()

        self._set_epoch_dataloaders(self.epoch - 1)

    # @profile_torch_trainer
    @measure_gpu_utilization
    def train(self) -> None:
        # TODO: define dynamically
        self.config.dtype = torch.float32
        self.config.comet_experiment = None
        self.config.device_type = "cpu" if self.strategy.device() == "cpu" else "cuda"

        # run per-worker setup here so all processes / threads get configured.
        # Ignore divide by 0 errors
        np.seterr(divide="ignore", invalid="ignore")

        # t0_initial = time.time()

        # epoch_time_tracker: EpochTimeTracker | None = None
        # if self.strategy.is_main_worker:
        num_nodes = int(os.environ.get("SLURM_NNODES", 1))
        epoch_time_output_dir = Path("scalability-metrics/epoch-time")
        epoch_time_file_name = f"{uuid.uuid4()}_{self.strategy.name}_{num_nodes}N.csv"
        epoch_time_output_path = epoch_time_output_dir / epoch_time_file_name
        epoch_time_tracker = EpochTimeTracker(
            strategy_name=self.strategy.name,
            save_path=epoch_time_output_path,
            num_nodes=num_nodes,
        )

        # Early stopping setup
        stale_epochs = torch.tensor(0, device=self.device)

        self.scaler = torch.amp.GradScaler()

        for self.epoch in range(self.epoch, self.epochs + 1):
            epoch_start_time = time.time()

            self.set_epoch()

            lt = timer()

            # Training epoch
            losses_train = self.train_epoch()
            train_time = time.time() - epoch_start_time

            # if self.strategy.is_main_worker:
            #     assert epoch_time_tracker is not None
            epoch_time_tracker.add_epoch_time(self.epoch - 1, timer() - lt)

            # Validation epoch
            losses_valid = self.validation_epoch()
            valid_time = time.time() - train_time - epoch_start_time
            total_time = time.time() - epoch_start_time

            # # Metrics logging
            # for loss_name in losses_train.keys():
            #     self.log(
            #         item=losses_train[loss_name],
            #         identifier="epoch_train_loss_" + loss_name,
            #         kind="metric",
            #         step=self.epoch,
            #     )
            # for loss_name in losses_valid.keys():
            #     self.log(
            #         item=losses_train[loss_name],
            #         identifier="epoch_valid_loss_" + loss_name,
            #         kind="metric",
            #         step=self.epoch,
            #     )
            # self.log(
            #     item=self.lr_scheduler.get_last_lr()[0],
            #     identifier="learning_rate",
            #     kind="metric",
            #     step=self.epoch,
            # )
            # # Epoch times
            # self.log(
            #     item=train_time,
            #     identifier="epoch_train_time",
            #     kind="metric",
            #     step=self.epoch,
            # )
            # self.log(
            #     item=valid_time,
            #     identifier="epoch_valid_time",
            #     kind="metric",
            #     step=self.epoch,
            # )
            # self.log(
            #     item=total_time,
            #     identifier="epoch_total_time",
            #     kind="metric",
            #     step=self.epoch,
            # )

            # # Checkpointing
            # best_ckpt_path = None
            # periodic_ckpt_path = None
            # if self.strategy.is_main_worker:
            #     # Save best model if validation loss improved
            #     if losses_valid["Total"] < self.best_validation_loss:
            #         self.best_validation_loss = losses_valid["Total"]
            #         stale_epochs = 0
            #         best_ckpt_path = self.save_checkpoint(
            #             name="best_model",
            #             best_validation_loss=self.best_validation_loss,
            #         )
            #     else:
            #         stale_epochs += 1

            #     # Periodic checkpointing
            #     periodic_ckpt_path = self.save_checkpoint(
            #         name=f"epoch_{self.epoch:02d}-{losses_valid['Total']:.6f}"
            #     )

            # # Save epoch stats to JSON
            # # TODO: remove this block
            # history_path = Path(self.config.outdir) / "history"
            # history_path.mkdir(parents=True, exist_ok=True)
            # stats = {
            #     "train": losses_train,
            #     "valid": losses_valid,
            #     "epoch_train_time": train_time,
            #     "epoch_valid_time": valid_time,
            #     "epoch_total_time": total_time,
            # }
            # with open(f"{history_path}/epoch_{self.epoch}.json", "w") as f:
            #     json.dump(stats, f)

            # Ray report
            self.ray_report(
                metrics={
                    "loss": losses_train["Total"],
                    "val_loss": losses_valid["Total"],
                    "epoch": self.epoch,
                    **{f"train_{k}": v for k, v in losses_train.items()},
                    **{f"valid_{k}": v for k, v in losses_valid.items()},
                },
                # checkpoint_dir=best_ckpt_path or periodic_ckpt_path,
            )

            # # Test epoch
            # if self.test_every and self.epoch % self.test_every == 0:
            #     self.test_epoch()

            # # Check early stopping
            # if stale_epochs > self.config.patience:
            #     logging.info(f"Breaking due to stale epochs: {stale_epochs}")
            #     break

            # Sync workers
            self.strategy.barrier()

        # if self.strategy.is_main_worker:
        #     assert epoch_time_tracker is not None
        epoch_time_tracker.save()

    def train_epoch(self):
        """Run one training epoch

        Returns:
            Dict | None: Dictionary of epoch losses, if on the main worker.
                Retuns None when not on main worker (global rank == 0).
        """
        self.model.train()
        epoch_loss = defaultdict(lambda: torch.tensor(0.0, device=self.device))
        batch_counter = 0

        progress_bar = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc=f"Epoch {self.epoch} train loop on rank={self.strategy.global_rank()}",
            disable=self.disable_tqdm or not self.strategy.is_main_worker,
            leave=False,  # Set this to true to see how many batches were used
        )

        for itrain, batch in progress_bar:
            # Iterate over training batches
            loss = self.train_step(batch=batch, batch_idx=itrain)

            # Accumulate losses
            for loss_name in loss:
                epoch_loss[loss_name] += loss[loss_name]
            batch_counter += 1

            # Update global step counter
            self.train_glob_step += 1

        # Reduce losses only on the main worker -- save communication time compare to allreduce
        num_steps = torch.tensor(batch_counter, device=self.device, dtype=torch.float32)
        tot_steps = self.strategy.gather(num_steps, dst_rank=0)
        tot_losses = {}

        for loss_name in epoch_loss:
            tot_losses[loss_name] = self.strategy.gather(
                epoch_loss[loss_name], dst_rank=0
            )

        if self.strategy.global_rank() == 0:
            # The gathered values are available only on the main worker (global rank == 0)
            tot_steps = torch.sum(torch.stack(tot_steps))
            tot_epoch_loss = {}
            for loss_name in epoch_loss:
                tot_epoch_loss[loss_name] = (
                    torch.sum(torch.stack(tot_losses[loss_name])) / tot_steps
                ).item()
            return tot_epoch_loss

        # Otherwise, report epoch_loss on workers with rank != 0
        for loss_name in epoch_loss:
            epoch_loss[loss_name] = epoch_loss[loss_name].item()
        return epoch_loss

    def train_step(self, batch: Batch, batch_idx: int):
        """Run one optimization step.

        Args:
            batch (torch.tensor): training batch.
            batch_idx (int): batch number.
        """
        batch = batch.to(self.device, non_blocking=True)

        with torch.autocast(
            device_type=self.config.device_type,
            dtype=self.config.dtype,
            enabled=self.config.device_type == "cuda",
        ):
            # Model step
            ypred_raw = self.model(batch.X, batch.mask)
            ypred = unpack_predictions(ypred_raw)
            ytarget = unpack_target(batch.ytarget, self.model)
            loss_opt, loss = mlpf_loss(ytarget, ypred, batch)

        # Optimizer step
        for param in self.model.parameters():
            # Clear gradients
            param.grad = None
        # Backward pass and optimization
        self.scaler.scale(loss_opt).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.lr_scheduler:
            self.lr_scheduler.step()

        # # Log step
        # # get the number of elements, excluding padded elements
        # num_elems = batch.X[batch.mask].shape[0]

        # self.log(
        #     identifier="step/loss",
        #     item=loss["Total"] / num_elems,
        #     kind="metric",
        #     step=self.train_glob_step,
        #     batch_idx=batch_idx,
        # )
        # self.log(
        #     identifier="step/num_elems",
        #     item=num_elems,
        #     kind="metric",
        #     step=self.train_glob_step,
        #     batch_idx=batch_idx,
        # )
        # self.log(
        #     identifier="step/num_batch",
        #     item=batch.X.shape[0],
        #     kind="metric",
        #     step=self.train_glob_step,
        #     batch_idx=batch_idx,
        # )
        # self.log(
        #     identifier="step/learning_rate",
        #     item=self.lr_scheduler.get_last_lr()[0],
        #     kind="metric",
        #     step=self.train_glob_step,
        #     batch_idx=batch_idx,
        # )
        return loss

    def validation_epoch(self):
        self.model.eval()
        epoch_loss = defaultdict(lambda: torch.tensor(0.0, device=self.device))
        batch_counter = 0

        # # Confusion matrix tracking
        # cm_X_target = np.zeros((13, 13))
        # cm_X_pred = np.zeros((13, 13))
        # cm_id = np.zeros((13, 13))

        progress_bar = tqdm(
            enumerate(self.validation_dataloader),
            total=len(self.validation_dataloader),
            desc=f"Epoch {self.epoch} eval loop on rank={self.strategy.global_rank()}",
            disable=self.disable_tqdm or not self.strategy.is_main_worker,
            leave=False,  # Set this to true to see how many batches were used
        )

        for ival, batch in progress_bar:
            batch = batch.to(self.device, non_blocking=True)

            # # Save attention on first batch if requested
            # if (
            #     self.config.save_attention
            #     and self.strategy.is_main_worker
            #     and ival == 0
            # ):
            #     set_save_attention(self.model, self.config.outdir, True)
            # else:
            #     set_save_attention(self.model, self.config.outdir, False)

            # Validation step
            batch = batch.to(self.device, non_blocking=True)

            with torch.autocast(
                device_type=self.config.device_type,
                dtype=self.config.dtype,
                enabled=self.config.device_type == "cuda",
            ):
                # Model step
                ypred_raw = self.model(batch.X, batch.mask)
                ypred = unpack_predictions(ypred_raw)
                ytarget = unpack_target(batch.ytarget, self.model)
                _, loss = mlpf_loss(ytarget, ypred, batch)

            # # Update confusion matrices
            # cm_X_target += sklearn.metrics.confusion_matrix(
            #     batch.X[:, :, 0][batch.mask].detach().cpu().numpy(),
            #     ytarget["cls_id"][batch.mask].detach().cpu().numpy(),
            #     labels=range(13),
            # )
            # cm_X_pred += sklearn.metrics.confusion_matrix(
            #     batch.X[:, :, 0][batch.mask].detach().cpu().numpy(),
            #     ypred["cls_id"][batch.mask].detach().cpu().numpy(),
            #     labels=range(13),
            # )
            # cm_id += sklearn.metrics.confusion_matrix(
            #     ytarget["cls_id"][batch.mask].detach().cpu().numpy(),
            #     ypred["cls_id"][batch.mask].detach().cpu().numpy(),
            #     labels=range(13),
            # )

            # # Save validation plots for first batch
            # if self.strategy.is_main_worker and ival == 0:
            #     self.validation_plots(batch=batch, ypred_raw=ypred_raw)

            # Accumulate losses
            for loss_name in loss:
                epoch_loss[loss_name] += loss[loss_name]
            batch_counter += 1

            # Update global step counter
            self.validation_glob_step += 1

        # # Log confusion matrices
        # self.log_all_confusion_matrices(
        #     cm_X_target=cm_X_target,
        #     cm_X_pred=cm_X_pred,
        #     cm_id=cm_id,
        # )

        # Reduce losses only on the main worker -- save communication time compare to allreduce
        num_steps = torch.tensor(batch_counter, device=self.device, dtype=torch.float32)
        tot_steps = self.strategy.gather(num_steps, dst_rank=0)
        tot_losses = {}

        for loss_name in epoch_loss:
            tot_losses[loss_name] = self.strategy.gather(
                epoch_loss[loss_name], dst_rank=0
            )

        if self.strategy.global_rank() == 0:
            # The gathered values are available only on the main worker (global rank == 0)
            tot_steps = torch.sum(torch.stack(tot_steps))
            tot_epoch_loss = {}
            for loss_name in epoch_loss:
                tot_epoch_loss[loss_name] = (
                    (torch.sum(torch.stack(tot_losses[loss_name])) / tot_steps)
                    .cpu()
                    .item()
                )
            return tot_epoch_loss

        # Otherwise, report epoch_loss on workers with rank != 0
        for loss_name in epoch_loss:
            epoch_loss[loss_name] = epoch_loss[loss_name].item()
        return epoch_loss

    def validation_plots(self, batch, ypred_raw):
        X = batch.X[batch.mask].cpu()
        ytarget_flat = batch.ytarget[batch.mask].cpu()
        ypred_binary = ypred_raw[0][batch.mask].detach().cpu()
        ypred_binary_cls = torch.argmax(ypred_binary, axis=-1)
        ypred_cls = ypred_raw[1][batch.mask].detach().cpu()
        ypred_p4 = ypred_raw[2][batch.mask].detach().cpu()

        arr = torch.concatenate(
            [X, ytarget_flat, ypred_binary, ypred_cls, ypred_p4],
            axis=-1,
        ).numpy()
        df = pd.DataFrame(arr)
        df.to_parquet(f"{self.config.outdir}/batch0_epoch{self.epoch}.parquet")

        sig_prob = torch.softmax(ypred_binary, axis=-1)[:, 1].to(torch.float32)
        for xcls in np.unique(X[:, 0]):
            fig = plt.figure()
            msk = X[:, 0] == xcls
            etarget = ytarget_flat[msk & (ytarget_flat[:, 0] != 0), 6]
            epred = ypred_p4[msk & (ypred_binary_cls != 0), 4]
            b = np.linspace(-2, 2, 100)
            plt.hist(etarget, bins=b, histtype="step")
            plt.hist(epred, bins=b, histtype="step")
            plt.xlabel("log [E/E_elem]")
            plt.yscale("log")
            self.log(
                item=fig,
                identifier="energy_elemtype{}".format(int(xcls)),
                kind="figure",
                step=self.epoch,
            )
            plt.close(fig)

            fig = plt.figure()
            msk = X[:, 0] == xcls
            pt_target = ytarget_flat[msk & (ytarget_flat[:, 0] != 0), 2]
            pt_pred = ypred_p4[msk & (ypred_binary_cls != 0), 0]
            b = np.linspace(-2, 2, 100)
            plt.hist(etarget, bins=b, histtype="step")
            plt.hist(epred, bins=b, histtype="step")
            plt.xlabel("log [pt/pt_elem]")
            plt.yscale("log")
            self.log(
                item=fig,
                identifier="pt_elemtype{}".format(int(xcls)),
                kind="figure",
                step=self.epoch,
            )
            plt.close(fig)

            fig = plt.figure(figsize=(5, 5))
            msk = (
                (X[:, 0] == xcls) & (ytarget_flat[:, 0] != 0) & (ypred_binary_cls != 0)
            )
            etarget = ytarget_flat[msk, 6]
            epred = ypred_p4[msk, 4]
            b = np.linspace(-2, 2, 100)
            plt.hist2d(
                etarget, epred, bins=b, cmap="hot", norm=matplotlib.colors.LogNorm()
            )
            plt.plot([-4, 4], [-4, 4], color="black", ls="--")
            plt.xlabel("log [E_target/E_elem]")
            plt.ylabel("log [E_pred/E_elem]")
            self.log(
                item=fig,
                identifier="energy_elemtype{}_corr".format(int(xcls)),
                kind="figure",
                step=self.epoch,
            )
            plt.close(fig)

            fig = plt.figure(figsize=(5, 5))
            msk = (
                (X[:, 0] == xcls) & (ytarget_flat[:, 0] != 0) & (ypred_binary_cls != 0)
            )
            pt_target = ytarget_flat[msk, 2]
            pt_pred = ypred_p4[msk, 0]
            b = np.linspace(-2, 2, 100)
            plt.hist2d(
                pt_target,
                pt_pred,
                bins=b,
                cmap="hot",
                norm=matplotlib.colors.LogNorm(),
            )
            plt.plot([-4, 4], [-4, 4], color="black", ls="--")
            plt.xlabel("log [pt_target/pt_elem]")
            plt.ylabel("log [pt_pred/pt_elem]")
            self.log(
                item=fig,
                identifier="pt_elemtype{}_corr".format(int(xcls)),
                kind="figure",
                step=self.epoch,
            )
            plt.close(fig)

            fig = plt.figure(figsize=(5, 5))
            msk = (
                (X[:, 0] == xcls) & (ytarget_flat[:, 0] != 0) & (ypred_binary_cls != 0)
            )
            eta_target = ytarget_flat[msk, 3]
            eta_pred = ypred_p4[msk, 1]
            b = np.linspace(-6, 6, 100)
            plt.hist2d(
                eta_target,
                eta_pred,
                bins=b,
                cmap="hot",
                norm=matplotlib.colors.LogNorm(),
            )
            plt.plot([-6, 6], [-6, 6], color="black", ls="--")
            plt.xlabel("eta_target")
            plt.ylabel("eta_pred")
            self.log(
                item=fig,
                identifier="eta_elemtype{}_corr".format(int(xcls)),
                kind="figure",
                step=self.epoch,
            )
            plt.close(fig)

            fig = plt.figure(figsize=(5, 5))
            msk = (
                (X[:, 0] == xcls) & (ytarget_flat[:, 0] != 0) & (ypred_binary_cls != 0)
            )
            sphi_target = ytarget_flat[msk, 4]
            sphi_pred = ypred_p4[msk, 2]
            b = np.linspace(-2, 2, 100)
            plt.hist2d(
                sphi_target,
                sphi_pred,
                bins=b,
                cmap="hot",
                norm=matplotlib.colors.LogNorm(),
            )
            plt.plot([-2, 2], [-2, 2], color="black", ls="--")
            plt.xlabel("sin_phi_target")
            plt.ylabel("sin_phi_pred")
            self.log(
                item=fig,
                identifier="sphi_elemtype{}_corr".format(int(xcls)),
                kind="figure",
                step=self.epoch,
            )
            plt.close(fig)

            fig = plt.figure(figsize=(5, 5))
            msk = (
                (X[:, 0] == xcls) & (ytarget_flat[:, 0] != 0) & (ypred_binary_cls != 0)
            )
            cphi_target = ytarget_flat[msk, 5]
            cphi_pred = ypred_p4[msk, 3]
            b = np.linspace(-2, 2, 100)
            plt.hist2d(
                cphi_target,
                cphi_pred,
                bins=b,
                cmap="hot",
                norm=matplotlib.colors.LogNorm(),
            )
            plt.plot([-2, 2], [-2, 2], color="black", ls="--")
            plt.xlabel("cos_phi_target")
            plt.ylabel("cos_phi_pred")
            self.log(
                item=fig,
                identifier="cphi_elemtype{}_corr".format(int(xcls)),
                kind="figure",
                step=self.epoch,
            )
            plt.close(fig)

            fig = plt.figure()
            msk = X[:, 0] == xcls
            b = np.linspace(0, 1, 100)
            plt.hist(sig_prob[msk & (ytarget_flat[:, 0] == 0)], bins=b, histtype="step")
            plt.hist(sig_prob[msk & (ytarget_flat[:, 0] != 0)], bins=b, histtype="step")
            plt.xlabel("particle proba")
            self.log(
                item=fig,
                identifier="sig_proba_elemtype{}".format(int(xcls)),
                kind="figure",
                step=self.epoch,
            )
            plt.close(fig)

        try:
            self.log(
                item=get_histogram_figure(
                    torch.clamp(batch.ytarget[batch.mask][:, 2], -10, 10)
                ),
                identifier="pt_target",
                kind="figure",
                step=self.epoch,
            )
            self.log(
                item=get_histogram_figure(
                    torch.clamp(ypred_raw[2][batch.mask][:, 0], -10, 10)
                ),
                identifier="pt_pred",
                kind="figure",
                step=self.epoch,
            )
            ratio = (ypred_raw[2][batch.mask][:, 0] / batch.ytarget[batch.mask][:, 2])[
                batch.ytarget[batch.mask][:, 0] != 0
            ]
            self.log(
                item=get_histogram_figure(torch.clamp(ratio, -10, 10)),
                identifier="pt_ratio",
                kind="figure",
                step=self.epoch,
            )

            self.log(
                item=get_histogram_figure(
                    torch.clamp(batch.ytarget[batch.mask][:, 3], -10, 10)
                ),
                identifier="eta_target",
                kind="figure",
                step=self.epoch,
            )

            self.log(
                item=get_histogram_figure(
                    torch.clamp(ypred_raw[2][batch.mask][:, 1], -10, 10)
                ),
                identifier="eta_pred",
                kind="figure",
                step=self.epoch,
            )

            ratio = (ypred_raw[2][batch.mask][:, 1] / batch.ytarget[batch.mask][:, 3])[
                batch.ytarget[batch.mask][:, 0] != 0
            ]
            self.log(
                item=get_histogram_figure(torch.clamp(ratio, -10, 10)),
                identifier="eta_ratio",
                kind="figure",
                step=self.epoch,
            )

            self.log(
                item=get_histogram_figure(
                    torch.clamp(batch.ytarget[batch.mask][:, 4], -10, 10)
                ),
                identifier="sphi_target",
                kind="figure",
                step=self.epoch,
            )

            self.log(
                item=get_histogram_figure(
                    torch.clamp(ypred_raw[2][batch.mask][:, 2], -10, 10)
                ),
                identifier="sphi_pred",
                kind="figure",
                step=self.epoch,
            )

            # Calculate and log sphi ratio
            ratio = (ypred_raw[2][batch.mask][:, 2] / batch.ytarget[batch.mask][:, 4])[
                batch.ytarget[batch.mask][:, 0] != 0
            ]
            self.log(
                item=get_histogram_figure(torch.clamp(ratio, -10, 10)),
                identifier="sphi_ratio",
                kind="figure",
                step=self.epoch,
            )

            self.log(
                item=get_histogram_figure(
                    torch.clamp(batch.ytarget[batch.mask][:, 5], -10, 10)
                ),
                identifier="cphi_target",
                kind="figure",
                step=self.epoch,
            )

            self.log(
                item=get_histogram_figure(
                    torch.clamp(ypred_raw[2][batch.mask][:, 3], -10, 10)
                ),
                identifier="cphi_pred",
                kind="figure",
                step=self.epoch,
            )

            ratio = (ypred_raw[2][batch.mask][:, 3] / batch.ytarget[batch.mask][:, 5])[
                batch.ytarget[batch.mask][:, 0] != 0
            ]
            self.log(
                item=get_histogram_figure(torch.clamp(ratio, -10, 10)),
                identifier="cphi_ratio",
                kind="figure",
                step=self.epoch,
            )

            self.log(
                item=get_histogram_figure(
                    torch.clamp(batch.ytarget[batch.mask][:, 6], -10, 10)
                ),
                identifier="energy_target",
                kind="figure",
                step=self.epoch,
            )

            self.log(
                item=get_histogram_figure(
                    torch.clamp(ypred_raw[2][batch.mask][:, 4], -10, 10)
                ),
                identifier="energy_pred",
                kind="figure",
                step=self.epoch,
            )

            ratio = (ypred_raw[2][batch.mask][:, 4] / batch.ytarget[batch.mask][:, 6])[
                batch.ytarget[batch.mask][:, 0] != 0
            ]
            self.log(
                item=get_histogram_figure(torch.clamp(ratio, -10, 10)),
                identifier="energy_ratio",
                kind="figure",
                step=self.epoch,
            )

            # tensorboard_writer.add_histogram(
            #     "pt_target",
            #     torch.clamp(batch.ytarget[batch.mask][:, 2], -10, 10),
            #     global_step=epoch,
            # )
            # tensorboard_writer.add_histogram(
            #     "pt_pred",
            #     torch.clamp(ypred_raw[2][batch.mask][:, 0], -10, 10),
            #     global_step=epoch,
            # )
            # ratio = (ypred_raw[2][batch.mask][:, 0] / batch.ytarget[batch.mask][:, 2])[
            #     batch.ytarget[batch.mask][:, 0] != 0
            # ]
            # tensorboard_writer.add_histogram(
            #     "pt_ratio", torch.clamp(ratio, -10, 10), global_step=epoch
            # )

            # tensorboard_writer.add_histogram(
            #     "eta_target",
            #     torch.clamp(batch.ytarget[batch.mask][:, 3], -10, 10),
            #     global_step=epoch,
            # )
            # tensorboard_writer.add_histogram(
            #     "eta_pred",
            #     torch.clamp(ypred_raw[2][batch.mask][:, 1], -10, 10),
            #     global_step=epoch,
            # )
            # ratio = (ypred_raw[2][batch.mask][:, 1] / batch.ytarget[batch.mask][:, 3])[
            #     batch.ytarget[batch.mask][:, 0] != 0
            # ]
            # tensorboard_writer.add_histogram(
            #     "eta_ratio", torch.clamp(ratio, -10, 10), global_step=epoch
            # )

            # tensorboard_writer.add_histogram(
            #     "sphi_target",
            #     torch.clamp(batch.ytarget[batch.mask][:, 4], -10, 10),
            #     global_step=epoch,
            # )
            # tensorboard_writer.add_histogram(
            #     "sphi_pred",
            #     torch.clamp(ypred_raw[2][batch.mask][:, 2], -10, 10),
            #     global_step=epoch,
            # )
            # ratio = (ypred_raw[2][batch.mask][:, 2] / batch.ytarget[batch.mask][:, 4])[
            #     batch.ytarget[batch.mask][:, 0] != 0
            # ]
            # tensorboard_writer.add_histogram(
            #     "sphi_ratio", torch.clamp(ratio, -10, 10), global_step=epoch
            # )

            # tensorboard_writer.add_histogram(
            #     "cphi_target",
            #     torch.clamp(batch.ytarget[batch.mask][:, 5], -10, 10),
            #     global_step=epoch,
            # )
            # tensorboard_writer.add_histogram(
            #     "cphi_pred",
            #     torch.clamp(ypred_raw[2][batch.mask][:, 3], -10, 10),
            #     global_step=epoch,
            # )
            # ratio = (ypred_raw[2][batch.mask][:, 3] / batch.ytarget[batch.mask][:, 5])[
            #     batch.ytarget[batch.mask][:, 0] != 0
            # ]
            # tensorboard_writer.add_histogram(
            #     "cphi_ratio", torch.clamp(ratio, -10, 10), global_step=epoch
            # )

            # tensorboard_writer.add_histogram(
            #     "energy_target",
            #     torch.clamp(batch.ytarget[batch.mask][:, 6], -10, 10),
            #     global_step=epoch,
            # )
            # tensorboard_writer.add_histogram(
            #     "energy_pred",
            #     torch.clamp(ypred_raw[2][batch.mask][:, 4], -10, 10),
            #     global_step=epoch,
            # )
            # ratio = (ypred_raw[2][batch.mask][:, 4] / batch.ytarget[batch.mask][:, 6])[
            #     batch.ytarget[batch.mask][:, 0] != 0
            # ]
            # tensorboard_writer.add_histogram(
            #     "energy_ratio", torch.clamp(ratio, -10, 10), global_step=epoch
            # )
        except ValueError as e:
            print(e)

        try:
            for attn in sorted(
                list(glob.glob(f"{self.config.outdir}/attn_conv_*.npz"))
            ):
                attn_name = os.path.basename(attn).split(".")[0]
                attn_matrix = np.load(attn)["att"]
                batch_size = min(attn_matrix.shape[0], 8)
                fig, axes = plt.subplots(
                    1, batch_size, figsize=((batch_size * 3, 1 * 3))
                )
                if isinstance(axes, matplotlib.axes._axes.Axes):
                    axes = [axes]
                for ibatch in range(batch_size):
                    plt.sca(axes[ibatch])
                    # plot the attention matrix of the first event in the batch
                    plt.imshow(
                        attn_matrix[ibatch].T,
                        cmap="hot",
                        norm=matplotlib.colors.LogNorm(),
                    )
                    plt.xticks([])
                    plt.yticks([])
                    plt.colorbar()
                    plt.title(
                        "event {}, m={:.2E}".format(
                            ibatch,
                            np.mean(attn_matrix[ibatch][attn_matrix[ibatch] > 0]),
                        )
                    )
                plt.suptitle(attn_name)
                self.log(
                    item=fig,
                    identifier=attn_name,
                    kind="figure",
                    step=self.epoch,
                )
        except ValueError as e:
            print(e)

    def log_all_confusion_matrices(
        self, cm_X_target, cm_X_pred, cm_id, normalize=False
    ) -> None:
        """
        Generate and return figures for all three confusion matrices.

        Args:
            cm_X_target: Confusion matrix for Element to Target
            cm_X_pred: Confusion matrix for Element to Prediction
            cm_id: Confusion matrix for Target to Prediction
            normalize: Whether to normalize the matrices
        """
        # Create the three figures
        fig_X_target = visualize_confusion_matrix(
            matrix=cm_X_target,
            title="Element to Target",
            row_label="X",
            column_label="target",
            epoch=self.epoch,
            normalize=normalize,
        )
        self.log(
            item=fig_X_target,
            identifier="cm_X_target",
            kind="figure",
            step=self.epoch,
        )

        fig_X_pred = visualize_confusion_matrix(
            matrix=cm_X_pred,
            title="Element to Prediction",
            row_label="X",
            column_label="pred",
            epoch=self.epoch,
            normalize=normalize,
        )
        self.log(
            item=fig_X_pred,
            identifier="cm_X_pred",
            kind="figure",
            step=self.epoch,
        )

        fig_id = visualize_confusion_matrix(
            matrix=cm_id,
            title="Target to Prediction",
            row_label="target",
            column_label="pred",
            epoch=self.epoch,
            normalize=normalize,
        )
        self.log(
            item=fig_id,
            identifier="cm_id",
            kind="figure",
            step=self.epoch,
        )

    def test_epoch(self):
        # TODO: need to be integrated
        pass
