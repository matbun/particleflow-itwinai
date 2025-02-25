"""itwinai integration of MLPF"""

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
    # count_parameters,
)

from .training import configure_model_trainable, train_all_epochs

from itwinai.torch.trainer import TorchTrainer as ItwinaiTorchTrainer
# from itwinai.components import DataGetter as ItwinaiDataGetter

# class PFDatasetGetter(ItwinaiDataGetter):
#     """Instantiate and return a dataset."""


class MLPFTrainer(ItwinaiTorchTrainer):
    """itwinai trainer class for MLPF"""

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

        # TODO: move at the beginning of training explicitly
        configure_model_trainable(
            model=self.model, trainable=config["model"]["trainable"], is_training=True
        )

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

        # rank = self.strategy.global_rank() if self.strategy.is_distributed else "cpu"

        train_all_epochs(
            rank=self.strategy.global_rank(),
            world_size=self.strategy.global_world_size(),
            model=self.model,
            optimizer=self.optimizer,
            train_loader=self.train_dataloader,
            valid_loader=self.validation_dataloader,
            num_epochs=config["num_epochs"],
            patience=config["patience"],
            outdir=config["outdir"],
            trainable=config["model"]["trainable"],
            dtype=dtype,
            start_epoch=1,  # Epochs must start from 1
            lr_schedule=self.lr_scheduler,
            use_ray=False,
            checkpoint_freq=config["checkpoint_freq"],
            comet_experiment=comet_experiment,
            comet_step_freq=config["comet_step_freq"],
            val_freq=config["val_freq"],
            save_attention=config["save_attention"],
        )