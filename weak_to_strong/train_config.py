from dataclasses import dataclass, field
import json
import os

from typing import Callable, Optional, Union

import torch

from weak_to_strong.common import assert_type
from weak_to_strong.model_config import MODELS_DICT, ModelConfig
from weak_to_strong.loss import LOSS_DICT
from weak_to_strong.datasets import VALID_DATASETS


@dataclass
class TrainConfig:
    # training batch size (number of examples per update)
    batch_size: int = 32
    max_ctx: int = 1024
    ds_name: str = "sciq"
    # number of documents
    n_train_docs: int = 20000
    n_inference_docs: int = 0
    n_test_docs: int = 10000
    model_size: str = "gpt2"
    lr: Optional[float] = None
    # Use the default learning rate times this factor, or ignore and use `lr` if set
    lr_factor: float = 1.0
    optim: Optional[str] = None
    epochs: int = 1
    seed: int = 0
    train_with_dropout: bool = False
    # if True, keep the transformer weights frozen and only train the head
    linear_probe: bool = False
    lr_schedule: str = "cosine_anneal"
    # Set to a very large value so that by default we don't do any intermediate evals but
    # still do final evals (which requires eval_every to be set to a non-zero, non-None value).
    eval_every: int = 10000000
    # Similar to HF trainer load_best_model_at_end behavior
    # https://huggingface.co/docs/transformers/main_classes/trainer
    # always uses AUROC against supervision
    load_best_model_at_end: bool = False
    save_total_limit: Optional[int] = 1
    disable_lora: bool = False
    # Using model_cfg_name allows you to use the model config specified by
    # model_cfg_name but request a different model from the hub
    model_cfg_name: Optional[str] = None
    # loss is only configurable in w2s runs
    loss: Optional[str] = None
    # If you pass weak_labels_path, we will use that path for labels.
    # Otherwise we train on ground truth.
    weak_labels_path: Optional[str] = None
    take_test_from_train: bool = False
    store_grads: bool = False
    d_downsample: Union[int, str] = "sqrt"
    store_hiddens: bool = False

    ### the following args shouldn't affect results ###

    force_retrain: bool = False
    # number of examples per forward pass per device
    minibatch_size_per_replica: Optional[int] = None
    results_folder: str = "/tmp/results"
    # The subfolder in results_folder to save the results to
    sweep_subfolder: str = "default"
    # non-positive values mean we don't save any checkpoints
    save_every: int = 1000000
    skip_if_exists: bool = False
    print_every: int = 10
    eval_batch_size: int = 32
    disable_gradient_checkpointing: bool = False

    ### not passed in by the user ###

    is_w2s: bool = field(init=False)
    model_config: ModelConfig = field(init=False)
    loss_fn: Callable = field(init=False)

    def __post_init__(self):
        assert (
            self.ds_name in VALID_DATASETS
        ), f"Unknown dataset {self.ds_name} not in {VALID_DATASETS}"

        self.is_w2s = self.weak_labels_path is not None
        self.loss = self.loss or ("kl" if self.is_w2s else "xent")

        print(f"model_size: {self.model_size}, model_cfg_name: {self.model_cfg_name}")
        mcfg = MODELS_DICT[self.model_cfg_name or self.model_size].copy()
        if self.disable_lora and "lora_modules" in mcfg:
            del mcfg["lora_modules"]
        # override the model name if we're borrowing a model config from a different model
        if self.model_cfg_name is not None:
            mcfg["name"] = self.model_size
        if self.disable_gradient_checkpointing:
            mcfg["gradient_checkpointing"] = False
        self.model_config = ModelConfig(**mcfg)
        self.eval_batch_size = (
            self.eval_batch_size or self.model_config.eval_batch_size
        ) or self.batch_size
        # this is per device!
        self.minibatch_size_per_replica = (
            self.minibatch_size_per_replica
            or self.model_config.minibatch_size_per_replica
        ) or self.batch_size
        if self.model_config.model_parallel:
            print(f"Using model parallelism for {self.model_size}")

        use_model_default_lr = self.lr is None
        if use_model_default_lr:
            # NOTE: LRs are not super tuned but are best for bs=32,
            # so we use a simple linear scaling otherwise
            # https://stackoverflow.com/questions/53033556/how-should-the-learning-rate-change-as-the-batch-size-change
            self.lr = self.model_config.default_lr * self.batch_size / 32
            if self.batch_size != 32:
                print(
                    f"Scaling learning rate linearly to {self.lr} based on batch"
                    f" size {self.batch_size}. "
                    "LRs were tuned for bs=32."
                )
            self.lr *= self.lr_factor

        self.optim = (self.optim or self.model_config.default_optimizer).lower()
        self.loss_fn = LOSS_DICT[self.loss]  # type: ignore

        # modify batch sizes if we're running multiple replicas
        if torch.cuda.device_count() > 1 and not self.model_config.model_parallel:
            self.minibatch_size_per_replica = min(
                assert_type(int, self.minibatch_size_per_replica)
                * torch.cuda.device_count(),
                self.batch_size,
            )
            self.eval_batch_size = min(torch.cuda.device_count(), self.eval_batch_size)

        if self.weak_labels_path and not self.weak_labels_path.endswith("weak_labels"):
            self.weak_labels_path = self.weak_labels_path + "/weak_labels"

    @property
    def effective_config(self):
        # grab the config items that may affect final results
        effective_config = {
            k: v
            for k, v in vars(self).items()
            if k
            not in {
                "force_retrain",
                "minibatch_size_per_replica",
                "results_folder",
                "sweep_subfolder",
                "save_every",
                "skip_if_exists",
                "print_every",
                "is_w2s",
                "model_config",
                "loss_fn",
                "eval_batch_size",
                "disable_gradient_checkpointing",
            }
        }
        assert (
            len(vars(self)) == 38
        ), f"!={len(vars(self))} Make sure to update effective_config if you modify TrainConfig!"

        if self.weak_labels_path is not None:
            weak_model_config = json.load(
                open(self.weak_labels_path.replace("weak_labels", "config.json"))
            )
            effective_config["weak_model_size"] = weak_model_config["model_size"]
            effective_config["weak_model"] = weak_model_config
        return effective_config

    @property
    def config_name(self):
        config = self.effective_config.copy()
        if "weak_model" in config:
            del config["weak_model"]
        return self.get_config_foldername(config)

    @property
    def save_path(self):
        return os.path.join(self.results_folder, self.sweep_subfolder, self.config_name)

    @staticmethod
    def get_config_foldername(config: dict) -> str:
        def shorten_key(key: str) -> str:
            return "".join(word[0] for word in key.split("_"))

        def shorten_value(value) -> str:
            if isinstance(value, bool):
                return "1" if value else "0"
            elif isinstance(value, str):
                value = value.split("/")[-1]
                if "_" in value:
                    return "_".join(word[:4] for word in value.split("_"))
                else:
                    return value
            else:
                return str(value)

        return "-".join(
            f"{shorten_key(k)}={shorten_value(v)}" for k, v in sorted(config.items())
        )
