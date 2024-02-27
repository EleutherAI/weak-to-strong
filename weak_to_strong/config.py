import torch
from typing import Optional

import yaml
import subprocess

from weak_to_strong.loss import logconf_loss_fn, product_loss_fn, xent_loss, kl_loss


def load_config(config_path="configs/default.yaml"):
    """
    Load the YAML configuration file.

    Parameters:
    - config_path (str): Path to the YAML configuration file.

    Returns:
    - dict: Configuration settings.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading the config file {config_path}: {e}")
        return {}


class ModelConfig:
    """
    Configuration class for the model.

    Args:
        name (str): The name of the model.
        memory (float):
            The memory required for the model in bytes.
        default_lr (float, optional):
            The default learning rate. Defaults to 1e-5.
        eval_batch_size (int, optional):
            The batch size for evaluation. Defaults to 32.
        minibatch_size_per_replica (int, optional):
            The minibatch size per device. Defaults to None.
        lora_modules (list[str], optional):
            The list of LORA modules. Defaults to None.
            If None, then LORA is not used.
        custom_kwargs (dict, optional):
            Arguments to pass to HF's from_pretrained(). Defaults to None.
        required_packages (list[str], optional):
            The list of additional packages required to run the model.
        gradient_checkpointing (bool, optional):
            Whether to use gradient checkpointing. Defaults to None.
        model_parallel (bool, optional):
            Whether to use model parallelism.
            Defaults to true if the memory requirement exceeds a threshold and
            there are multiple GPUs available.
            Model parallelism uses accelerate's automatic model sharding,
            while if model-parallel is false and you're using multiple GPUs,
            then it uses data parallelism.
        default_optimizer (str, optional):
            The default optimizer. Defaults to "adam".
        torch_dtype (str, optional):
            The torch data type. Defaults to None.
            If None and not set in custom_kwargs, then it defaults to
            "torch.bfloat16". We cast LoRA modules to fp32,
            and our training script wraps model calls in autocast to avoid dtype issues,
            and does not use gradscaling because we don't support fp16,
            and stores optimizer buffers in fp32.
    """

    CHECKPOINTING_MEMORY = 3e9
    MODEL_PARALLEL_FACTOR = 2
    name: str
    memory: float
    default_lr: float
    eval_batch_size: int
    minibatch_size_per_replica: int
    lora_modules: Optional[list[str]]
    custom_kwargs: dict
    gradient_checkpointing: bool
    model_parallel: bool
    default_optimizer: str

    def __init__(
        self,
        name: str,
        # memory, in bytes, of the model
        memory: float,
        default_lr: float = 4e-5,
        eval_batch_size: int = 32,
        minibatch_size_per_replica: Optional[int] = None,
        lora_modules: Optional[list[str]] = None,
        custom_kwargs: Optional[dict] = None,
        required_packages: Optional[list[str]] = None,
        gradient_checkpointing: Optional[bool] = None,
        model_parallel: Optional[bool] = None,
        default_optimizer: str = "adam",
        torch_dtype: Optional[str] = None,
    ):
        assert name is not None
        memory = float(memory)
        custom_kwargs = custom_kwargs or {}
        per_device_ram = torch.cuda.get_device_properties(0).total_memory
        n_devices = torch.cuda.device_count()
        if torch_dtype is not None:
            assert custom_kwargs.get("torch_dtype") is None
            custom_kwargs["torch_dtype"] = {
                "torch.float32": torch.float32,
                "torch.bfloat16": torch.bfloat16,
            }[torch_dtype]
        else:
            custom_kwargs["torch_dtype"] = torch.bfloat16
        if (
            not torch.cuda.is_bf16_supported() or lora_modules is None
        ) and custom_kwargs[  # we enforce fp32 for full finetuning
            "torch_dtype"
        ] == torch.bfloat16:
            custom_kwargs["torch_dtype"] = torch.float32
        self.name = name
        memory_util_est = memory
        if custom_kwargs["torch_dtype"] == torch.float32:
            memory_util_est *= 2
        # NOTE: this memory estimate doesn't account for the
        # optimizer, LoRA, checkpointing, seq len, batch size, etc.

        if model_parallel is None:
            model_parallel = (
                n_devices > 1
                and per_device_ram < self.MODEL_PARALLEL_FACTOR * memory_util_est
            )
        if gradient_checkpointing is None:
            gradient_checkpointing = memory_util_est > self.CHECKPOINTING_MEMORY

        if required_packages is not None:
            for pkg in required_packages:
                try:
                    __import__(pkg)
                except ImportError:
                    print(
                        f"Package `{pkg}` required for `{name}`. Attempting to install..."
                    )
                    subprocess.check_call(["pip", "install", pkg])
                    print(f"Successfully installed `{pkg}`.")

        if minibatch_size_per_replica is None:
            minibatch_size_per_replica = eval_batch_size
        self.memory = memory
        self.default_lr = default_lr
        self.eval_batch_size = eval_batch_size
        self.minibatch_size_per_replica = minibatch_size_per_replica
        self.lora_modules = lora_modules
        self.custom_kwargs = custom_kwargs
        self.gradient_checkpointing = gradient_checkpointing
        self.model_parallel = model_parallel
        self.default_optimizer = default_optimizer


MODELS_DICT: dict[str, dict] = {
    cfg["name"]: cfg for cfg in load_config("configs/models.yaml")["models"]
}


loss_dict = {
    "logconf": logconf_loss_fn(),
    "product": product_loss_fn(),
    "xent": xent_loss(),
    "kl": kl_loss(),
}

VALID_LOSSES: list[str] = list(loss_dict.keys())


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
