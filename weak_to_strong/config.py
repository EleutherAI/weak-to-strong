import torch
from typing import Optional

import yaml

from weak_to_strong.loss import logconf_loss_fn, product_loss_fn, xent_loss, kl_loss


def load_config(config_path='configs/default.yaml'):
    """
    Load the YAML configuration file.

    Parameters:
    - config_path (str): Path to the YAML configuration file.

    Returns:
    - dict: Configuration settings.
    """
    try:
        with open(config_path, 'r') as file:
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
        minibatch_size_per_device (int, optional):
            The minibatch size per device. Defaults to None.
        lora_modules (list[str], optional):
            The list of LORA modules. Defaults to None.
            If None, then LORA is not used.
        custom_kwargs (dict, optional):
            Arguments to pass to HF's from_pretrained(). Defaults to None.
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
        max_ctx (int, optional):
            The maximum context length. Defaults to 512.
    """

    CHECKPOINTING_MEMORY = 3e9
    MODEL_PARALLEL_FACTOR = 2
    name: str
    memory: float
    default_lr: float
    eval_batch_size: int
    minibatch_size_per_device: int
    lora_modules: Optional[list[str]]
    custom_kwargs: dict
    gradient_checkpointing: bool
    model_parallel: bool
    default_optimizer: str
    max_ctx: int

    def __init__(
        self,
        name: str,
        memory: float,
        default_lr: float = 1e-5,
        eval_batch_size: int = 32,
        minibatch_size_per_device: Optional[int] = None,
        lora_modules: Optional[list[str]] = None,
        custom_kwargs: Optional[dict] = None,
        gradient_checkpointing: Optional[bool] = None,
        model_parallel: Optional[bool] = None,
        default_optimizer: str = "adam",
        max_ctx: int = 512,
    ):
        custom_kwargs = custom_kwargs or {}
        per_device_ram = torch.cuda.get_device_properties(0).total_memory
        n_devices = torch.cuda.device_count()
        if model_parallel is None:
            model_parallel = (
                n_devices > 1 and
                per_device_ram < self.MODEL_PARALLEL_FACTOR * memory
            )
        if gradient_checkpointing is None:
            gradient_checkpointing = memory > self.CHECKPOINTING_MEMORY
        if minibatch_size_per_device is None:
            minibatch_size_per_device = eval_batch_size
        if not torch.cuda.is_bf16_supported() and (
            custom_kwargs.get("torch_dtype") == "torch.bfloat16"
        ):
            custom_kwargs["torch_dtype"] = "torch.float32"
        if not torch.cuda.is_bf16_supported() and custom_kwargs.get("bf16"):
            custom_kwargs["bf16"] = False
            custom_kwargs["fp32"] = True
        self.name = name
        self.memory = memory
        self.default_lr = default_lr
        self.eval_batch_size = eval_batch_size
        self.minibatch_size_per_device = minibatch_size_per_device
        self.lora_modules = lora_modules
        self.custom_kwargs = custom_kwargs
        self.gradient_checkpointing = gradient_checkpointing
        self.model_parallel = model_parallel
        self.default_optimizer = default_optimizer
        self.max_ctx = max_ctx


MODELS_DICT: dict[str, ModelConfig] = {
    cfg["name"]: ModelConfig(**cfg)
    for cfg in load_config("configs/models.yaml")["models"]
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
