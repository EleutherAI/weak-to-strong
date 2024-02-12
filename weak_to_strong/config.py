import torch
from dataclasses import dataclass
from typing import Optional

from weak_to_strong.loss import logconf_loss_fn, product_loss_fn, xent_loss, kl_loss


class ModelConfig:

    def __init__(
        self, 
        name: str, 
        memory: float,
        default_lr: float = 1e-5,
        eval_batch_size: int = 32,
        minibatch_size_per_device: Optional[int] = None,
        lora_modules: Optional[list[str]] = None,
        custom_kwargs: Optional[dict] = None,
        gradient_checkpointing: bool = False,
        model_parallel: Optional[bool] = None,
        default_optimizer: str = "adam",
    ):
        self.per_device_ram = torch.cuda.get_device_properties(0).total_memory
        self.n_devices = torch.cuda.device_count()
        if model_parallel is None:
            model_parallel = (
                self.n_devices > 1 and self.per_device_ram < 2 * self.memory
            )
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


GPT_NEOX_LORA_MODULES = ["dense_h_to_4h", "dense_4h_to_h", "query_key_value"]
GPT2_LORA_MODULES = ["c_fc", "c_proj", "c_attn"]


# NOTE learning rates are not particularly tuned, work somewhat reasonably at train batch size 32
MODEL_CONFIGS = [
    ModelConfig(
        name="gpt2",
        memory=550e6,
        default_lr=5e-5,
        lora_modules=GPT2_LORA_MODULES,
    ),
    ModelConfig(
        name="gpt2-medium",
        memory=1.5e9,
        default_lr=5e-5,
        lora_modules=GPT2_LORA_MODULES,
    ),
    ModelConfig(
        name="gpt2-large",
        memory=3.25e6,
        lora_modules=GPT2_LORA_MODULES,
    ),
    ModelConfig(
        name="gpt2-xl",
        memory=6.43e9,
        eval_batch_size=2,
        gradient_checkpointing=True,
        lora_modules=GPT2_LORA_MODULES,
    ),
    ModelConfig(
        name="EleutherAI/pythia-70m",
        memory=166e6,
        minibatch_size_per_device=32,  # this needs adjusting for GPU/dataset
        lora_modules=GPT_NEOX_LORA_MODULES,
    ),
    ModelConfig(
        name="EleutherAI/pythia-14m",
        memory=53e6,
        minibatch_size_per_device=32,  # this needs adjusting for GPU/dataset
        lora_modules=GPT_NEOX_LORA_MODULES,
    ),
    ModelConfig(
        name="EleutherAI/pythia-160m-v0",
        memory=375e6,
        minibatch_size_per_device=32,  # this needs adjusting for GPU/dataset
        lora_modules=GPT_NEOX_LORA_MODULES,
    ),
    ModelConfig(
        name="EleutherAI/pythia-410m",
        memory=911e6,
        minibatch_size_per_device=32,  # this needs adjusting for GPU/dataset
        lora_modules=GPT_NEOX_LORA_MODULES,
    ),
    ModelConfig(
        name="EleutherAI/pythia-2.8b",
        memory=5.68e9,
        minibatch_size_per_device=2,  # this needs adjusting for GPU/dataset
        lora_modules=GPT_NEOX_LORA_MODULES,
    ),
    ModelConfig(
        name="EleutherAI/pythia-12b",
        memory=24e9,
        minibatch_size_per_device=2,  # this needs adjusting for GPU/dataset
        lora_modules=GPT_NEOX_LORA_MODULES,
        custom_kwargs={
            "torch_dtype": torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float32  # we can only do this because we're using LoRA
        },
    ),
    ModelConfig(
        name="mistralai/Mistral-7B-v0.1",
        memory=15e9,
        eval_batch_size=2,
        lora_modules=[
            "up_proj",
            "down_proj",
            "gate_proj",
            "k_proj",
            "q_proj",
            "v_proj",
        ],
        minibatch_size_per_device=1,  # this needs adjusting for GPU/dataset
        gradient_checkpointing=True,
        custom_kwargs={
            "torch_dtype": torch.bfloat16  # we can only do this because we're using LoRA
            if torch.cuda.is_bf16_supported()
            else torch.float32,
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-1_8B",
        memory=3.67e9,
        eval_batch_size=2,
        gradient_checkpointing=True,
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "5fde88dff770a7d036847211f5d9d9705f0caa69",
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-7B",
        memory=16e9,
        eval_batch_size=2,
        gradient_checkpointing=True,
        # note: you will probably not be able to run this without many gpus
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "d4efd21e866b9cb3466cb65b963933f5e98016d1",
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-14B",
        memory=30e9,
        eval_batch_size=2,
        gradient_checkpointing=True,
        # note: you will probably not be able to run this bf16 support and without many gpus
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "8be2854218fea9054331e217fd26a06f3fd02004",
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-72B",
        memory=164e9,
        eval_batch_size=1,
        gradient_checkpointing=True,
        # note: you will probably not be able to run this without bf16 support and many gpus
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "fec78c0e3b3b10dd9f0ce775c34a686a3255a7d1",
        },
        # This model is really big, save space by using adafactor.
        # Note that even then it will take up ~60GB per GPU on an 8-GPU machine.
        default_optimizer="adafactor",
    ),
]
MODELS_DICT: dict[str, ModelConfig] = {
    model_config.name: model_config for model_config in MODEL_CONFIGS
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
