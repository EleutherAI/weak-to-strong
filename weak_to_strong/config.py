import torch
from dataclasses import dataclass
from typing import Optional

from weak_to_strong.loss import logconf_loss_fn, product_loss_fn, xent_loss, kl_loss
from weak_to_strong.model import TransformerWithHead

MODEL_TYPE = TransformerWithHead | torch.nn.DataParallel[TransformerWithHead]


@dataclass
class ModelConfig:
    """Configuration for a model used in LORA finetuning.
    Arguments:
        name: str
            The name of the model, e.g. "gpt2", "gpt2-medium", etc.
        default_lr: float
            The default learning rate for the model.
        eval_batch_size: int
            The batch size to use for evaluation.
        minibatch_size_per_device: Optional[int]
            The minibatch size per device to use for training. 
            If None, the default is 1.
        lora_modules: Optional[list[str]]
            The LoRA modules to use for the model. 
            If None, the default is None.
        custom_kwargs: Optional[dict]
            Custom keyword arguments to pass to the model. 
            If None, the default is None.
        gradient_checkpointing: bool
            Whether to use gradient checkpointing. The default is False.
        model_parallel: bool
            Whether to use model parallel. The default is False.
        default_optimizer: str
            The default optimizer to use for the model. The default is "adam".
    """
    name: str
    default_lr: float
    eval_batch_size: int
    minibatch_size_per_device: Optional[int] = None
    lora_modules: Optional[list[str]] = None
    custom_kwargs: Optional[dict] = None
    gradient_checkpointing: bool = False
    model_parallel: bool = False
    default_optimizer: str = "adam"

    def load_model(
        self,
        batch_size: int,
        use_lm_head: bool,
        minibatch_size_per_device: Optional[int] = None,
        num_labels: int = 2,
        linear_probe: bool = False,
    ) -> tuple[MODEL_TYPE, int]:
        custom_kwargs = self.custom_kwargs or {}
        if minibatch_size_per_device is None:
            minibatch_size_per_device = self.minibatch_size_per_device or 1
        if self.model_parallel:
            assert (
                torch.cuda.device_count() > 1
            ), f"you might want more gpus for {self.name}"
            model = TransformerWithHead.from_pretrained(
                self.name,
                lora_modules=self.lora_modules,
                use_lm_head=use_lm_head,
                num_labels=num_labels,
                device_map="auto",
                linear_probe=linear_probe,
                **custom_kwargs,
            )
            # slight misnomer, more like minibatch_size_per_dp_replica
            minibatch_size = minibatch_size_per_device
        else:
            model = TransformerWithHead.from_pretrained(
                self.name,
                lora_modules=self.lora_modules,
                use_lm_head=use_lm_head,
                num_labels=num_labels,
                linear_probe=linear_probe,
                **custom_kwargs,
            ).to(
                "cuda"  # type: ignore
            )
            # data parallel:  currently not supported with model parallel
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model, output_device=0)
                minibatch_size = min(
                    minibatch_size_per_device * torch.cuda.device_count(), 
                    batch_size
                )
                print(
                    "Using",
                    torch.cuda.device_count(),
                    "GPUs, setting minibatch_size to",
                    minibatch_size,
                )
            else:
                minibatch_size = minibatch_size_per_device
        return model, minibatch_size


GPT_NEOX_LORA_MODULES = ["dense_h_to_4h", "dense_4h_to_h", "query_key_value"]
GPT2_LORA_MODULES = ["c_fc", "c_proj", "c_attn"]
per_device_ram = torch.cuda.get_device_properties(0).total_memory

# NOTE learning rates are not particularly tuned, work somewhat reasonably at train batch size 32
MODEL_CONFIGS = [
    ModelConfig(
        name="gpt2",
        default_lr=5e-5,
        eval_batch_size=32,
        lora_modules=GPT2_LORA_MODULES,
    ),
    ModelConfig(
        name="gpt2-medium",
        default_lr=5e-5,
        eval_batch_size=32,
        lora_modules=GPT2_LORA_MODULES,
    ),
    ModelConfig(
        name="gpt2-large",
        default_lr=1e-5,
        eval_batch_size=32,
        lora_modules=GPT2_LORA_MODULES,
    ),
    ModelConfig(
        name="gpt2-xl",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        lora_modules=GPT2_LORA_MODULES,
        # Should use model_parallel on V100s (note: ironically if you have a single V100
        # it should run, but if you have multiple it won't run without model_parallel
        # because of the overhead of data parallel training).
        model_parallel=(per_device_ram < 35e9 and torch.cuda.device_count() > 1),
    ),
    ModelConfig(
        name="EleutherAI/pythia-70m",
        default_lr=1e-5,
        eval_batch_size=32,
        minibatch_size_per_device=32,  # this needs adjusting for GPU/dataset
        model_parallel=False,
        lora_modules=GPT_NEOX_LORA_MODULES,
    ),
    ModelConfig(
        name="EleutherAI/pythia-14m",
        default_lr=1e-5,
        eval_batch_size=32,
        minibatch_size_per_device=32,  # this needs adjusting for GPU/dataset
        model_parallel=False,
        lora_modules=GPT_NEOX_LORA_MODULES,
    ),
    ModelConfig(
        name="EleutherAI/pythia-160m-v0",
        default_lr=1e-5,
        eval_batch_size=32,
        minibatch_size_per_device=32,  # this needs adjusting for GPU/dataset
        model_parallel=False,
        lora_modules=GPT_NEOX_LORA_MODULES,
    ),
    ModelConfig(
        name="EleutherAI/pythia-410m",
        default_lr=1e-5,
        eval_batch_size=32,
        minibatch_size_per_device=32,  # this needs adjusting for GPU/dataset
        model_parallel=False,
        lora_modules=GPT_NEOX_LORA_MODULES,
    ),
    ModelConfig(
        name="EleutherAI/pythia-2.8b",
        default_lr=1e-5,
        eval_batch_size=32,
        minibatch_size_per_device=2,  # this needs adjusting for GPU/dataset
        model_parallel=False,
        lora_modules=GPT_NEOX_LORA_MODULES,
        custom_kwargs={
            "torch_dtype": torch.bfloat16  # we can only do this because we're using LoRA
            if torch.cuda.is_bf16_supported()
            else torch.float32,
        },
    ),
    ModelConfig(
        name="EleutherAI/pythia-12b",
        default_lr=1e-5,
        eval_batch_size=32,
        minibatch_size_per_device=2,  # this needs adjusting for GPU/dataset
        model_parallel=False,
        lora_modules=GPT_NEOX_LORA_MODULES,
        custom_kwargs={
            "torch_dtype": torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float32  # we can only do this because we're using LoRA
        },
    ),
    ModelConfig(
        name="mistralai/Mistral-7B-v0.1",
        default_lr=1e-5,
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
        model_parallel=False,
        custom_kwargs={
            "torch_dtype": torch.bfloat16  # we can only do this because we're using LoRA
            if torch.cuda.is_bf16_supported()
            else torch.float32,
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-1_8B",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=(per_device_ram < 35e9 and torch.cuda.device_count() > 1),
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "5fde88dff770a7d036847211f5d9d9705f0caa69",
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-7B",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=True,
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
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=True,
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
        default_lr=1e-5,
        eval_batch_size=1,
        gradient_checkpointing=True,
        model_parallel=True,
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
