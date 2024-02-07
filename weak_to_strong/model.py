from dataclasses import dataclass
from functools import partial
import gc
import json
import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_utils import (
    WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    is_safetensors_available,
    safe_load_file,
    logger,
    is_torch_greater_or_equal_than_1_13,
)
from peft import get_peft_model, LoraConfig, TaskType, PeftType  # type: ignore
from typing import Optional


def update_sharded_checkpoint(
    model, folder, update_coef: float = 1.0, strict=True, prefer_safe=True
):
    """
    This is the same as transformers.modeling_utils.update_sharded_checkpoint,
    but with  the new update_coef parameter, allowing incremental updates to
    the model weights.
    See START OF CHANGE below.
    """
    # Load the index
    index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
    safe_index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)

    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)

    if not index_present and not (
        safe_index_present and is_safetensors_available()
    ):
        filenames = (
            (WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME)
            if is_safetensors_available() else (WEIGHTS_INDEX_NAME,)
        )
        raise ValueError(
            f"Can't find a checkpoint index ({' or '.join(filenames)})"
            f" in {folder}."
        )

    load_safe = False
    if safe_index_present:
        if prefer_safe:
            if is_safetensors_available():
                load_safe = True  # load safe due to preference
            else:
                logger.warning(
                    f"Cannot load sharded checkpoint at {folder} safely since "
                    "safetensors is not installed!"
                )
        elif not index_present:
            load_safe = True  # load safe since we have no other choice

    load_index = safe_index_file if load_safe else index_file

    with open(load_index, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))

    # If strict=True, error before loading any of the state dicts.
    loaded_keys = index["weight_map"].keys()
    model_keys = model.state_dict().keys()
    missing_keys = [key for key in model_keys if key not in loaded_keys]
    unexpected_keys = [key for key in loaded_keys if key not in model_keys]
    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        error_message = (
            f"Error(s) in loading state_dict for {model.__class__.__name__}"
        )
        if len(missing_keys) > 0:
            str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
            error_message += f"\nMissing key(s): {str_missing_keys}."
        if len(unexpected_keys) > 0:
            str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
            error_message += f"\nMissing key(s): {str_unexpected_keys}."
        raise RuntimeError(error_message)

    if is_torch_greater_or_equal_than_1_13:
        weights_only_kwarg = {"weights_only": True}
    else:
        weights_only_kwarg = {}
    loader = safe_load_file if load_safe else partial(
        torch.load, map_location="cpu", **weights_only_kwarg
    )

    for shard_file in shard_files:
        state_dict = loader(os.path.join(folder, shard_file))
        
        # START OF CHANGE
        # Update each weight using the given formula
        for name, param in model.named_parameters():
            if name in state_dict:
                update = update_coef * (state_dict[name] - param.data)
                param.data.add_(update)
        # END OF CHANGE

        # Make sure memory is freed before we load the next state dict.
        del state_dict
        gc.collect()

    # Return the same thing as PyTorch load_state_dict function.
    return torch.nn.modules.module._IncompatibleKeys(
        missing_keys, unexpected_keys
    )


@dataclass
class HeadOutput:
    logits: torch.FloatTensor


class TransformerWithHead(PreTrainedModel):
    """
    This class initializes the linear head to zeros
    """

    def __init__(
        self,
        name,
        lora_modules=None,
        use_lm_head=False,
        linear_probe=False,
        lora_rank=8,
        lora_alpha=8,
        lora_dropout=0.0,
        **kwargs,
    ):
        config = AutoConfig.from_pretrained(name, **kwargs)
        super().__init__(config)
        self.num_labels = config.num_labels
        self.use_lm_head = use_lm_head
        self.lora_modules = lora_modules
        self.lm = AutoModelForCausalLM.from_pretrained(name, **kwargs)

        if lora_modules is not None:
            print(f"Using LoraModel on modules {lora_modules}")
            peft_config = LoraConfig(
                peft_type=PeftType.LORA,
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                target_modules=lora_modules,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            self.lm = get_peft_model(self.lm, peft_config)

        lm_head = getattr(self.lm, "lm_head", getattr(self.lm, "embed_out", None))
        assert isinstance(lm_head, torch.nn.Linear)
        if use_lm_head:
            print("Using LM head instead of learned head because choices are provided")
            self.score = None
        else:
            hidden_size = getattr(
                config, "n_embd", getattr(config, "hidden_size", None)
            )
            assert isinstance(hidden_size, int)
            self.score = torch.nn.Linear(hidden_size, self.num_labels, bias=False).to(
                lm_head.weight.dtype
            )
            torch.nn.init.normal_(self.score.weight, std=0.0)
        self.linear_probe = linear_probe

    @property
    def transformer(self):
        if self.lora_modules is not None:
            return (
                self.lm.base_model.base_model
            )  # PeftModel -> LoraModel -> PreTrainedModel
        return self.lm.base_model  # CausalLM -> PreTrainedModel

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        return cls(name, **kwargs)
    
    def save_torch(self, path, optimizer=None, scheduler=None):
        save_dict = self.state_dict()
        if optimizer is not None:
            save_dict["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            save_dict["scheduler"] = scheduler.state_dict()
        torch.save(save_dict, path)

    def gradient_checkpointing_enable(self):
        model = self.transformer if self.score is not None else self.lm
        (
            model if hasattr(model, "save_pretrained") else model.module
        ).gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: torch.LongTensor,
        choice_input_ids: Optional[torch.LongTensor] = None,
    ):
        """
        Forward pass of the model with a linear head.

        Parameters:
        input_ids (torch.LongTensor): Input tensor containing the token ids.

        Returns:
        HeadOutput: Output dataclass containing the logits.
        """
        input_lens = (input_ids != 0).sum(dim=-1)

        if self.score is None:  # use LM head
            assert choice_input_ids is not None
            all_logits = self.lm(input_ids).logits
            logits_at_last = [
                all_logits[i, input_lens[i] - 1, choice_input_ids[i]]
                for i in range(len(input_lens))
            ]  # [batch_size, num_choices]
            logits = torch.stack(logits_at_last)
        else:  # use learned head
            transformer_outputs = self.transformer(input_ids)
            hidden_states = torch.stack(
                [
                    transformer_outputs[0][i, input_lens[i] - 1, :]
                    for i in range(len(input_lens))
                ]
            )
            self.score.to(hidden_states.device)
            if self.linear_probe:
                hidden_states = hidden_states.detach()
            logits = self.score(hidden_states)

        return logits
