import warnings
from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from peft import (
    get_peft_model,  # type: ignore
    LoraConfig,  # type: ignore
    TaskType,  # type: ignore
    PeftType,  # type: ignore
    AutoPeftModelForCausalLM,  # type: ignore
)
from peft.tuners.lora.layer import LoraLayer
from typing import Optional


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
        try:
            config = AutoConfig.from_pretrained(name, **kwargs)
            lm = AutoModelForCausalLM.from_pretrained(
                name, attn_implementation="eager", **kwargs
            )
        except OSError:
            print(
                f"Could not find config for {name} on the hub. Assuming this is a PEFT model"
            )
            lm = AutoPeftModelForCausalLM.from_pretrained(
                name, attn_implementation="eager", **kwargs
            )
            lm = lm.merge_and_unload()
        config = lm.config
        super().__init__(config)
        self.lm = lm
        self.name = name
        self.num_labels = config.num_labels
        self.use_lm_head = use_lm_head
        self.lora_modules = lora_modules

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

            # cast LoRA parameters to float32
            for p in self.lm.parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)

        lm_head = getattr(self.lm, "lm_head", getattr(self.lm, "embed_out", None))
        assert isinstance(lm_head, torch.nn.Linear)
        if use_lm_head:
            print("Using LM head instead of learned head because choices are provided")
            self.score = None
        else:
            hidden_size = getattr(
                config,
                "word_embed_proj_dim",
                getattr(config, "n_embd", getattr(config, "hidden_size", None)),
            )
            assert isinstance(hidden_size, int)
            self.score = torch.nn.Linear(hidden_size, self.num_labels, bias=False).to(
                lm_head.weight.dtype
            )
            torch.nn.init.normal_(self.score.weight, std=0.01 / hidden_size**0.5)
            # remove the LM head so it isn't in model.parameters()
            if hasattr(self.lm, "lm_head"):
                del (
                    self.lm.lm_head
                )  # TODO: this doesn't work with LoRA for some reason because of attribute hiding
            elif hasattr(self.lm, "embed_out"):
                del self.lm.embed_out
            else:
                warnings.warn("Tried to remove LM head but it wasn't found.")
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

    @property
    def lora_modules_to_save(self):
        save_modules: list = [m for m in self.lm.modules() if isinstance(m, LoraLayer)]
        if self.score is not None:
            save_modules.append(self.score)
        return save_modules

    def save_state_dict(self, path):
        if self.lora_modules is None:
            save_dict_or_list = self.state_dict()
        else:
            # only save lora parameters
            save_dict_or_list = [m.state_dict() for m in self.lora_modules_to_save]
        torch.save(save_dict_or_list, path)

    def load_state_dict(self, state_dict, strict=True, assign=True):
        if self.lora_modules is None:
            return super().load_state_dict(state_dict, strict, assign)
        else:
            assert isinstance(state_dict, list)
            modules_to_save = self.lora_modules_to_save
            assert len(state_dict) == len(modules_to_save)
            for m, sd in zip(modules_to_save, state_dict):
                m.load_state_dict(sd, strict, assign)

    def gradient_checkpointing_enable(self):
        model = self.transformer if self.score is not None else self.lm
        (
            model if hasattr(model, "save_pretrained") else model.module
        ).gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: torch.LongTensor,
        choice_input_ids: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
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
            outputs = self.lm(input_ids, output_hidden_states=True)
            all_logits = outputs.logits
            all_hidden_states = outputs.hidden_states
            logits_at_last = [
                all_logits[i, input_lens[i] - 1, choice_input_ids[i]]
                for i in range(len(input_lens))
            ]  # [batch_size, num_choices]
            logits = torch.stack(logits_at_last)
        else:  # use learned head
            assert choice_input_ids is None
            transformer_outputs = self.transformer(input_ids, output_hidden_states=True)
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
            all_hidden_states = transformer_outputs.hidden_states

        if output_hidden_states:
            return logits, all_hidden_states
        return logits
