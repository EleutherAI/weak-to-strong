import os
from datasets import load_from_disk
import numpy as np

import json
from tqdm.auto import tqdm

import torch
from collections import defaultdict

from weak_to_strong.config import ModelConfig, MODELS_DICT, LOSS_DICT
from weak_to_strong.model import TransformerWithHead
from weak_to_strong.common import to_batch
from weak_to_strong.train import maybe_load_model


def get_grads(
    strong_path="../results/grads/sciq/bs=32-dl=1-dn=sciq-gib=1-ge=6-gee=10000000-lp=0-lbmae=0-l=xent-l=1e-07-ls=cosi_anne-mc=512-mfbm=auro_agai_supe-ms=Mistral-7B-v0.1-ntd=2000-ntd=3000-ntd=6000-o=adam-stl=50-s=0-twd=0",
    w2s_path="../results/grads/sciq/bs=32-dl=1-dn=sciq-gib=1-lp=0-lbmae=0-l=kl-l=5e-08-ls=cosi_anne-mc=512-mfbm=auro_agai_supe-ms=Mistral-7B-v0.1-ntd=2000-ntd=3000-ntd=6000-o=adam-stl=50-s=0-twd=0-we=3-wee=25-wlf=0.5-wms=Qwen1.5-0.5B",
    d_proj=10_000,
    device="cuda:6",
    grads_device="cuda:7",
    n_eval=100,
):
    strong_eval_results_path = os.path.join(strong_path, "eval_results_final")
    final_eval_results_path = os.path.join(w2s_path, "eval_results_final")
    final_eval_results = load_from_disk(final_eval_results_path)
    strong_eval_results = load_from_disk(strong_eval_results_path)
    print(len(final_eval_results), len(strong_eval_results))
    strong_eval_results = strong_eval_results.select(range(len(final_eval_results)))
    assert np.all(
        np.array(final_eval_results["id"]) == np.array(strong_eval_results["id"])
    )
    final_eval_results = final_eval_results.add_column(
        "strong_soft_pred", strong_eval_results["soft_pred"]
    )
    final_eval_results = final_eval_results.add_column(
        "strong_hard_pred", strong_eval_results["hard_pred"]
    )
    final_eval_results = final_eval_results.with_format("torch")

    print(
        "weak_hard_label acc",
        (
            (final_eval_results["weak_soft_label"][:, 1] > 0.5)
            == final_eval_results["hard_label"]
        )
        .float()
        .mean(),
    )
    print(
        "strong_hard_pred acc",
        (final_eval_results["strong_hard_pred"] == final_eval_results["hard_label"])
        .float()
        .mean(),
    )
    print(
        "w2s pred acc",
        (final_eval_results["hard_pred"] == final_eval_results["hard_label"])
        .float()
        .mean(),
    )

    config = json.load(open(os.path.join(w2s_path, "config.json"), "r"))
    model_name = config["model_size"]
    use_lm_head = "choice_input_ids" in final_eval_results.column_names
    loss_fn = LOSS_DICT[config["loss"]]

    proj_grads, hiddens = defaultdict(lambda: defaultdict(dict)), defaultdict(
        lambda: defaultdict(dict)
    )
    proj_grads_path = os.path.join(w2s_path, "proj_grads.pt")
    hiddens_path = os.path.join(w2s_path, "hiddens.pt")

    if os.path.exists(proj_grads_path):
        proj_grads = torch.load(proj_grads_path)
        hiddens = torch.load(hiddens_path)
        print("Loaded existing proj_grads and hiddens")
    else:
        for run_type, results_path in [
            ("w2s", w2s_path),
            ("strong", strong_path),
        ]:
            # init model
            mcfg = MODELS_DICT[model_name].copy()
            if config["disable_lora"]:
                mcfg["lora_modules"] = None
            model_config = ModelConfig(**mcfg)
            model = TransformerWithHead.from_pretrained(  # type: ignore
                model_config.name,
                lora_modules=model_config.lora_modules,
                use_lm_head=use_lm_head,
                num_labels=2,
                linear_probe=config["linear_probe"],
                **model_config.custom_kwargs,
            )
            # # TODO: delete model.lm.lm_head from learned head models before saving state dict
            model_n_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            torch.manual_seed(
                0
            )  # ensure that our projection is the same across runs and examples
            proj_basis_indices = torch.randint(0, model_n_params, (d_proj,))
            proj_basis_indices, _ = proj_basis_indices.sort()

            print(f"Hash(projection indices): {proj_basis_indices.sum().item()}")

            for target_label_column in ["weak_soft_label", "soft_label"]:
                print(f"Computing gradients for {target_label_column} in {run_type}")
                for checkpoint in os.listdir(results_path):
                    if (
                        not checkpoint.startswith("checkpoint")
                        or run_type in proj_grads
                        and checkpoint in proj_grads[run_type]
                        and target_label_column in proj_grads[run_type][checkpoint]
                    ):
                        continue
                    print(f"Loading model from {checkpoint}")
                    proj_grads[run_type][checkpoint][target_label_column] = -torch.ones(
                        (n_eval, d_proj), device=grads_device
                    )
                    hiddens[run_type][checkpoint][target_label_column] = -torch.ones(
                        (n_eval, model.config.hidden_size), device="cpu"
                    )

                    # load model checkpoint
                    assert maybe_load_model(
                        model, os.path.join(results_path, checkpoint)
                    )
                    model.eval().to(
                        torch.bfloat16
                        if torch.cuda.is_bf16_supported()
                        else torch.float32
                    ).to(device)

                    # unfortunately we have to use a batch size of 1 to get examplewise grads
                    # because only 1 backwards pass is allowed per forward pass
                    for i, batch in tqdm(
                        enumerate(
                            to_batch(
                                final_eval_results.select(range(n_eval)), batch_size=1
                            )
                        )
                    ):
                        input_ids = torch.nn.utils.rnn.pad_sequence(
                            [ex for ex in batch["input_ids"]], batch_first=True
                        ).to(model.device)
                        label = batch[target_label_column].to(model.device)
                        choice_ids = batch.get("choice_input_ids")
                        logits, hs = model(
                            input_ids,
                            choice_input_ids=choice_ids.to(model.device)
                            if choice_ids is not None
                            else None,
                            output_hidden_states=True,
                        )

                        loss = loss_fn(logits, label, step_frac=0)
                        loss.backward()

                        # this mess avoids concatenating all the grads
                        # into one tensor before projecting, to save memory
                        grad_iter = iter(
                            p.grad for p in model.parameters() if p.grad is not None
                        )
                        pg = next(grad_iter)
                        start_i = 0  # index into grad of the first component of pg
                        for proj_i, grad_idxr in enumerate(
                            proj_basis_indices
                        ):  # iterate over sorted projection indices
                            while (
                                start_i + pg.numel() <= grad_idxr
                            ):  # while the current param is earlier than the desired index
                                start_i += pg.numel()
                                pg = next(grad_iter)
                            proj_grads[run_type][checkpoint][target_label_column][
                                i, proj_i
                            ] = pg.flatten()[grad_idxr - start_i]

                        hiddens[run_type][checkpoint][target_label_column][i, :] = (
                            hs[-1][0, -1, :].detach().clone().cpu()
                        )

                        # zero out grads
                        model.zero_grad()

        def to_dict(defdict):
            return {
                k: to_dict(v) if isinstance(v, defaultdict) else v
                for k, v in defdict.items()
            }

        torch.save(to_dict(proj_grads), proj_grads_path)
        torch.save(to_dict(hiddens), hiddens_path)
