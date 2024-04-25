import os
import pickle
import time
from typing import Optional

import datasets
import numpy as np
import torch
import torch_optimizer as toptim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

import weak_to_strong.logger as logger
from weak_to_strong.common import to_batch, get_gpu_mem_used
from weak_to_strong.eval import eval_loop, compute_metrics
from weak_to_strong.common import assert_type
from weak_to_strong.model import TransformerWithHead
from weak_to_strong.train_config import TrainConfig
from weak_to_strong import grads


def save(
    model: torch.nn.Module,
    save_file: Optional[str],
):
    # Note: If the model is wrapped by DataParallel, we need to unwrap it before saving
    model_to_save = model.module if hasattr(model, "module") else model

    model_to_save.save_state_dict(save_file)
    print("saved torch weights", save_file)


def train_model(
    cfg: TrainConfig,
    model: torch.nn.Module,
    ds: datasets.Dataset,
    final_eval_ds: Optional[datasets.Dataset] = None,
    max_val_size: int = 500,
    val_frac: float = 0.2,
):
    """
    ds is a dataset of examples, each of which is a dict with keys:
    - input_ids: a list of token ids
    - soft_label: a list of soft label probabilities
    - choice_input_ids (optional): a pair of token ids for the answer choices,
        indicating to use the LM head of the model
    """
    is_w2s = cfg.is_w2s
    minibatch_size = assert_type(int, cfg.minibatch_size_per_replica)
    model_n_params = sum(p.numel() for p in model.parameters())
    model_or_module = (
        model if hasattr(model, "gradient_checkpointing_enable") else model.module
    )
    if cfg.d_downsample == "sqrt":
        # e.g. 7B -> 16.7m, 410m -> 4m
        d_downsample = int(200 * model_n_params**0.5)
    else:
        d_downsample = assert_type(int, cfg.d_downsample)

    if cfg.load_best_model_at_end:
        metric_for_best_model = "eval/auroc_against_weak" if is_w2s else "eval/auroc"
    if cfg.store_grads:
        minibatch_size = 1
        print("Setting minibatch_size to 1 for w2s training to with store_grads=True")
        assert final_eval_ds is not None, "must provide eval_ds if store_grads"

    assert (
        cfg.batch_size % minibatch_size == 0
    ), "batch size must be divisible by minibatch size"

    def checkpoint_name(step):
        return os.path.join(cfg.save_path, f"checkpoint_{step}.bin")

    if cfg.load_best_model_at_end:
        # split off a fraction of the train ds for evaluation
        # when we're selecting using it
        n_eval = min(max_val_size, int(len(ds) * val_frac))
        print(f"Taking {n_eval} examples from train set for selecting best model")
        ddict = datasets.Dataset.train_test_split(ds, test_size=n_eval)
        ds, val_ds = ddict["train"], ddict["test"]
        if is_w2s:
            val_ds = val_ds.rename_columns(
                {
                    "soft_label": "weak_soft_label",
                    "hard_label": "weak_hard_label",
                    "gt_soft_label": "soft_label",
                    "gt_hard_label": "hard_label",
                }
            )
    else:
        val_ds = final_eval_ds

    print(f"LR: {cfg.lr}, BS: {cfg.batch_size}, MBS: {minibatch_size}, n: {len(ds)}")

    ### Prepare model, optimizer, and scheduler ###
    # we purposefully turn off dropout, for determinism
    # this seems to help for 1 epoch finetuning anyways
    model.train(mode=cfg.train_with_dropout)
    if cfg.model_config.gradient_checkpointing:
        model_or_module.gradient_checkpointing_enable()

    nsteps = len(ds) * cfg.epochs // cfg.batch_size

    def lr_schedule_fn(step):
        if cfg.lr_schedule == "constant":
            return 1
        else:
            assert False, f"invalid lr schedule, {cfg.lr_schedule}"

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert cfg.optim is not None and cfg.lr is not None
    if cfg.optim.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=cfg.lr, betas=(0.9, 0.95))
    elif cfg.optim.lower() == "adafactor":
        optimizer = toptim.Adafactor(trainable_params, lr=cfg.lr)
    elif cfg.optim.lower() == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=cfg.lr)
    else:
        assert False, f"invalid optimizer {cfg.optim}, must be adam or adafactor"
    if cfg.lr_schedule == "cosine_anneal":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, nsteps)
    elif cfg.lr_schedule == "linear_with_warmup":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=50, num_training_steps=nsteps
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule_fn)

    ds.save_to_disk(os.path.join(cfg.save_path, "train_ds"))

    step = 0
    ids = []
    logodds = []
    losses = []
    accuracies = []
    aurocs = []
    best_eval = float("-inf")
    best_step = 0 if cfg.load_best_model_at_end else None
    ckpt_names = []
    per_step_expected_effects = []
    initial_eval_outputs = None
    hiddens = []

    def delete_old_checkpoints():
        if cfg.save_total_limit is None:
            return
        num_to_delete = len(ckpt_names) - cfg.save_total_limit
        # delete the oldest checkpoints that aren't the best or the most recent
        to_delete = [
            name
            for name in ckpt_names[:-1]
            if name != checkpoint_name(best_step) and name
        ][:num_to_delete]
        for name in to_delete:
            ckpt_names.remove(name)
            os.remove(name)

    def update_best():
        nonlocal best_eval, best_step
        if cfg.load_best_model_at_end:
            current_eval = eval_metrics[metric_for_best_model]
            if current_eval > best_eval:
                assert os.path.exists(checkpoint_name(step)), (
                    "No checkpoint found "
                    "for the current step, "
                    "but load_best_model_at_end was set to True and the current step is "
                    "best. Please set eval_every to a multiple of save_every, and less "
                    "than the total number of steps."
                )
                best_eval = current_eval
                best_step = step
                print(f"New best model found at step {step}")

    # If the model is wrapped by DataParallel, it doesn't have a device. In this case,
    # we use GPU 0 as the output device. This sadly means that this device will store
    # a bit more data than other ones, but hopefully should not be too big of a deal.
    io_device = model.device if hasattr(model, "device") else 0
    grads_device = "cpu"

    for epoch in range(cfg.epochs):
        # iterate over batches, skipping the last one if it's too small
        for start in range(0, len(ds) - (len(ds) % cfg.batch_size), cfg.batch_size):
            loss_tot = 0

            # compute behaviorally relevant directions in parameter space
            if cfg.store_grads:
                assert final_eval_ds is not None
                (
                    downsampled_eval_jacobians,
                    eval_outputs,
                    proj_basis_indices,
                ) = grads.get_jacobians(
                    model=model,
                    dataset=final_eval_ds,
                    postprocess_logits_fn=grads.Diff(),
                    target_label_column="soft_label",  # is not used
                    d_down=d_downsample,
                    step_frac=step / nsteps,
                    io_device=grads_device,
                )
                if initial_eval_outputs is None:
                    initial_eval_outputs = eval_outputs
                # note that these jacobians have only 1 (squeezed) column
                # so the overall shape is (n_eval, d_proj)

            # save
            if (
                cfg.save_every
                and step % cfg.save_every == 0
                and cfg.save_every < nsteps
            ):
                ckpt_names.append(checkpoint_name(step))
                save(model, ckpt_names[-1])
                delete_old_checkpoints()

            # eval
            if (
                cfg.eval_every
                and step % cfg.eval_every == 0
                and cfg.eval_every < nsteps
            ):
                assert val_ds is not None
                eval_results, eval_metrics = eval_loop(
                    model,
                    val_ds,
                    cfg.eval_batch_size,
                    metric_prefix="eval",
                    remove_large_columns=True,
                )
                logger.logkvs(eval_metrics)

                eval_results.save_to_disk(
                    os.path.join(cfg.save_path, f"eval_results_{step}")
                )
                update_best()

            if cfg.model_config.gradient_checkpointing:
                model_or_module.gradient_checkpointing_enable()
            model.train(mode=cfg.train_with_dropout)

            # train step
            all_logits = []
            all_labels = []
            if cfg.store_grads:
                downsampled_cumul_grads = torch.full(
                    (cfg.batch_size, d_downsample),
                    fill_value=torch.nan,
                    device=grads_device,
                )
            for j, mbatch in tqdm(
                enumerate(
                    to_batch(
                        ds, minibatch_size, start=start, end=start + cfg.batch_size
                    )
                ),
                disable=not cfg.store_grads,
                total=cfg.batch_size // minibatch_size,
            ):
                input_ids = (
                    torch.nn.utils.rnn.pad_sequence(
                        [torch.tensor(ids) for ids in mbatch["input_ids"]]  # type: ignore
                    )
                    .transpose(0, 1)
                    .to(io_device)  # type: ignore
                )
                labels = torch.tensor(mbatch["soft_label"]).to(io_device)  # type: ignore
                logits, hidden_states = model(
                    input_ids=input_ids,
                    choice_input_ids=mbatch.get("choice_input_ids"),
                    output_hidden_states=True,
                )
                logits = logits.to(io_device)
                loss = cfg.loss_fn(logits, labels, step_frac=step / nsteps) * (
                    minibatch_size / cfg.batch_size
                )
                loss_tot += loss.item()
                # we don't need to use a gradscaler because we're using bf16 instead of fp16
                loss.backward()

                logodds.extend(logits.diff(dim=1).detach().cpu().numpy().flatten())
                ids.extend(mbatch["id"])

                if cfg.store_grads:
                    assert minibatch_size == 1

                    downsampled_cumul_grads[j, :] = grads.gather_grad_components(
                        model,
                        proj_basis_indices,
                        io_device=grads_device,
                        optimizer=optimizer,
                    )
                if cfg.store_hiddens:
                    h = torch.stack(
                        hidden_states
                    ).cpu()  # [n_layers, batch_size, seq_len, hidden_size]
                    # grab the last token position at all layers
                    seq_lens = (input_ids != 0).sum(dim=-1).cpu()
                    h = torch.stack(
                        [h[:, i, seq_lens[i] - 1, :] for i in range(len(seq_lens))],
                        dim=1,
                    )
                    h = (
                        h.detach().bfloat16().transpose(0, 1)
                    )  # [batch_size, n_layers, hidden_size]
                    hiddens.append(h)

                all_logits.extend(logits)
                all_labels.extend(labels)

            # gradients accumulate, so we need to take the difference at the end
            if cfg.store_grads:
                assert (downsampled_cumul_grads == torch.nan).float().sum() == 0  # type: ignore
                downsampled_grads = downsampled_cumul_grads.diff(
                    dim=0, prepend=downsampled_cumul_grads.new_zeros(1, d_downsample)
                )

                # compute expected effect on eval outputs
                updates = -optimizer.param_groups[0]["lr"] * downsampled_grads
                jvps = updates @ downsampled_eval_jacobians.mT  # [batch_size, n_eval]

                # the computed JVP only includes d_down of the model_n_params terms,
                # so we expect the actual JVP to be `rescale` times larger
                rescale = model_n_params / d_downsample  # type: ignore
                expected_effects = rescale * jvps

                tot_expected_effect = expected_effects.sum(0)
                per_step_expected_effects.append(tot_expected_effect)

                approx_new_outputs = tot_expected_effect + eval_outputs
                torch.save(
                    {
                        "ids": ds["id"][start : start + cfg.batch_size],
                        "expected_effects": expected_effects,
                        "proj_basis_indices": proj_basis_indices,
                        "step": step,
                        "lr": optimizer.param_groups[0]["lr"],
                        "approx_new_outputs": approx_new_outputs,
                        "eval_outputs": eval_outputs,
                    },
                    os.path.join(cfg.save_path, f"gradients_{step}.pt"),
                )
                del (
                    downsampled_cumul_grads,
                    downsampled_grads,
                    updates,
                    downsampled_eval_jacobians,
                )

            if len(all_logits) == 0:
                # skip batches too small to form a single minibatch
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            # train metrics
            all_logits = torch.stack(all_logits)
            all_labels = torch.stack(all_labels)
            pred_probs = np.array(
                torch.nn.functional.softmax(all_logits.detach().float().cpu(), dim=1)
            )[:, 1]
            supervision_soft_labels = np.array(all_labels.cpu())[:, 1]

            if is_w2s:
                # then supervision labels are weak
                gt_soft_labels = np.array(
                    ds[start : start + len(pred_probs)]["gt_soft_label"]
                )[:, 1]
                weak_soft_labels = supervision_soft_labels
            else:
                gt_soft_labels = supervision_soft_labels
                weak_soft_labels = None

            train_metrics = compute_metrics(
                gt_soft_labels=gt_soft_labels,
                pred_probs=pred_probs,
                weak_soft_labels=weak_soft_labels,
                metric_prefix="train",
            )

            # these three are printed every print_every steps
            losses.append(loss_tot)
            accuracies.append(
                train_metrics["train/acc_against_weak" if is_w2s else "train/acc"]
            )
            aurocs.append(
                train_metrics["train/auroc_against_weak" if is_w2s else "train/auroc"]
            )

            train_metrics.update(
                {
                    "step": step,
                    "progress": step / nsteps,
                    "loss": loss_tot,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
            )
            logger.logkvs(train_metrics)

            if cfg.print_every and step % cfg.print_every == 0:
                print(
                    f"Step: {step}/{nsteps}; loss: {np.mean(losses)}; "
                    f"train acc: {np.mean(accuracies)}; "
                    f"train auroc: {np.mean(aurocs)}; ({len(losses)} losses)"
                )
                losses = []
                accuracies = []
                aurocs = []

            step += 1
            logger.dumpkvs()

    # save final checkpoint
    if cfg.save_every and checkpoint_name(step) not in ckpt_names:
        ckpt_names.append(checkpoint_name(step))
        save(model, ckpt_names[-1])
        delete_old_checkpoints()

    # save logodds
    logodds = {
        "ids": ids,
        "logodds": torch.tensor(logodds),
    }
    torch.save(logodds, os.path.join(cfg.save_path, "train_logodds.pt"))

    # maybe save hiddens
    if cfg.store_hiddens:
        hiddens = torch.cat(hiddens)
        torch.save(hiddens, os.path.join(cfg.save_path, "train_hiddens.pt"))

    # final eval
    final_eval_results = None
    if cfg.eval_every:
        print("Final evaluation:")
        assert (
            final_eval_ds is not None
        ), "must provide eval_ds if eval_every is not None"
        final_eval_results, final_eval_metrics = eval_loop(
            model,
            final_eval_ds,
            cfg.eval_batch_size,
            metric_prefix="eval",
            remove_large_columns=False,
        )
        logger.logkvs(final_eval_metrics)
        logger.dumpkvs()
        final_eval_results.save_to_disk(
            os.path.join(cfg.save_path, "eval_results_final")
        )
        eval_metrics = final_eval_metrics
        update_best()

    # load and and save best model
    if cfg.load_best_model_at_end and best_step and best_step != step:
        print(f"Loading best model from step {best_step}")
        assert best_step is not None
        assert maybe_load_model(
            model, checkpoint_name(best_step)
        ), f"Failed to load the best model from step {best_step}"
    if cfg.save_every:
        ckpt_names.append(os.path.join(cfg.save_path, "pytorch_model.bin"))
        save(model, ckpt_names[-1])
        delete_old_checkpoints()

    print("done.")
    return final_eval_results, final_eval_metrics


def maybe_load_model(model, checkpoint_path, disable=False):
    if os.path.exists(checkpoint_path) and not disable:
        state_dict = torch.load(checkpoint_path)
        (model.module if hasattr(model, "module") else model).load_state_dict(
            state_dict
        )
        return True
    return False


def train_and_save_model(
    cfg: TrainConfig,
    train_ds: datasets.Dataset,
    test_ds: datasets.Dataset,
    inference_ds: Optional[datasets.Dataset] = None,
) -> tuple:
    # if the dataset has a "choice_input_ids" field, we use the LM head
    use_lm_head = "choice_input_ids" in train_ds.features

    print(f"{get_gpu_mem_used() * 100:.2f}% of all GPU memory in use before training")

    already_trained = False
    checkpoint_path = os.path.join(cfg.save_path, "pytorch_model.bin")
    # Load the model
    if cfg.model_config.model_parallel:
        assert (
            torch.cuda.device_count() > 1
        ), f"you might want more gpus for {cfg.model_config.name}"
        model = TransformerWithHead.from_pretrained(
            cfg.model_config.name,
            lora_modules=cfg.model_config.lora_modules,
            use_lm_head=use_lm_head,
            device_map="auto",
            linear_probe=cfg.linear_probe,
            **cfg.model_config.custom_kwargs,
        )
        already_trained = maybe_load_model(model, checkpoint_path, cfg.force_retrain)
    else:
        model = TransformerWithHead.from_pretrained(
            cfg.model_config.name,
            lora_modules=cfg.model_config.lora_modules,
            use_lm_head=use_lm_head,
            linear_probe=cfg.linear_probe,
            **cfg.model_config.custom_kwargs,
        ).to(
            "cuda"  # type: ignore
        )
        already_trained = maybe_load_model(model, checkpoint_path, cfg.force_retrain)
        # data parallel:  currently not supported with model parallel
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, output_device=0)

    if already_trained:
        print("Model already trained, skipping training")
        test_results, test_metrics = eval_loop(
            model,
            test_ds,
            cfg.eval_batch_size,
            metric_prefix="eval",
            remove_large_columns=False,
        )
    else:
        start = time.time()
        test_results, test_metrics = train_model(
            cfg,
            model,
            train_ds,
            final_eval_ds=test_ds,
        )
        print("Model training took", time.time() - start, "seconds")

    inference_results = None
    if inference_ds:
        inference_results, inferenece_metrics = eval_loop(
            model,
            inference_ds,
            cfg.eval_batch_size,
            metric_prefix="inference",
            remove_large_columns=False,
        )
        logger.logkvs(inferenece_metrics)

    with open(os.path.join(cfg.save_path, "results.pkl"), "wb") as f:
        pickle.dump(
            {
                "avg_acc_test": float(
                    np.mean([r["acc"] for r in test_results])  # type: ignore
                ),
                "avg_acc_inference": float(
                    np.mean(
                        [r["acc"] for r in inference_results]  # type: ignore
                        if inference_results
                        else [np.nan]
                    )
                ),
                **test_metrics,
            },
            f,
        )
    logger.shutdown()

    return test_results, inference_results
