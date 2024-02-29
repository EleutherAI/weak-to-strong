import os
import pickle
import time
from typing import Callable, Optional

import datasets
import numpy as np
import torch
import torch_optimizer as toptim
from transformers import get_linear_schedule_with_warmup

import weak_to_strong.logger as logger
from weak_to_strong.common import to_batch, get_gpu_mem_used
from weak_to_strong.eval import eval_loop, compute_metrics
from weak_to_strong.loss import kl_loss
from weak_to_strong.model import TransformerWithHead
from weak_to_strong.config import ModelConfig


def save(
    model: torch.nn.Module,
    save_file: Optional[str],
):
    # Note: If the model is wrapped by DataParallel, we need to unwrap it before saving
    model_to_save = model.module if hasattr(model, "module") else model

    model_to_save.save_state_dict(save_file)
    print("saved torch weights", save_file)


def train_model(
    model: torch.nn.Module,
    ds: datasets.Dataset,
    batch_size: int,
    lr: float = 1e-5,
    loss_fn: Callable = kl_loss,
    print_every: int = 10,
    eval_every: Optional[int] = None,
    save_every: Optional[int] = None,
    eval_batch_size: int = 256,
    minibatch_size: int = 8,
    eval_ds: Optional[datasets.Dataset] = None,
    gradient_checkpointing: bool = False,
    train_with_dropout: bool = False,
    epochs: int = 1,
    save_path: Optional[str] = None,
    lr_schedule: str = "cosine_anneal",
    optimizer_name: str = "adam",
    # Similar to HF trainer load_best_model_at_end behavior
    # https://huggingface.co/docs/transformers/main_classes/trainer
    load_best_model_at_end: bool = False,
    # using the "against_supervision" suffix will ensure that the metric is measured
    # against whatever labeler was used to train the model, whether weak or strong
    metric_for_best_model: str = "eval/auroc_against_supervision",
    greater_is_better: bool = True,
    save_total_limit: Optional[int] = 1,
):
    """
    ds is a dataset of examples, each of which is a dict with keys:
    - input_ids: a list of token ids
    - soft_label: a list of soft label probabilities
    - choice_input_ids (optional): a pair of token ids for the answer choices,
        indicating to use the LM head of the model
    """
    print(
        f"LR: {lr}, batch size: {batch_size}, mbatch size: {minibatch_size}, n: {len(ds)}"
    )
    assert (
        batch_size % minibatch_size == 0
    ), "batch size must be divisible by minibatch size"

    def checkpoint_name(step):
        assert (
            save_path is not None
        ), "save_path must not be None if save_every is not None"
        return os.path.join(save_path, f"checkpoint_{step}.bin")

    is_w2s = "gt_soft_label" in ds.features
    if metric_for_best_model.endswith("_against_supervision"):
        metric_for_best_model = metric_for_best_model.replace(
            "_against_supervision", "_against_weak" if is_w2s else ""
        )

    # we purposefully turn off dropout, for determinism
    # this seems to help for 1 epoch finetuning anyways
    model.train(mode=train_with_dropout)
    if gradient_checkpointing:
        (
            model if hasattr(model, "gradient_checkpointing_enable") else model.module
        ).gradient_checkpointing_enable()

    nsteps = len(ds) * epochs // batch_size

    def lr_schedule_fn(step):
        if lr_schedule == "constant":
            return 1
        else:
            assert (
                False
            ), f"invalid lr schedule, {lr_schedule}, must be constant or cosine_anneal"

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=lr, betas=(0.9, 0.95))
    elif optimizer_name.lower() == "adafactor":
        optimizer = toptim.Adafactor(trainable_params, lr=lr)
    else:
        assert False, f"invalid optimizer {optimizer_name}, must be adam or adafactor"
    if lr_schedule == "cosine_anneal":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, nsteps)
    elif lr_schedule == "linear_with_warmup":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=50, num_training_steps=nsteps
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule_fn)

    step = 0
    losses = []
    accuracies = []
    aurocs = []
    best_eval = float("-inf") if greater_is_better else float("inf")
    best_step = None
    ckpt_names = []

    def delete_old_checkpoints():
        if save_total_limit is None:
            return
        num_to_delete = len(ckpt_names) - save_total_limit
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
        if load_best_model_at_end:
            current_eval = eval_metrics[metric_for_best_model]
            if (greater_is_better and current_eval > best_eval) or (
                not greater_is_better and current_eval < best_eval
            ):
                assert os.path.exists(checkpoint_name(step)), (
                    "No checkpoint found "
                    "for the current step, "
                    "but load_best_model_at_end was set to True and the current step is "
                    "best. Please set save_every to a multiple of eval_every."
                )
                best_eval = current_eval
                best_step = step
                print(f"New best model found at step {step}")

    # If the model is wrapped by DataParallel, it doesn't have a device. In this case,
    # we use GPU 0 as the output device. This sadly means that this device will store
    # a bit more data than other ones, but hopefully should not be too big of a deal.
    io_device = model.device if hasattr(model, "device") else 0

    for epoch in range(epochs):
        for start in range(0, len(ds), batch_size):
            loss_tot = 0

            # save
            if save_every and step % save_every == 0 and save_every < nsteps:
                ckpt_names.append(checkpoint_name(step))
                save(model, ckpt_names[-1])
                delete_old_checkpoints()

            # eval
            if eval_every and step % eval_every == 0 and eval_every < nsteps:
                assert (
                    eval_ds is not None
                ), "must provide eval_ds if eval_every is not None"
                eval_results, eval_metrics = eval_loop(
                    model,
                    eval_ds,
                    eval_batch_size,
                    metric_prefix="eval",
                    remove_large_columns=True,
                )
                logger.logkvs(eval_metrics)
                if save_path is not None:
                    eval_results.save_to_disk(
                        os.path.join(save_path, f"eval_results_{step}")
                    )
                if gradient_checkpointing:
                    (
                        model
                        if hasattr(model, "gradient_checkpointing_enable")
                        else model.module
                    ).gradient_checkpointing_enable()
                model.train(mode=train_with_dropout)

                update_best()

            # train step
            all_logits = []
            all_labels = []
            for mbatch in to_batch(
                ds, minibatch_size, start=start, end=start + batch_size
            ):
                input_ids = (
                    torch.nn.utils.rnn.pad_sequence(
                        [torch.tensor(ids) for ids in mbatch["input_ids"]]  # type: ignore
                    )
                    .transpose(0, 1)
                    .to(io_device)  # type: ignore
                )
                labels = torch.tensor(mbatch["soft_label"]).to(io_device)  # type: ignore
                logits = model(
                    input_ids, choice_input_ids=mbatch.get("choice_input_ids")
                ).to(io_device)
                loss = loss_fn(logits, labels, step_frac=step / nsteps)
                loss_tot += loss.item()
                # we don't need to use a gradscaler because we're using bf16 instead of fp16
                loss.backward()

                all_logits.extend(logits)
                all_labels.extend(labels)

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

            if print_every and step % print_every == 0:
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
    if save_every and checkpoint_name(step) not in ckpt_names:
        ckpt_names.append(checkpoint_name(step))
        save(model, ckpt_names[-1])
        delete_old_checkpoints()

    # final eval
    final_eval_results = None
    if eval_every:
        print("Final evaluation:")
        assert eval_ds is not None, "must provide eval_ds if eval_every is not None"
        final_eval_results, final_eval_metrics = eval_loop(
            model,
            eval_ds,
            eval_batch_size,
            metric_prefix="eval",
            remove_large_columns=False,
        )
        logger.logkvs(final_eval_metrics)
        logger.dumpkvs()
        if save_path is not None:
            final_eval_results.save_to_disk(
                os.path.join(save_path, "eval_results_final")
            )
        eval_metrics = final_eval_metrics
        update_best()

    # load and and save best model
    if load_best_model_at_end and best_step != step:
        print(f"Loading best model from step {best_step}")
        assert best_step is not None
        assert maybe_load_model(model, checkpoint_name(best_step)), (
            "Failed to load " "the best model."
        )
    if save_every:
        assert (
            save_path is not None
        ), "save_path must not be None if save_every is not None"
        ckpt_names.append(os.path.join(save_path, "pytorch_model.bin"))
        save(model, ckpt_names[-1])
        delete_old_checkpoints()

    print("done.")
    return final_eval_results, final_eval_metrics


def maybe_load_model(model, checkpoint_path, disable=False):
    if os.path.exists(checkpoint_path) and not disable:
        state_dict = torch.load(checkpoint_path)
        state_dict = {
            k.replace("transformer.module", "transformer"): v
            for (k, v) in state_dict.items()
        }
        (model if hasattr(model, "load_state_dict") else model.module).load_state_dict(
            state_dict
        )
        return True
    return False


def train_and_save_model(
    model_config: ModelConfig,
    train_ds: datasets.Dataset,
    test_ds: datasets.Dataset,
    inference_ds: Optional[datasets.Dataset] = None,
    *,
    batch_size: int,
    lr: float,
    epochs: int,
    save_path: str,
    eval_batch_size: Optional[int] = None,
    minibatch_size_per_replica: Optional[int] = None,
    loss_fn: Callable = kl_loss,
    force_retrain: bool = False,
    train_with_dropout: bool = False,
    linear_probe: bool = False,
    lr_schedule: str = "constant",
    optimizer_name: str = "adam",
    eval_every: Optional[int] = None,
    save_every: Optional[int] = None,
    load_best_model_at_end: bool = False,
    metric_for_best_model: str = "eval/auroc",
    greater_is_better: bool = True,
    save_total_limit: Optional[int] = 1,
) -> tuple:
    if eval_batch_size is None:
        eval_batch_size = batch_size

    if minibatch_size_per_replica is None:
        minibatch_size_per_replica = 1

    # if the dataset has a "choice_input_ids" field, we use the LM head
    use_lm_head = "choice_input_ids" in train_ds.features

    gradient_checkpointing = model_config.gradient_checkpointing
    custom_kwargs = model_config.custom_kwargs or {}

    print(f"{get_gpu_mem_used() * 100:.2f}% of all GPU memory in use before training")

    already_trained = False
    checkpoint_path = os.path.join(save_path, "pytorch_model.bin")
    # Load the model
    if model_config.model_parallel:
        assert (
            torch.cuda.device_count() > 1
        ), f"you might want more gpus for {model_config.name}"
        model = TransformerWithHead.from_pretrained(
            model_config.name,
            lora_modules=model_config.lora_modules,
            use_lm_head=use_lm_head,
            num_labels=2,
            device_map="auto",
            linear_probe=linear_probe,
            **custom_kwargs,
        )
        already_trained = maybe_load_model(model, checkpoint_path, force_retrain)
        minibatch_size = minibatch_size_per_replica
    else:
        model = TransformerWithHead.from_pretrained(
            model_config.name,
            lora_modules=model_config.lora_modules,
            use_lm_head=use_lm_head,
            num_labels=2,
            linear_probe=linear_probe,
            **custom_kwargs,
        ).to(
            "cuda"  # type: ignore
        )
        already_trained = maybe_load_model(model, checkpoint_path, force_retrain)
        # data parallel:  currently not supported with model parallel
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, output_device=0)
            minibatch_size = min(
                minibatch_size_per_replica * torch.cuda.device_count(), batch_size
            )
            eval_batch_size = min(torch.cuda.device_count(), eval_batch_size)
            print(
                "Using",
                torch.cuda.device_count(),
                "GPUs, setting minibatch_size to",
                minibatch_size,
                "and eval_batch_size to",
                eval_batch_size,
            )
        else:
            minibatch_size = minibatch_size_per_replica

    if already_trained:
        test_results, test_metrics = eval_loop(
            model,
            test_ds,
            eval_batch_size,
            metric_prefix="eval",
            remove_large_columns=False,
        )
    else:
        start = time.time()
        test_results, test_metrics = train_model(
            model,
            train_ds,
            batch_size,
            lr=lr,
            epochs=epochs,
            save_path=save_path,
            eval_ds=test_ds,
            gradient_checkpointing=gradient_checkpointing,
            loss_fn=loss_fn,
            eval_batch_size=eval_batch_size,
            eval_every=eval_every,
            minibatch_size=minibatch_size,
            train_with_dropout=train_with_dropout,
            lr_schedule=lr_schedule,
            optimizer_name=optimizer_name,
            save_every=save_every,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            save_total_limit=save_total_limit,
        )
        print("Model training took", time.time() - start, "seconds")

    inference_results = None
    if inference_ds:
        inference_results, inferenece_metrics = eval_loop(
            model,
            inference_ds,
            eval_batch_size,
            metric_prefix="inference",
            remove_large_columns=False,
        )
        logger.logkvs(inferenece_metrics)

    if save_path:
        with open(os.path.join(save_path, "results.pkl"), "wb") as f:
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
