import os
import pickle
import time
from typing import Callable, Optional

import datasets
import numpy as np
import torch
import torch_optimizer as toptim
from transformers.modeling_utils import load_sharded_checkpoint
from sklearn.metrics import roc_auc_score
from transformers import get_linear_schedule_with_warmup
import wandb

import weak_to_strong.logger as logger
from weak_to_strong.common import clear_mem, to_batch, wandb_finish
from weak_to_strong.eval import eval_model_acc, extract_accuracy
from weak_to_strong.loss import kl_loss
from weak_to_strong.config import ModelConfig


def save(
    model: torch.nn.Module, save_path: str, name: str,
    optimizer=None, scheduler=None
):
    # Note: If the model is wrapped by DataParallel, we need to
    # unwrap it before saving
    model_to_save = model.module if hasattr(model, "module") else model

    save_file = os.path.join(save_path, f"{name}.bin")
    model_to_save.save_torch(save_file, optimizer, scheduler)
    print("saved torch weights", save_file)


def train_model(
    model: torch.nn.Module,
    ds: datasets.Dataset,
    batch_size: int,
    lr: float = 1e-5,
    loss_fn: Callable = kl_loss,
    log_every: Optional[int] = None,
    eval_every: Optional[int] = None,
    eval_batch_size: int = 256,
    minibatch_size: int = 8,
    eval_ds: Optional[datasets.Dataset] = None,
    gradient_checkpointing: bool = False,
    train_with_dropout: bool = False,
    epochs: int = 1,
    save_path: Optional[str] = None,
    lr_schedule: str = "cosine_anneal",
    optimizer_name: str = "adam",
):
    """
    ds is a dataset of examples, each of which is a dict with keys:
    - input_ids: a list of token ids
    - soft_label: a list of soft label probabilities
    """

    print("LR", lr, "batch_size", batch_size, "minibatch_size", minibatch_size)
    assert (
        batch_size % minibatch_size == 0
    ), "batch size must be divisible by minibatch size"
    # we purposefully turn off dropout, for determinism
    # this seems to help for 1 epoch finetuning anyways
    if train_with_dropout:
        model.train()
    else:
        model.eval()
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
    eval_acc_dict = {}
    max_acc = 0.0

    # If the model is wrapped by DataParallel, it doesn't have a device. In this case,
    # we use GPU 0 as the output device. This sadly means that this device will store
    # a bit more data than other ones, but hopefully should not be too big of a deal.
    io_device = model.device if hasattr(model, "device") else 0
    best_eval_results = None
    for epoch in range(epochs):
        for start in range(0, len(ds), batch_size):
            loss_tot = 0
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
                )

                all_logits.extend(logits.to(io_device))
                all_labels.extend(labels)
            all_logits = torch.stack(all_logits)
            all_labels = torch.stack(all_labels)
            all_hard_labels = torch.argmax(all_labels, dim=1)
            all_logprobs = torch.nn.functional.log_softmax(
                all_logits.detach().float(), dim=1
            )[:, 1]
            loss = loss_fn(all_logits, all_labels, step_frac=step / nsteps)
            loss_tot += loss.item()
            loss.backward()
            losses.append(loss_tot)
            accuracies.append(
                torch.mean(
                    (torch.argmax(all_logits, dim=1) == all_hard_labels).to(
                        torch.float32
                    )
                ).item()
            )

            try:
                auroc = roc_auc_score(all_hard_labels.cpu(), all_logprobs.cpu())
            except ValueError as e:
                print(f"Warning: {e}")
                auroc = np.nan
            aurocs.append(auroc)

            log_dict = {
                    "step": step,
                    "progress": step / nsteps,
                    "loss": loss_tot,
                    "train_accuracy": accuracies[-1],
                    "train_auroc": aurocs[-1],
                    "lr": lr_scheduler.get_last_lr()[0],
            }
            logger.logkvs(log_dict)
            wandb.log(log_dict)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            if log_every and step % log_every == 0:
                print(
                    f"Step: {step}/{nsteps}; loss: {np.mean(losses)}; "
                    f"train acc: {np.mean(accuracies)}; "
                    f"train auroc: {np.mean(aurocs)}; ({len(losses)} losses)"
                )
                losses = []
                accuracies = []
                aurocs = []
            if eval_every and step % eval_every == 0:
                assert (
                    eval_ds is not None
                ), "must provide eval_ds if eval_every is not None"
                eval_results = eval_model_acc(model, eval_ds, eval_batch_size)
                if gradient_checkpointing:
                    (
                        model
                        if hasattr(model, "gradient_checkpointing_enable")
                        else model.module
                    ).gradient_checkpointing_enable()
                if train_with_dropout:
                    model.train()
                eval_acc = extract_accuracy(eval_results)
                eval_acc_dict[step] = eval_acc
                if eval_acc > max_acc:
                    best_eval_results = eval_results
                    if save_path:
                        save(
                            model,
                            save_path,
                            "best_model",
                            optimizer,
                            lr_scheduler
                        )
                    
                max_acc = max(max_acc, eval_acc)
                logger.logkv("eval_accuracy", eval_acc)
                wandb.log({"eval/accuracy": eval_acc})

            step += 1
            logger.dumpkvs()

    final_eval_results = None
    if eval_every:
        print("Final evaluation:")
        assert eval_ds is not None, "must provide eval_ds if eval_every is not None"
        final_eval_results = eval_model_acc(model, eval_ds, eval_batch_size)
        eval_acc = extract_accuracy(final_eval_results)
        logger.logkv("eval_accuracy", eval_acc)
        wandb.log({"eval/accuracy": eval_acc})
        logger.dumpkvs()
    if save_path:
        save(model, save_path, "final_model", optimizer, lr_scheduler)
    return best_eval_results, final_eval_results


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
    minibatch_size_per_device: Optional[int] = None,
    loss_fn: Callable = kl_loss,
    force_retrain: bool = False,
    train_with_dropout: bool = False,
    linear_probe: bool = False,
    lr_schedule: str = "constant",
    optimizer_name: str = "adam",
    eval_every: Optional[int] = None,
    log_every: Optional[int] = None,
) -> tuple:
    if eval_batch_size is None:
        eval_batch_size = batch_size

    if minibatch_size_per_device is None:
        minibatch_size_per_device = 1

    # if the dataset has a "choice_input_ids" field, we use the LM head
    use_lm_head = "choice_input_ids" in train_ds.features

    gradient_checkpointing = model_config.gradient_checkpointing
    custom_kwargs = model_config.custom_kwargs or {}

    def maybe_load_model(model):
        pkl_path = os.path.join(save_path, "results.pkl")
        if os.path.exists(pkl_path) and not force_retrain:
            print("loading from", save_path)
            checkpoint_path = os.path.join(save_path, "final_model.bin")
            if not os.path.exists(checkpoint_path):
                # Assume this means we have a sharded checkpoint, and
                # load it appropriately
                load_sharded_checkpoint(model, checkpoint_path)
            else:
                state_dict = torch.load(checkpoint_path)
                state_dict = {
                    k.replace("transformer.module", "transformer"): v
                    for (k, v) in state_dict.items()
                }
                custom_kwargs["state_dict"] = state_dict
            return True
        return False

    # Load the model
    model, minibatch_size = model_config.load_model(
        batch_size=batch_size,
        use_lm_head=use_lm_head,
        linear_probe=linear_probe,
        minibatch_size_per_device=minibatch_size_per_device,
    )
    already_trained = maybe_load_model(model)

    if already_trained:
        best_test_results = None
        final_test_results = eval_model_acc(model, test_ds, eval_batch_size)
    else:
        start = time.time()
        best_test_results, final_test_results = train_model(
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
            log_every=log_every,
            minibatch_size=minibatch_size,
            train_with_dropout=train_with_dropout,
            lr_schedule=lr_schedule,
            optimizer_name=optimizer_name,
        )
        print("Model training took", time.time() - start, "seconds")    

    inference_results = None
    if inference_ds:
        inference_results = eval_model_acc(model, inference_ds, eval_batch_size)
        inf_acc = extract_accuracy(inference_results)
        logger.logkv("inference_accuracy", inf_acc)
        wandb.log({"inference/accuracy": inf_acc})

    if save_path:
        with open(os.path.join(save_path, "results.pkl"), "wb") as f:
            pickle.dump(
                {
                    "avg_acc_test": float(
                        np.mean([r["acc"] for r in final_test_results])  # type: ignore
                    ),
                    "avg_acc_inference": float(
                        np.mean(
                            [r["acc"] for r in inference_results]  # type: ignore
                            if inference_results
                            else [np.nan]
                        )
                    ),
                    "test_results": final_test_results,
                    "inference_results": inference_results if inference_results else [],
                },
                f,
            )
    # try to clean up memory
    clear_mem()
    logger.shutdown()
    wandb_finish()

    return best_test_results, final_test_results, inference_results
