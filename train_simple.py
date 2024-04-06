import json
import os
import random
import subprocess
from typing import Optional

import fire
import numpy as np
from datasets import load_from_disk

import weak_to_strong.logger as logger
from weak_to_strong.common import get_tokenizer, clear_mem, get_gpu_mem_used
from weak_to_strong.config import (
    MODELS_DICT,
    ModelConfig,
    get_config_foldername,
    LOSS_DICT,
)
from weak_to_strong.datasets import (
    VALID_DATASETS,
    tokenize_dataset,
    load_and_process_dataset,
)
from weak_to_strong.train import train_and_save_model


def main(
    # training batch size (number of examples per update)
    gt_batch_size: int = 32,
    w2s_batch_size: int = 32,
    max_ctx: int = 1024,
    ds_name: str = "sciq",
    loss: str = "kl",
    # number of documents
    n_train1_docs: int = 20000,
    n_train2_docs: int = 10000,
    n_test_docs: int = 10000,
    model_size: str = "gpt2",
    lr: Optional[float] = None,
    # because we use kl loss which typically has smaller gradients,
    # we optionally scale up the learning rate for w2s training
    w2s_lr_factor: float = 1.0,
    optim: Optional[str] = None,
    gt_epochs: int = 1,
    w2s_epochs: int = 1,
    force_retrain: bool = False,
    seed: int = 0,
    # number of examples per forward pass per device
    minibatch_size_per_replica: Optional[int] = None,
    train_with_dropout: bool = False,
    results_folder: str = "/tmp/results",
    # if True, keep the transformer weights frozen and only train the head
    linear_probe: bool = False,
    lr_schedule: str = "cosine_anneal",
    # Note: you can pass either weak_model_size or weak_labels_path. If you pass
    # weak_model_size, we will guess the path to the weak labels based on the weak
    # model. If you pass weak_labels_path, we will use that path instead.
    # If you pass neither, we will train on ground truth.
    weak_model_size: Optional[str] = None,
    weak_labels_path: Optional[str] = None,
    # The subfolder in results_folder to save the results to
    sweep_subfolder: str = "default",
    # Set to a very large value so that by default we don't do any intermediate evals but
    # still do final evals (which requires eval_every to be set to a non-zero, non-None value).
    w2s_eval_every: int = 10000000,
    gt_eval_every: int = 10000000,
    # If set, this command will be run to sync the results to remote storage
    # non-positive values mean we don't save any checkpoints
    sync_command: Optional[str] = None,
    save_every: int = 1000000,
    skip_inference: bool = False,
    skip_if_exists: bool = False,
    # Similar to HF trainer load_best_model_at_end behavior
    # https://huggingface.co/docs/transformers/main_classes/trainer
    load_best_model_at_end: bool = False,
    # using the "against_supervision" suffix will ensure that the metric is measured
    # against whatever labeler was used to train the model, whether weak or strong
    metric_for_best_model: str = "eval/auroc_against_supervision",
    greater_is_better: bool = True,
    save_total_limit: Optional[int] = 1,
    disable_lora: bool = False,
    store_grads: bool = False,
):
    # try to clean up memory
    clear_mem()
    print(f"{get_gpu_mem_used()*100:.2f}% of all GPU memory in use")

    assert (
        ds_name in VALID_DATASETS
    ), f"Unknown dataset {ds_name} not in {VALID_DATASETS}"
    assert (
        weak_model_size is None or weak_labels_path is None
    ), "Can't pass both weak_model_size and weak_labels_path"

    is_w2s = weak_labels_path is not None or weak_model_size is not None
    eval_every = w2s_eval_every if is_w2s else gt_eval_every
    epochs = w2s_epochs if is_w2s else gt_epochs
    loss = loss if is_w2s else "xent"
    batch_size = w2s_batch_size if is_w2s else gt_batch_size
    store_grads = store_grads and is_w2s

    mcfg = MODELS_DICT[model_size].copy()
    if disable_lora:
        del mcfg["lora_modules"]
    model_config = ModelConfig(**mcfg)
    if model_config.model_parallel:
        print(f"Using model parallelism for {model_size}")

    # this is per device!
    if minibatch_size_per_replica is None:
        minibatch_size_per_replica = model_config.minibatch_size_per_replica

    use_model_default_lr = lr is None
    if use_model_default_lr:
        # NOTE: LRs are not super tuned but are best for bs=32,
        # so we use a simple linear scaling otherwise
        # https://stackoverflow.com/questions/53033556/how-should-the-learning-rate-change-as-the-batch-size-change
        lr = model_config.default_lr * batch_size / 32
        print(
            f"Scaling learning rate linearly to {lr} based on batch size {batch_size}. "
            "LRs were tuned for bs=32."
        )
    if is_w2s:
        lr = lr * w2s_lr_factor
        print(
            f"Using learning rate {lr} ({w2s_lr_factor}x the default) for w2s training"
        )

    if optim is None:
        optim = model_config.default_optimizer

    # The commented out terms are the ones that should not change final results
    config = {
        "batch_size": batch_size,
        "max_ctx": max_ctx,
        "ds_name": ds_name,
        "loss": loss,
        "n_train1_docs": n_train1_docs,
        "n_train2_docs": n_train2_docs,
        "n_test_docs": n_test_docs,
        "model_size": model_size,
        "lr": lr,
        "optim": optim,
        "epochs": epochs,
        # "force_retrain": force_retrain,
        "seed": seed,
        # "minibatch_size_per_replica": minibatch_size_per_replica,
        "train_with_dropout": train_with_dropout,
        # "results_folder": results_folder,
        "linear_probe": linear_probe,
        "lr_schedule": lr_schedule,
        # "save_every": save_every,
        # "sweep_subfolder": sweep_subfolder,
        "eval_every": eval_every,
        "load_best_model_at_end": load_best_model_at_end,
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": greater_is_better,
        "save_total_limit": save_total_limit,
        "disable_lora": disable_lora,
        "store_grads": store_grads,
    }

    if weak_model_size is not None:
        weak_model_config = config.copy()
        weak_model_config["model_size"] = weak_model_size
        weak_model_config["loss"] = "xent"
        weak_model_config["epochs"] = gt_epochs
        weak_model_config["eval_every"] = gt_eval_every
        weak_model_config["lr"] = (
            ModelConfig(**MODELS_DICT[weak_model_size]).default_lr
            if use_model_default_lr
            else lr / w2s_lr_factor
        )
        weak_model_config["batch_size"] = gt_batch_size
        weak_model_config["store_grads"] = False

        weak_model_config_name = get_config_foldername(weak_model_config)

        weak_labels_path = (
            results_folder
            + "/"
            + sweep_subfolder
            + "/"
            + weak_model_config_name
            + "/weak_labels"
        )

    eval_batch_size = model_config.eval_batch_size
    random.seed(seed)

    print("DS NAME:", ds_name)
    # Load dataset
    dataset = load_and_process_dataset(
        ds_name,
        seed=seed,
        split_sizes=dict(train=n_train1_docs + n_train2_docs, test=n_test_docs),
    )

    # Split the training dataset in half
    train_dataset, test_ds = dataset["train"], dataset["test"]  # type: ignore

    if weak_labels_path is None:  # train on ground truth
        # split off half for getting weak labels
        split_data = train_dataset.train_test_split(test_size=n_train2_docs, seed=seed)
        train1_ds, train2_ds = split_data["train"], split_data["test"]
        if skip_inference:
            train2_ds = None
            print("len(train1):", len(train1_ds), "(skipping inference)")
        else:
            print("len(train1):", len(train1_ds), "len(train2):", len(train2_ds))
        config_name = get_config_foldername(config)
    else:
        if not weak_labels_path.endswith("weak_labels"):
            weak_labels_path = weak_labels_path + "/weak_labels"
        if sync_command is not None:
            sync_command_list = sync_command.split(" ")
            sync_command_list.extend(
                [
                    "download",
                    weak_labels_path.replace("/weak_labels", ""),
                    results_folder,
                ]
            )
            print(f"Running sync command: {' '.join(sync_command_list)}")
            result = subprocess.run(sync_command_list, check=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Sync command failed with return code {result.returncode}"
                )

        # take the predictions from the weak model to be the labels
        train1_ds = load_from_disk(weak_labels_path).rename_columns(
            {
                "hard_label": "gt_hard_label",
                "soft_label": "gt_soft_label",
                "hard_pred": "hard_label",
                "soft_pred": "soft_label",
            }
        )
        train2_ds = None

        weak_model_config = json.load(
            open(weak_labels_path.replace("weak_labels", "config.json"))
        )
        config["weak_model_size"] = weak_model_config["model_size"]
        config_name = get_config_foldername(config)
        config["weak_model"] = weak_model_config

    save_path = os.path.join(results_folder, sweep_subfolder, config_name)

    if (
        os.path.exists(os.path.join(save_path, "results_summary.json"))
        and skip_if_exists
    ):
        print(f"Skipping {save_path} because it already exists")
        return

    logger.configure(
        save_path=save_path,
        wandb_args=dict(
            project="weak-to-strong",
            config=config,
            group=sweep_subfolder,
            job_type="gt" if weak_labels_path is None else "w2s",
            name=f"{model_size.split('/')[-1]}_{ds_name}_{loss}",
            dir=results_folder,
        ),
    )

    # Tokenize datasets
    tokenizer = get_tokenizer(model_config.name)
    train1_ds = tokenize_dataset(train1_ds, tokenizer, max_ctx)  # type: ignore
    test_ds = tokenize_dataset(test_ds, tokenizer, max_ctx)  # type: ignore
    if train2_ds:
        train2_ds = tokenize_dataset(train2_ds, tokenizer, max_ctx)
    if "for_lm_head" in ds_name:
        assert "choice_input_ids" in train1_ds.column_names
        assert "choice_input_ids" in test_ds.column_names

    # try to add a weak_labels column to the test dataset if running w2s
    if weak_labels_path is not None:
        weak_test_results_path = weak_labels_path.replace(
            "weak_labels", "eval_results_final"
        )
        if os.path.exists(weak_test_results_path):
            weak_test_results = load_from_disk(weak_test_results_path)
            # the last minibatch is dropped, so we don't have weak test results for it
            test_ds = test_ds.select(range(len(weak_test_results))).add_column(
                "weak_soft_label", weak_test_results["soft_pred"]
            )  # type: ignore
            assert test_ds["id"] == weak_test_results["id"], "IDs don't match"
        else:
            print(
                f"No weak test results at {weak_test_results_path}, "
                "some metrics will not be logged."
            )

    loss_fn = LOSS_DICT[loss]
    print(f"Training model {model_size}")
    test_results, weak_ds = train_and_save_model(
        model_config,
        train1_ds,  # this has weak labels iff weak_labels_path is not None
        test_ds,  # this has ground truth labels no matter what
        inference_ds=train2_ds,  # make weak training dataset for strong model
        batch_size=batch_size,
        save_path=save_path,
        loss_fn=loss_fn,
        lr=lr,
        epochs=epochs,
        force_retrain=force_retrain,
        eval_batch_size=eval_batch_size,
        minibatch_size_per_replica=minibatch_size_per_replica,
        train_with_dropout=train_with_dropout,
        linear_probe=linear_probe,
        lr_schedule=lr_schedule,
        optimizer_name=optim,
        eval_every=eval_every,
        save_every=save_every,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        save_total_limit=save_total_limit,
        store_grads=store_grads,
    )

    if weak_ds is not None:
        weak_ds.save_to_disk(save_path + "/" + "weak_labels")

    acc = np.mean([x["acc"] for x in test_results])  # type: ignore
    res_dict = {"accuracy": acc}
    print("accuracy:", acc)

    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(save_path, "results_summary.json"), "w") as f:
        json.dump(res_dict, f, indent=2)

    if sync_command is not None:
        print("Syncing results to remote storage...")
        try:
            sync_command_list = sync_command.split(" ")
            sync_command_list.extend(["upload", save_path, results_folder])
            print(f"Running sync command: {' '.join(sync_command_list)}")
            result = subprocess.run(sync_command_list, check=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Sync command failed with return code {result.returncode}"
                )
        except Exception as e:
            raise RuntimeError("Failed to sync results to remote storage.") from e


if __name__ == "__main__":
    fire.Fire(main)
