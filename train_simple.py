import json
import os
import random

import numpy as np
from datasets import load_from_disk
from simple_parsing import ArgumentParser

import weak_to_strong.logger as logger
from weak_to_strong.common import get_tokenizer, clear_mem, get_gpu_mem_used
from weak_to_strong.train_config import TrainConfig
from weak_to_strong.datasets import tokenize_dataset, load_and_process_dataset
from weak_to_strong.train import train_and_save_model


def main(cfg: TrainConfig):
    # try to clean up memory
    clear_mem()
    print(f"{get_gpu_mem_used()*100:.2f}% of all GPU memory in use")

    random.seed(cfg.seed)

    print("DS NAME:", cfg.ds_name)
    # Load dataset
    dataset = load_and_process_dataset(
        cfg.ds_name,
        seed=cfg.seed,
        split_sizes=dict(
            train=cfg.n_train_docs + cfg.n_inference_docs, test=cfg.n_test_docs
        ),
    )

    # Split the training dataset in half
    train_dataset, test_ds = dataset["train"], dataset["test"]  # type: ignore

    if cfg.weak_labels_path is None:  # train on ground truth
        # split off half for getting weak labels
        if cfg.n_inference_docs:
            split_data = train_dataset.train_test_split(
                test_size=cfg.n_inference_docs, seed=cfg.seed
            )
            train_ds, inference_ds = split_data["train"], split_data["test"]
        else:
            train_ds, inference_ds = train_dataset, None

        print(
            "len(train):",
            len(train_ds),
            "len(inference):",
            len(inference_ds) if inference_ds else 0,
        )
    else:
        # take the predictions from the weak model to be the labels
        train_ds = load_from_disk(cfg.weak_labels_path).rename_columns(
            {
                "hard_label": "gt_hard_label",
                "soft_label": "gt_soft_label",
                "hard_pred": "hard_label",
                "soft_pred": "soft_label",
            }
        )
        inference_ds = None

    if (
        os.path.exists(os.path.join(cfg.save_path, "results_summary.json"))
        and cfg.skip_if_exists
    ):
        print(f"Skipping {cfg.save_path} because it already exists")
        return

    logger.configure(
        save_path=cfg.save_path,
        wandb_args=dict(
            project="weak-to-strong",
            config=cfg.effective_config,
            group=cfg.sweep_subfolder,
            job_type="w2s" if cfg.is_w2s else "gt",
            name=f"{cfg.model_config.name.split('/')[-1]}_{cfg.ds_name}_{cfg.loss}",
            dir=cfg.results_folder,
        ),
    )

    # Tokenize datasets
    tokenizer = get_tokenizer(cfg.model_config.name)
    train_ds = tokenize_dataset(train_ds, tokenizer, cfg.max_ctx)  # type: ignore
    test_ds = tokenize_dataset(test_ds, tokenizer, cfg.max_ctx)  # type: ignore
    if inference_ds:
        inference_ds = tokenize_dataset(inference_ds, tokenizer, cfg.max_ctx)
    if "for_lm_head" in cfg.ds_name:
        assert "choice_input_ids" in train_ds.column_names
        assert "choice_input_ids" in test_ds.column_names

    # try to add a weak_labels column to the test dataset if running w2s
    if cfg.weak_labels_path is not None:
        weak_test_results_path = cfg.weak_labels_path.replace(
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

    print(f"Training model {cfg.model_size}")
    test_results, weak_ds = train_and_save_model(
        cfg,
        train_ds,
        test_ds,
        inference_ds,
    )

    if weak_ds is not None:
        weak_ds.save_to_disk(cfg.save_path + "/" + "weak_labels")

    acc = np.mean([x["acc"] for x in test_results])  # type: ignore
    res_dict = {"accuracy": acc}
    print("accuracy:", acc)

    with open(os.path.join(cfg.save_path, "config.json"), "w") as f:
        json.dump(cfg.effective_config, f, indent=2)

    with open(os.path.join(cfg.save_path, "results_summary.json"), "w") as f:
        json.dump(res_dict, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser(add_config_path_arg=True)
    # Example usage:
    # python train_simple.py --config_path configs/train_simple.yaml --model_size gpt2
    parser.add_arguments(TrainConfig, dest="cfg")  # type: ignore
    main(parser.parse_args().cfg)


# assert (
#     ds_name in VALID_DATASETS
# ), f"Unknown dataset {ds_name} not in {VALID_DATASETS}"
# assert (
#     weak_model_size is None or weak_labels_path is None
# ), "Can't pass both weak_model_size and weak_labels_path"

# is_w2s = weak_labels_path is not None or weak_model_size is not None
# eval_every = w2s_eval_every if is_w2s else gt_eval_every
# epochs = w2s_epochs if is_w2s else gt_epochs
# loss = loss if is_w2s else "xent"
# batch_size = w2s_batch_size if is_w2s else gt_batch_size
# store_grads = store_grads and is_w2s

# print(f"model_size: {model_size}, model_cfg_name: {model_cfg_name}")
# mcfg = MODELS_DICT[model_cfg_name or model_size].copy()
# if disable_lora:
#     del mcfg["lora_modules"]
# if model_cfg_name is not None:
#     mcfg["name"] = model_size
# model_config = ModelConfig(**mcfg)
# if model_config.model_parallel:
#     print(f"Using model parallelism for {model_size}")

# this is per device!
# if minibatch_size_per_replica is None:
#     minibatch_size_per_replica = model_config.minibatch_size_per_replica

# use_model_default_lr = lr is None
# if use_model_default_lr:
#     # NOTE: LRs are not super tuned but are best for bs=32,
#     # so we use a simple linear scaling otherwise
#     # https://stackoverflow.com/questions/53033556/how-should-the-learning-rate-change-as-the-batch-size-change
#     lr = model_config.default_lr * batch_size / 32
#     print(
#         f"Scaling learning rate linearly to {lr} based on batch size {batch_size}. "
#         "LRs were tuned for bs=32."
#     )
# if is_w2s:
#     lr = lr * w2s_lr_factor
#     print(
#         f"Using learning rate {lr} ({w2s_lr_factor}x the default) for w2s training"
#     )

# if optim is None:
#     optim = model_config.default_optimizer

# The commented out terms are the ones that should not change final results
# config = {
#     "batch_size": batch_size,
#     "max_ctx": max_ctx,
#     "ds_name": ds_name,
#     "loss": loss,
#     "n_train1_docs": n_train1_docs,
#     "n_train2_docs": n_train2_docs,
#     "n_test_docs": n_test_docs,
#     "model_size": model_size,
#     "lr": lr,
#     "optim": optim,
#     "epochs": epochs,
#     # "force_retrain": force_retrain,
#     "seed": seed,
#     # "minibatch_size_per_replica": minibatch_size_per_replica,
#     "train_with_dropout": train_with_dropout,
#     # "results_folder": results_folder,
#     "linear_probe": linear_probe,
#     "lr_schedule": lr_schedule,
#     # "save_every": save_every,
#     # "sweep_subfolder": sweep_subfolder,
#     "eval_every": eval_every,
#     "load_best_model_at_end": load_best_model_at_end,
#     "save_total_limit": save_total_limit,
#     "disable_lora": disable_lora,
#     "store_grads": store_grads,
# }
# if store_grads:
#     config["d_downsample"] = d_downsample

# if weak_model_size is not None:
#     weak_model_config = config.copy()
#     weak_model_config["model_size"] = weak_model_size
#     weak_model_config["loss"] = "xent"
#     weak_model_config["epochs"] = gt_epochs
#     weak_model_config["eval_every"] = gt_eval_every
#     weak_model_config["lr"] = (
#         ModelConfig(**MODELS_DICT[weak_model_size]).default_lr
#         if use_model_default_lr
#         else lr / w2s_lr_factor
#     )
#     weak_model_config["batch_size"] = gt_batch_size
#     weak_model_config["store_grads"] = False

#     weak_model_config_name = get_config_foldername(weak_model_config)

#     weak_labels_path = (
#         results_folder
#         + "/"
#         + sweep_subfolder
#         + "/"
#         + weak_model_config_name
#         + "/weak_labels"
#     )

# eval_batch_size = model_config.eval_batch_size
