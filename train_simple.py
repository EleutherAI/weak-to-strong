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
    if (
        os.path.exists(os.path.join(cfg.save_path, "results_summary.json"))
        and cfg.skip_if_exists
    ):
        print(f"Skipping {cfg.save_path} because it already exists")
        return

    # try to clean up memory
    clear_mem()
    print(f"{get_gpu_mem_used()*100:.2f}% of all GPU memory in use")

    random.seed(cfg.seed)

    print("DS NAME:", cfg.ds_name)
    # Load dataset
    dataset = load_and_process_dataset(
        cfg.ds_name,
        split_sizes=dict(
            train=cfg.n_train_docs + cfg.n_inference_docs, test=cfg.n_test_docs
        ),
        seed=cfg.seed,
        take_test_from_train=cfg.take_test_from_train,
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
            try:
                weak_test_results = load_from_disk(weak_test_results_path)
                # the last minibatch is dropped, so we don't have weak test results for it
                test_ds = test_ds.select(range(len(weak_test_results))).add_column(
                    "weak_soft_label", weak_test_results["soft_pred"]
                )  # type: ignore
                assert test_ds["id"] == weak_test_results["id"], "IDs don't match"
            except (IndexError, AssertionError):
                print(
                    "Weak test results don't match the test dataset, "
                    "some metrics will not be logged."
                )
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
