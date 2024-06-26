import traceback
from typing import List, Optional, Union
import os
import json
import warnings

import numpy as np
import fire

from weak_to_strong.datasets import load_and_process_dataset
from weak_to_strong.train_config import TrainConfig
from train_simple import main as train_simple_main
from sweep import split_possible_string_list, split_args_by_run_type


def get_quirky_model_name(
    ds_name,
    base_model_id,
    templatization_method="first",
    standardize_templates=False,
    weak_only=False,
    full_finetuning=False,
    model_hub_user="EleutherAI",
):
    base_model_last = base_model_id.split("/")[-1]
    model_id = (
        f"{model_hub_user}/{base_model_last}-{ds_name}-"
        + templatization_method
        + ("-standardized" if standardize_templates else "")
        + ("-weak-only" if weak_only else "")
        + ("-ft" if full_finetuning else "")
    )
    model_last = model_id.split("/")[-1]
    return model_id, model_last


def create_weak_labels(
    ds_name: str,
    n_train_docs: int,
    n_test_docs: int,
    weak_label_accuracy: Optional[float] = None,
    **args,
):
    """
    ds_name: quirky dataset name of the form "quirky_x"
    Saves 3 files to the sweep subfolder:
    - weak_labels: train dataset with "hard_pred" and "soft_pred" columns
        corresponding to Bob's labels, and "hard_label" and "soft_label" columns
        corresponding to the ground truth labels.
    - eval_results_final: test dataset with similarly named columns.
    - config.json: a dictionary with the weak "model" configuration.

    """
    if "results_folder" not in args:
        warnings.warn("No results_folder provided, using /tmp/results")
    results_folder = args.get("results_folder", "/tmp/results")
    sweep_subfolder = args.get("sweep_subfolder", "default")
    seed = args.get("seed", 0)

    weak_model_config = {
        "model_size": "Bob",  # indicates Bob's labels
        "n_train_docs": n_train_docs,
        "n_test_docs": n_test_docs,
        "results_folder": results_folder,
        "sweep_subfolder": sweep_subfolder,
        "ds_name": ds_name,
        "seed": seed,
    }

    weak_labels_subfolder = os.path.join(
        results_folder,
        sweep_subfolder,
        f"dn={ds_name}-ntr={n_train_docs}-nte={n_test_docs}",
    )
    os.makedirs(weak_labels_subfolder, exist_ok=True)

    with open(os.path.join(weak_labels_subfolder, "config.json"), "w") as f:  # type: ignore
        json.dump(weak_model_config, f)

    def load(is_weak: bool):
        load_ds_name = ds_name + "_weak" if is_weak else ds_name

        return load_and_process_dataset(
            load_ds_name,
            seed=seed,
            split_sizes=dict(train=n_train_docs, test=n_test_docs),
        )

    ds_dicts = {
        "weak": load(is_weak=True),
        "gt": load(is_weak=False),
    }

    for split, save_name in [("train", "weak_labels"), ("test", "eval_results_final")]:
        assert (
            np.array(ds_dicts["gt"][split]["id"])
            == np.array(ds_dicts["weak"][split]["id"])
        ).all()

        # add weak labels
        ds = (
            ds_dicts["gt"][split]
            .add_column("hard_pred", ds_dicts["weak"][split]["hard_label"])
            .add_column("soft_pred", ds_dicts["weak"][split]["soft_label"])
        )

        if weak_label_accuracy is not None:
            # resample weak labels to have a specified accuracy

            is_correct = np.array(ds["hard_pred"]) == np.array(ds["hard_label"])
            acc = is_correct.mean()
            correct_indices = np.where(is_correct)[0]
            incorrect_indices = np.where(~is_correct)[0]
            if weak_label_accuracy == 1:
                keep_indices = correct_indices
            elif weak_label_accuracy == 0:
                keep_indices = incorrect_indices
            else:
                desired_c_to_i_ratio = weak_label_accuracy / (1 - weak_label_accuracy)
                if acc < weak_label_accuracy:
                    n_correct_keep = is_correct.sum()
                    n_incorrect_keep = int(n_correct_keep / desired_c_to_i_ratio)
                    incorrect_indices = np.random.choice(
                        incorrect_indices, n_incorrect_keep, replace=False
                    )
                else:
                    n_incorrect_keep = (~is_correct).sum()
                    n_correct_keep = int(n_incorrect_keep * desired_c_to_i_ratio)
                    correct_indices = np.random.choice(
                        correct_indices, n_correct_keep, replace=False
                    )
                keep_indices = np.concatenate([correct_indices, incorrect_indices])
            ds = ds.select(keep_indices).shuffle(seed=seed)
            new_gt, new_hard_pred = np.array(ds["hard_label"]), np.array(
                ds["hard_pred"]
            )
            assert np.isclose(
                (new_gt == new_hard_pred).mean(),
                weak_label_accuracy,
                atol=1 / len(new_gt),
            )

        ds.save_to_disk(os.path.join(weak_labels_subfolder, save_name))
    return os.path.join(weak_labels_subfolder, "weak_labels")


def pretrained_quirky(
    quirky_ds_names: List[str],
    strong_model_sizes: List[str],
    n_train_docs: int = 10_000,
    n_test_docs: int = 1_000,
    weak_label_accuracy: Optional[float] = None,
    **args,
):
    # resample weak labels to have a specified accuracy
    assert (
        "n_inference_docs" not in args
    ), "Use n_train_docs instead of n_inference_docs"
    quirky_ds_names = split_possible_string_list(quirky_ds_names)
    strong_model_sizes = split_possible_string_list(strong_model_sizes)

    gt_args, w2s_args = split_args_by_run_type(args)

    for quirky_ds_abbrev in quirky_ds_names:
        # first create a weak_labels_path
        weak_labels_path = create_weak_labels(
            ds_name=f"quirky_{quirky_ds_abbrev}",
            n_train_docs=n_train_docs,
            n_test_docs=n_test_docs,
            weak_label_accuracy=weak_label_accuracy,
            **gt_args,
        )

        for model_name in strong_model_sizes:
            # then run gt and w2s runs on the provided model
            print(f"Running {model_name} on {quirky_ds_abbrev}...")
            try:
                cfg = TrainConfig(
                    ds_name=f"quirky_{quirky_ds_abbrev}",
                    model_size=model_name,
                    n_train_docs=n_train_docs,
                    n_inference_docs=0,
                    n_test_docs=n_test_docs,
                    **gt_args,
                )
                train_simple_main(cfg)
            except Exception as e:
                print(
                    f"Failed to run ground truth {model_name} on {quirky_ds_abbrev}: {e}"
                )
                traceback.print_exc()

            try:
                # run weak-to-strong
                cfg = TrainConfig(
                    ds_name=f"quirky_{quirky_ds_abbrev}",
                    model_size=model_name,
                    n_train_docs=n_train_docs,
                    n_inference_docs=0,
                    n_test_docs=n_test_docs,
                    weak_labels_path=weak_labels_path,
                    **w2s_args,
                )
                train_simple_main(cfg)
            except Exception as e:
                print(
                    f"Failed to run weak-to-strong {model_name} on {quirky_ds_abbrev}: {e}"
                )
                traceback.print_exc()

    print(f"Finished running models on {quirky_ds_names} x {strong_model_sizes}")


def main(
    base_model_names: Union[List[str], str],
    quirky_ds_names: Union[List[str], str],
    n_train_docs: int = 10_000,
    n_test_docs: int = 1_000,
    standardized_templates: bool = False,
    full_finetuning: bool = False,
    **args,
):
    """
    base_model_names: list of base model names, optionally include hub user id
    quirky_ds_names: list of quirky dataset names `x` such that "EleutherAI/quirky_x_raw"
                     is the desired dataset name registered in `weak_to_strong/datasets.py`
    standardized_templates: whether to use the quirky model trained on standardized templates
    full_finetuning: whether to use the quirky model trained with full finetuning
    """
    assert (
        "n_inference_docs" not in args
    ), "Use n_train_docs instead of n_inference_docs"
    quirky_ds_names = split_possible_string_list(quirky_ds_names)
    base_model_names = split_possible_string_list(base_model_names)

    gt_args, w2s_args = split_args_by_run_type(args)

    for quirky_ds_abbrev in quirky_ds_names:
        # first create a weak_labels_path
        weak_labels_path = create_weak_labels(
            ds_name=f"quirky_{quirky_ds_abbrev}",
            n_train_docs=n_train_docs,
            n_test_docs=n_test_docs,
            kwargs=args,
        )

        for base_model_name in base_model_names:
            model_id, _ = get_quirky_model_name(
                ds_name=quirky_ds_abbrev,
                base_model_id=base_model_name,
                templatization_method="random",
                standardize_templates=standardized_templates,
                full_finetuning=full_finetuning,
            )

            # then run gt and w2s runs on the provided model
            print(f"Running {model_id} on {quirky_ds_abbrev}...")
            try:
                cfg = TrainConfig(
                    ds_name=f"quirky_{quirky_ds_abbrev}",
                    model_size=model_id,
                    model_cfg_name=base_model_name,
                    n_train_docs=n_train_docs,
                    n_inference_docs=0,
                    n_test_docs=n_test_docs,
                    **gt_args,
                )
                train_simple_main(cfg)
            except Exception as e:
                print(
                    f"Failed to run ground truth {model_id} on {quirky_ds_abbrev}: {e}"
                )
                traceback.print_exc()

            try:
                # run weak-to-strong
                cfg = TrainConfig(
                    ds_name=f"quirky_{quirky_ds_abbrev}",
                    model_size=model_id,
                    model_cfg_name=base_model_name,
                    n_train_docs=n_train_docs,
                    n_inference_docs=0,
                    n_test_docs=n_test_docs,
                    weak_labels_path=weak_labels_path,
                    **w2s_args,
                )
                train_simple_main(cfg)
            except Exception as e:
                print(
                    f"Failed to run weak-to-strong {model_id} on {quirky_ds_abbrev}: {e}"
                )
                traceback.print_exc()

    print(f"Finished running models on {quirky_ds_names} x {base_model_names}")


if __name__ == "__main__":
    # see train_simple.py for valid args
    # fire.Fire(main)
    fire.Fire(pretrained_quirky)
