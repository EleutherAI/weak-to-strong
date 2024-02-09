import json
import os
from typing import Optional

import torch
import wandb
from weak_to_strong.common import get_tokenizer
from weak_to_strong.config import MODELS_DICT
from weak_to_strong.datasets import tokenize_dataset
from weak_to_strong.eval import eval_model_accuracy_loss
from train_simple import main as train_simple_main
import fire

from weak_to_strong.datasets import load_and_process_dataset


def main(
    coef_best: Optional[float | torch.Tensor] = None,
    coef_final: Optional[float | torch.Tensor] = None,
    model_size: str = "mistralai/Mistral-7B-v0.1",
    weak_model_size: str = "gpt2",
    ds_name: str = "sciq",
    w2s_eval_every: int = 1,
    seed: int = 0,
    max_ctx: int = 1024,
    n_train1_docs: int = 1000,
    n_train2_docs: int = 1000,
    n_test_docs: int = 1000,
    linear_probe: bool = False,
    store: bool = False,
    verbose: bool = False,
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the task vector arithmetic on ground truth labels.
    Uses arithmetic
        w_base +
        coef_best * (w_best - w_base) +
        coef_final * (w_final - w_base).
    Arguments:
        coef_best: float
            Coefficient for the best model.
        coef_final: float
            Coefficient for the final model.
        model_size: str
            Model size to use.
        weak_model_size: str
            Weak model size to use.
        ds_name: str
            Dataset to use.
        w2s_eval_every: int
            How often to evaluate the model.
        seed: int
            Random seed.
        max_ctx: int
            Maximum context length.
        n_train1_docs: int
            Number of training documents for the first model.
        n_train2_docs: int
            Number of training documents for the second model.
        n_test_docs: int
            Number of test documents.
        linear_probe: bool
            Whether to use a linear probe.
        store: bool
            Whether to store the results.
        verbose: bool
            Whether to print verbose output.
        **kwargs: dict
            Other arguments to pass to train_simple_main.
    Returns:
        Ground truth accuracy of the new model
    """
    if coef_best is None or coef_final is None:
        # Called from wandb.sweep
        config = {}
        config.update(kwargs)
        config.update({
            "coef_best": coef_best,
            "coef_final": coef_final,
            "model_size": model_size,
            "weak_model_size": weak_model_size,
            "ds_name": ds_name,
            "w2s_eval_every": w2s_eval_every,
            "seed": seed,
            "max_ctx": max_ctx,
            "n_train1_docs": n_train1_docs,
            "n_train2_docs": n_train2_docs,
            "n_test_docs": n_test_docs,
            "linear_probe": linear_probe,
            "store": store,
            "verbose": verbose,
        })
        wandb_name = (
            f"model_{model_size.split('/')[-1]}_"
            f"weak_{weak_model_size}_"
            f"ds_{ds_name}_"
            f"evaluate_task_vector"
        )
        wandb.init(
            config=config,
            group=kwargs.get("sweep_subfolder", "default"),
            job_type="task_vector",
            name=wandb_name,
            dir=kwargs.get("results_folder", "/tmp/results"),
            reinit=True,
        )
        coef_best = wandb.config.coef_best
        coef_final = wandb.config.coef_final
    assert coef_best is not None
    assert coef_final is not None
    coef_best_float = (
        coef_best.item()
        if isinstance(coef_best, torch.Tensor)
        else coef_best
    )
    coef_best_str = f"{coef_best_float:.1f}".replace(".", "_")
    coef_final_float = (
        coef_final.item()
        if isinstance(coef_final, torch.Tensor)
        else coef_final
    )
    coef_final_str = f"{coef_final_float:.1f}".replace(".", "_")
    # Train weak model on ground truth
    train_simple_main(
        model_size=weak_model_size,
        ds_name=ds_name,
        w2s_eval_every=w2s_eval_every,
        seed=seed,
        max_ctx=max_ctx,
        n_train1_docs=n_train1_docs,
        n_train2_docs=n_train2_docs,
        n_test_docs=n_test_docs,
        linear_probe=linear_probe,
        **kwargs,
    )
    # Train strong model on weak labels
    save_path = train_simple_main(
        model_size=model_size,
        weak_model_size=weak_model_size,
        ds_name=ds_name,
        w2s_eval_every=w2s_eval_every,
        seed=seed,
        max_ctx=max_ctx,
        n_train1_docs=n_train1_docs,
        n_train2_docs=n_train2_docs,
        n_test_docs=n_test_docs,
        linear_probe=linear_probe,
        **kwargs,
    )
    best_path = os.path.join(save_path, "best_model.bin")
    final_path = os.path.join(save_path, "final_model.bin")
    model_config = MODELS_DICT[model_size]
    eval_batch_size = model_config.eval_batch_size

    if verbose:
        print(f"Loading dataset {ds_name}")
    dataset = load_and_process_dataset(
        ds_name,
        seed=seed,
        split_sizes=dict(
            train=n_train1_docs + n_train2_docs,
            test=n_test_docs
        ),
    )
    test_ds = dataset["test"]  # type: ignore
    tokenizer = get_tokenizer(model_config.name)
    test_ds = tokenize_dataset(test_ds, tokenizer, max_ctx)  # type: ignore

    use_lm_head = "choice_input_ids" in test_ds.features

    if verbose:
        print(f"Loading model {model_size}")
    model, minibatch_size = model_config.load_model(
        batch_size=eval_batch_size,
        use_lm_head=use_lm_head,
        linear_probe=linear_probe,
    )
    model.requires_grad_(False)  # training complete
    if verbose:
        print("Updating model weights")
        print(f"coef_best={coef_best_float}, best_path={best_path}")
    model.update_state(best_path, coef_best)
    if verbose:
        print(f"coef_final={coef_final_float}, final_path={final_path}")
    model.update_state(final_path, coef_final)

    if verbose:
        print("Evaluating model")
    test_acc, test_loss = eval_model_accuracy_loss(
        model,
        test_ds,
        batch_size=eval_batch_size,
        minibatch_size=minibatch_size,
    )
    if verbose:
        print(f"Test accuracy: {test_acc.item()}")

    # Save results
    if store:
        result_path = os.path.join(save_path, "results_summary.json")
        with open(result_path, "r") as f:
            res_dict = json.load(f)
        key = f"task_vector_{coef_best_str}_{coef_final_str}"
        res_dict[key] = test_acc.item()
        with open(os.path.join(save_path, "results_summary.json"), "w") as f:
            json.dump(res_dict, f, indent=2)
    if wandb.config.get("coef_best") is not None:
        wandb.log({"task_vector/accuracy": test_acc.item()})
    return test_acc, test_loss


if __name__ == "__main__":
    fire.Fire(main)
