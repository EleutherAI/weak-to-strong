import numpy as np
import os
from weak_to_strong.config import MODELS_DICT
from weak_to_strong.eval import eval_model_acc
from weak_to_strong.model import update_sharded_checkpoint
from train_simple import main as train_simple_main
import fire

from weak_to_strong.datasets import load_and_process_dataset


def main(
    coef_best: float,
    coef_final: float,
    model_size: str = "gpt2",
    ds_name: str = "sciq",
    w2s_eval_every: int = 1,
    seed: int = 0,
    n_train1_docs: int = 1000,
    n_train2_docs: int = 1000,
    n_test_docs: int = 1000,
    linear_probe: bool = False,
    verbose: bool = False,
    **kwargs
) -> float:
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
        ds_name: str
            Dataset to use.
        w2s_eval_every: int
            How often to evaluate the model.
        seed: int
            Random seed.
        n_train1_docs: int
            Number of training documents for the first model.
        n_train2_docs: int
            Number of training documents for the second model.
        n_test_docs: int
            Number of test documents.
        linear_probe: bool
            Whether to use a linear probe.
        verbose: bool
            Whether to print verbose output.
        **kwargs: dict
            Other arguments to pass to train_simple_main.
    Returns:
        Ground truth accuracy of the new model
    """
    save_path = train_simple_main(
        model_size=model_size,
        ds_name=ds_name,
        w2s_eval_every=w2s_eval_every,
        seed=seed,
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

    eval_ds = dataset["test"]  # type: ignore
    use_lm_head = "choice_input_ids" in eval_ds.features

    if verbose:
        print(f"Loading model {model_size}")
    model, _ = model_config.load_model(
        batch_size=eval_batch_size,
        use_lm_head=use_lm_head,
        linear_probe=linear_probe,
    )
    if verbose:
        print("Updating model weights")
        print(f"coef_best={coef_best}, best_path={best_path}")
    update_sharded_checkpoint(model, best_path, coef_best)
    if verbose:
        print(f"coef_final={coef_final}, final_path={final_path}")
    update_sharded_checkpoint(model, final_path, coef_final)

    test_results = eval_model_acc(model, eval_ds, eval_batch_size)
    return float(
        np.mean([r["acc"] for r in test_results])  # type: ignore
    )


if __name__ == "__main__":
    fire.Fire(main)
