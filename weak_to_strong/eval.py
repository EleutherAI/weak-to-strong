import datasets
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from weak_to_strong.common import to_batch


def unpack(x):
    assert isinstance(x, torch.Tensor), type(x)
    return x.detach().float().cpu().numpy().tolist()


def eval_model_acc(
    model: nn.Module, ds: datasets.Dataset, eval_batch_size: int = 16
) -> datasets.Dataset:
    """
    This function evaluates the accuracy of a given model on a given dataset.

    Parameters:
    model (nn.Module): The model to be evaluated.
    ds (datasets.Dataset): The dataset on which the model is to be evaluated.

    Returns:
    results (list): A list of dictionaries containing the input_ids, ground truth label,
                    predicted label, accuracy of prediction, logits and soft label for
                    each example in the dataset.
    """

    model.eval()

    with torch.no_grad():
        results = []
        # for ex in ds:
        for batch in to_batch(ds, eval_batch_size):
            # pad input_ids to common length
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ex) for ex in batch["input_ids"]], batch_first=True
            ).to(model.device if hasattr(model, "device") else "cpu")
            soft_labels = batch["soft_label"]

            # run forward pass
            raw_logits = model(
                input_ids, choice_input_ids=batch.get("choice_input_ids")
            )

            raw_logprobs = torch.nn.functional.log_softmax(raw_logits, dim=-1)
            hard_labels = np.argmax(soft_labels, axis=-1)
            logprobs = unpack(raw_logprobs)
            preds = np.argmax(logprobs, axis=-1)
            r = {
                "txt": batch["txt"],
                "input_ids": batch["input_ids"],
                "supervisor_hard_label": hard_labels,
                "supervisor_soft_label": soft_labels,
                "hard_pred": preds,
                "soft_pred": unpack(raw_logprobs.exp()),
                "acc": preds == hard_labels,
                "logits": unpack(raw_logits),
                "logprobs": logprobs,
            }
            results.extend([dict(zip(r, t)) for t in zip(*r.values())])
        accs = [r["acc"] for r in results]
        print(
            "Accuracy against ground truth:",
            np.mean(accs),
            "+/-",
            np.std(accs) / np.sqrt(len(accs)),
        )
        gt, logprob = (
            np.array([r["gt_label"] for r in results]),
            np.array([r["logprob"] for r in results])[:, 1],
        )
        print("AUC against ground truth:", roc_auc_score(gt, logprob))

        return datasets.Dataset.from_list(results)
