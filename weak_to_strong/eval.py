from typing import Optional
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
    model: nn.Module,
    ds: datasets.Dataset,
    eval_batch_size: int = 16,
    conf_thresh: float = 0.95,
    verbose: bool = True,
    metric_prefix: Optional[str] = None,
) -> tuple[datasets.Dataset, dict[str, float]]:
    """
    This function evaluates the accuracy of a given model on a given dataset.

    Parameters:
    model (nn.Module): The model to be evaluated.
    ds (datasets.Dataset): The dataset on which the model is to be evaluated.

    Returns:
    results (list): A list of dictionaries containing the input_ids, ground truth label,
                    predicted label, accuracy of prediction, logits and soft label for
                    each example in the dataset.
    metrics (dict): A dictionary containing summary metrics for logging (e.g. AUROC).
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

        accs, gts, logprobs, soft_labels, probs = (
            np.array([r["acc"] for r in results]),
            np.array([r["gt_label"] for r in results]),
            np.array([r["logprob"] for r in results])[:, 1],
            np.array([r["supervisor_soft_label"] for r in results])[:, 1],
            np.array([r["soft_pred"] for r in results])[:, 1],
        )

        # confident disagreement rate
        pred_yes, pred_no = (probs > conf_thresh), (probs < (1 - conf_thresh))
        label_yes, label_no = (soft_labels > conf_thresh), (
            soft_labels < (1 - conf_thresh)
        )
        num_confident_disagreements = (pred_yes & label_no).sum() + (
            pred_no & label_yes
        ).sum()
        num_confident_predictions = (pred_yes | pred_no).sum()
        CDR = num_confident_disagreements / num_confident_predictions
        CDR_std_err = np.sqrt(CDR * (1 - CDR) / num_confident_predictions)

        metrics = {
            "accuracy": np.mean(accs),
            "accuracy_std_err": np.std(accs) / np.sqrt(len(accs)),
            "roc_auc": roc_auc_score(gts, logprobs),
            "CDR": CDR,
            "CDR_std_err": CDR_std_err,
        }
        if metric_prefix:
            metrics = {f"{metric_prefix}_{k}": v for k, v in metrics.items()}

        if verbose:
            for k, v in metrics.items():
                print(f"\t{k}: {v:.3f}")

        return datasets.Dataset.from_list(results), metrics
