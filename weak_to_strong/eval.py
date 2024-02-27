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
                "logit": unpack(raw_logits),
                "logprob": logprobs,
            }
            results.extend([dict(zip(r, t)) for t in zip(*r.values())])

        accs, hard_labels, logprobs, soft_labels, probs = (
            np.array([r["acc"] for r in results]),
            np.array([r["supervisor_hard_label"] for r in results]),
            np.array([r["logprob"] for r in results])[:, 1],
            np.array([r["supervisor_soft_label"] for r in results])[:, 1],
            np.array([r["soft_pred"] for r in results])[:, 1],
        )

        metrics = {
            "accuracy": np.mean(accs),
            "accuracy_std_err": np.std(accs) / np.sqrt(len(accs)),
            "roc_auc": roc_auc_score(hard_labels, logprobs),
        }

        for metric in [confident_disagreement_rate, expected_overconfidence_error]:
            metrics.update(metric(probs, soft_labels))

        # if the current evaluation is weak to strong
        if "supervisor_hard_label" in ds.column_names:
            # these are the labels used to supervise the current model's *supervisor*
            sup_sup_hard_labels = np.array(ds["supervisor_hard_label"])
            # truncate away the last partial batch
            sup_sup_hard_labels = sup_sup_hard_labels[: len(accs)]
            metrics.update(CAR_given_incorrect(probs, soft_labels, sup_sup_hard_labels))

        if metric_prefix:
            metrics = {f"{metric_prefix}_{k}": v for k, v in metrics.items()}

        if verbose:
            for k, v in metrics.items():
                print(f"\t{k}: {v:.3f}")

        return datasets.Dataset.from_list(results), metrics


def confident_disagreement_rate(
    probs: np.ndarray, soft_labels: np.ndarray, conf_thresh: float = 0.95
) -> dict[str, float]:
    """
    This function calculates the confident disagreement rate (CDR) between the predicted
    probabilities and the soft labels: The proportion of confident predictions that disagree
    with confident soft labels.

    This metric aims to check whether a strong student learns to confidently ignore its supervision.
    It is simple but not the most discriminative. It works best when most confident predictions are
    near the confidence threshold and most confident soft labels are near 0 or 1, in which case the
    null hypothesis is that CDR = 1 - conf_thresh.

    Parameters:
    probs (np.ndarray): The predicted probabilities.
    soft_labels (np.ndarray): The soft labels.
    conf_thresh (float): The confidence threshold for the CDR.

    Returns:
    dict containing
    "CDR": The confident disagreement rate.
    "CDR_std_err": The standard error of the confident disagreement rate.
    """

    pred_yes, pred_no = (probs > conf_thresh), (probs < (1 - conf_thresh))
    lab_yes, lab_no = (soft_labels > conf_thresh), (soft_labels < (1 - conf_thresh))
    # we cast to singleton arrays to avoid division by zero errors
    num_confident_disagreements = np.array(
        (pred_yes & lab_no).sum() + (pred_no & lab_yes).sum()
    )
    num_confident_predictions = np.array((pred_yes | pred_no).sum())
    CDR = num_confident_disagreements / num_confident_predictions
    CDR_std_err = np.sqrt(CDR * (1 - CDR) / num_confident_predictions)

    return {"CDR": float(CDR), "CDR_std_err": float(CDR_std_err)}


def expected_overconfidence_error(
    probs: np.ndarray, soft_labels: np.ndarray
) -> dict[str, float]:
    """
    Computes a variant of Expected Calibration Error (ECE) that is sensitive to
    the average direction of calibration error. Rather than taking the absolute difference
    between predicted and label probabilities, it takes (predicted - label) for
    predictions >0.5 and (label - predicted) for predictions <=0.5.

    If EOE > 0, the model is overconfident. If EOE < 0, the model is underconfident.
    If the model is perfectly calibrated, EOE = 0, though miscalibrated models can
    also have EOE = 0 if they are over- and underconfident in equal measure.

    Parameters:
    probs (np.ndarray): The predicted probabilities.
    soft_labels (np.ndarray): The soft labels.

    Returns:
    dict containing
    "EOE": The expected overconfidence error.
    """

    # we don't need to make and average over bins because we're not
    # using the absolute value of the difference, making the
    # computation a lot simpler compared to ECE
    pointwise_overconfidence = np.where(
        probs > 0.5, probs - soft_labels, soft_labels - probs
    )
    return {"EOE": pointwise_overconfidence.mean()}


def CAR_given_incorrect(
    probs: np.ndarray, soft_labels: np.ndarray, gt_labels: np.ndarray, conf_thresh=0.55
) -> dict[str, float]:
    """
    Computes the Confident Agreement Rate (CAR) conditional on the student
    providing a ground-truth incorrect prediction.

    Parameters:
    probs (np.ndarray): The predicted probabilities.
    soft_labels (np.ndarray): The soft labels.
    gt_labels (np.ndarray): The ground truth hard labels.

    Returns:
    dict containing
    "CAR_given_incorrect": The confident agreement rate conditional on the student
    "CAR_given_incorrect_std_err": The standard error
    """

    # condition on being ground truth incorrect
    incorrect_mask = gt_labels != (probs > 0.5)
    probs = probs[incorrect_mask]
    soft_labels = soft_labels[incorrect_mask]

    pred_yes, pred_no = (probs > conf_thresh), (probs < (1 - conf_thresh))
    lab_yes, lab_no = (soft_labels > conf_thresh), (soft_labels < (1 - conf_thresh))
    num_incorrect_confident_agreements = np.array(
        (pred_yes & lab_yes).sum() + (pred_no & lab_no).sum()
    )
    num_incorrect_confident_predictions = np.array((pred_yes | pred_no).sum())
    CAR_given_incorrect = (
        num_incorrect_confident_agreements / num_incorrect_confident_predictions
    )
    CAR_given_incorrect_std_err = np.sqrt(
        CAR_given_incorrect
        * (1 - CAR_given_incorrect)
        / num_incorrect_confident_predictions
    )
    return {
        "CAR_given_incorrect": float(CAR_given_incorrect),
        "CAR_given_incorrect_std_err": float(CAR_given_incorrect_std_err),
    }
