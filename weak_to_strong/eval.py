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


def eval_loop(
    model: nn.Module,
    ds: datasets.Dataset,
    eval_batch_size: int = 16,
    verbose: bool = True,
    metric_prefix: Optional[str] = None,
    remove_large_columns: bool = False,
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
                "id": batch["id"],
                "txt": batch["txt"],
                "input_ids": batch["input_ids"],
                "hard_label": hard_labels,
                "soft_label": soft_labels,
                "hard_pred": preds,
                "soft_pred": unpack(raw_logprobs.exp()),
                "acc": preds == hard_labels,
                "logit": unpack(raw_logits),
                "logprob": logprobs,
            }
            if remove_large_columns:
                del r["input_ids"]
                del r["txt"]
            if "weak_soft_label" in batch:
                r["weak_soft_label"] = batch["weak_soft_label"]
            results.extend([dict(zip(r, t)) for t in zip(*r.values())])

        # compute metrics
        soft_labels, pred_probs = (
            np.array([r["soft_label"] for r in results])[:, 1],
            np.array([r["soft_pred"] for r in results])[:, 1],
        )

        # if the current evaluation is weak to strong
        if "weak_soft_label" in ds.column_names:
            # these are predictions from the weak supervisor on the eval set
            # we loaded in `train_simple.py`
            weak_soft_labels = np.array(ds["weak_soft_label"])[:, 1]
            # truncate away the last partial batch
            weak_soft_labels = weak_soft_labels[: len(pred_probs)]
        else:
            weak_soft_labels = None
        metrics = compute_metrics(
            gt_soft_labels=soft_labels,
            pred_probs=pred_probs,
            weak_soft_labels=weak_soft_labels,
            metric_prefix=metric_prefix,
        )

        if verbose:
            for k, v in metrics.items():
                print(f"\t{k}: {v:.3f}")

        return datasets.Dataset.from_list(results), metrics


def compute_metrics(
    gt_soft_labels: np.ndarray,
    pred_probs: np.ndarray,
    weak_soft_labels: Optional[np.ndarray] = None,
    metric_prefix: Optional[str] = None,
) -> dict[str, float]:
    np.seterr(divide="ignore", invalid="ignore")
    metrics = dict()
    gt_hard_labels = gt_soft_labels > 0.5
    if weak_soft_labels is not None:
        metrics.update(
            CAR_given_incorrect(pred_probs, weak_soft_labels, gt_hard_labels)
        )
        targets = [gt_soft_labels, weak_soft_labels]
    else:
        targets = [gt_soft_labels]

    # when evaluating w2s, compute metrics a second time against the weak supervision
    for target_soft_labels in targets:
        target_hard_labels = target_soft_labels > 0.5
        preds = pred_probs > 0.5

        accs = preds == target_hard_labels
        metrics_against_target = {
            "acc": float(accs.mean()),
            "acc_std_err": float(np.std(accs) / np.sqrt(len(accs))),
            "auroc": float(roc_auc_score_or_nan(target_hard_labels, pred_probs)),
        }

        for metric in [
            confident_disagreement_rate,
            expected_overconfidence_error,
            calibration_error,
        ]:
            metrics_against_target.update(
                metric(probs=pred_probs, soft_labels=target_soft_labels)
            )

        if target_soft_labels is weak_soft_labels:
            metrics_against_target = {
                f"{k}_against_weak": v for k, v in metrics_against_target.items()
            }
        metrics.update(metrics_against_target)

    if metric_prefix:
        metrics = {f"{metric_prefix}/{k}": v for k, v in metrics.items()}

    np.seterr(divide="warn", invalid="warn")
    return metrics


# ##############################################################################
# # Metrics
# ##############################################################################


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


def roc_auc_score_or_nan(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return np.nan


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
    return {"EOE": float(pointwise_overconfidence.mean())}


def CAR_given_incorrect(
    probs: np.ndarray, soft_labels: np.ndarray, gt_labels: np.ndarray, conf_thresh=0.55
) -> dict[str, float]:
    """
    Computes the Confident Agreement Rate (CAR) conditional on the student
    providing a ground-truth incorrect prediction. This is the number of confident
    student-supervisor agreements on incorrect predictions, divided by the number
    of confident incorrect predictions.

    This can help determine whether a model is answering incorrectly due
    to poor supervision or poor optimization.

    Parameters:
    probs (np.ndarray): The predicted probabilities.
    soft_labels (np.ndarray): The soft labels.
    gt_labels (np.ndarray): The ground truth hard labels.
    conf_thresh (float): The confidence threshold for the CAR.

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


def calibration_error(
    probs: np.ndarray, soft_labels: np.ndarray, p: int = 2
) -> dict[str, float]:
    """
    Taken from
    https://github.com/EleutherAI/elk/blob/84e99a36a5050881d85f1510a2486ce46ac1f942/elk/metrics/calibration.py#L14

    Monotonic Sweep Calibration Error for binary problems.

    This method estimates the True Calibration Error (TCE) by searching for the largest
    number of bins into which the data can be split that preserves the monotonicity
    of the predicted confidence -> empirical accuracy mapping. We use equal mass bins
    (quantiles) instead of equal width bins. Roelofs et al. (2020) show that this
    estimator has especially low bias in simulations where the TCE is analytically
    computable, and is hyperparameter-free (except for the type of norm used).

    Paper: "Mitigating Bias in Calibration Error Estimation" by Roelofs et al. (2020)
    Link: https://arxiv.org/abs/2012.08668

    Args:
        labels: The ground truth (soft) labels.
        probs: The predicted probabilities.
        p: The norm to use for the calibration error. Defaults to 2 (Euclidean).
    """

    # Convert to torch tensors
    labels = torch.as_tensor(soft_labels)
    pred_probs = torch.as_tensor(probs)

    n = len(pred_probs)
    if n < 2:
        raise ValueError("Not enough data to compute calibration error.")

    # Sort the predictions and labels
    pred_probs, indices = pred_probs.sort()
    labels = labels[indices].float()

    # Search for the largest number of bins which preserves monotonicity.
    # Based on Algorithm 1 in Roelofs et al. (2020).
    # Using a single bin is guaranteed to be monotonic, so we start there.
    b_star, accs_star = 1, labels.mean().unsqueeze(0)
    for b in range(2, n + 1):
        # Split into (nearly) equal mass bins
        freqs = torch.stack([h.mean() for h in labels.tensor_split(b)])

        # This binning is not strictly monotonic, let's break
        if not torch.all(freqs[1:] > freqs[:-1]):
            break

        elif not torch.all(freqs * (1 - freqs)):
            break

        # Save the current binning, it's monotonic and may be the best one
        else:
            accs_star = freqs
            b_star = b

    # Split into (nearly) equal mass bins. They won't be exactly equal, so we
    # still weight the bins by their size.
    conf_bins = pred_probs.tensor_split(b_star)
    w = pred_probs.new_tensor([len(c) / n for c in conf_bins])

    # See the definition of ECE_sweep in Equation 8 of Roelofs et al. (2020)
    mean_confs = torch.stack([c.mean() for c in conf_bins])
    ece = torch.sum(w * torch.abs(accs_star - mean_confs) ** p) ** (1 / p)

    return {"ECE": ece.item(), "ECE_num_bins": b_star}
