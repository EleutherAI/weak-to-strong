import datasets
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from weak_to_strong.common import to_batch


def unpack(x):
    assert isinstance(x, torch.Tensor), type(x)
    return x.detach().float().cpu().numpy().tolist()


def extract_accuracy(results: datasets.Dataset) -> float:
    return np.mean([r["acc"] for r in results])  # type: ignore


def extract_ce_loss(results: datasets.Dataset) -> torch.Tensor:
    print("extract_ce_loss", type(results[0]["loss"]))
    losses = torch.cat([r["loss"] for r in results])  # type: ignore
    return torch.mean(losses, dim=0)


def eval_model_acc(
    model: nn.Module, ds: datasets.Dataset, eval_batch_size: int = 16
) -> datasets.Dataset:
    """
    This function evaluates the accuracy of a given model on a given dataset.

    Parameters:
    model (nn.Module): The model to be evaluated.
    ds (datasets.Dataset): The dataset on which the model is to be evaluated.

    Returns:
    results (list): 
        A list of dictionaries containing the input_ids, ground truth label,
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
                [torch.tensor(ex) for ex in batch["input_ids"]],
                batch_first=True
            ).to(model.device if hasattr(model, "device") else "cpu")
            # run forward pass
            raw_logits = model(
                input_ids, choice_input_ids=batch.get("choice_input_ids")
            )
            labels = batch["soft_label"]
            raw_labels = torch.tensor(labels).to(raw_logits.device)

            raw_logprobs = torch.nn.functional.log_softmax(raw_logits, dim=-1)
            losses = torch.nn.functional.cross_entropy(
                raw_logits, raw_labels, reduction="none"
            )
            logprobs = unpack(raw_logprobs)
            probs = unpack(raw_logprobs.exp())
            logits = unpack(raw_logits)

            preds = np.argmax(logprobs, axis=-1)
            labels = np.argmax(labels, axis=-1)

            results.extend(
                [
                    dict(
                        txt=txt,
                        input_ids=input_id,
                        gt_label=label,
                        hard_label=pred,
                        acc=label == pred,
                        logits=logit,
                        soft_label=prob,
                        logprob=logprob,
                        loss=loss,
                    )
                    for input_id, txt, label, pred, prob, logprob, logit, loss in zip(
                        batch["input_ids"],
                        batch["txt"],
                        labels,
                        preds,
                        probs,
                        logprobs,
                        logits,
                        losses
                    )
                ]
            )
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
