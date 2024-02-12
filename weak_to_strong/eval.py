from typing import Tuple
import datasets
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from weak_to_strong.common import clear_mem, to_batch


def unpack(x):
    assert isinstance(x, torch.Tensor), type(x)
    return x.detach().float().cpu().numpy().tolist()


def extract_accuracy(results: datasets.Dataset) -> float:
    return np.mean([r["acc"] for r in results])  # type: ignore


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
            labels = batch["soft_label"]
            # run forward pass
            raw_logits = model(
                input_ids, choice_input_ids=batch.get("choice_input_ids")
            )

            raw_logprobs = torch.nn.functional.log_softmax(raw_logits, dim=-1)
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
                    )
                    for input_id, txt, label, pred, prob, logprob, logit in zip(
                        batch["input_ids"],
                        batch["txt"],
                        labels,
                        preds,
                        probs,
                        logprobs,
                        logits,
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


def eval_model_accuracy_loss(
    model: nn.Module,
    ds: datasets.Dataset,
    batch_size: int = 16,
    minibatch_size: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function evaluates the accuracy and CE loss of a given model on
    a given dataset.

    Parameters:
    model (nn.Module): The model to be evaluated.
    ds (datasets.Dataset): The dataset on which the model is to be evaluated.

    Returns:
    accuracy (float): The accuracy of the model on the given dataset.
    ce_loss (torch.Tensor): The cross-entropy loss of the model on the
        given dataset.
    """
    clear_mem()
    model.eval()
    io_device = model.device if hasattr(model, "device") else 0
    print("Evaluating model accuracy and loss")
    print(f"io_device={io_device}")
    print(f"batch_size={batch_size}")
    print(f"minibatch_size={minibatch_size}")

    total_loss = None
    total_accuracy = None
    n_batches = 0
    for start in range(0, len(ds), batch_size):
        for mbatch in to_batch(
            ds, minibatch_size, start=start, end=start + batch_size
        ):
            # pad input_ids to common length
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ex) for ex in mbatch["input_ids"]],
                batch_first=True
            ).to(device=io_device)
            # run forward pass
            raw_logits = model(
                input_ids, choice_input_ids=mbatch.get("choice_input_ids")
            )
            labels = mbatch["soft_label"]
            raw_labels = torch.tensor(labels).to(raw_logits.device)

            raw_logprobs = torch.nn.functional.log_softmax(raw_logits, dim=-1)
            batch_loss = torch.nn.functional.cross_entropy(
                raw_logits, raw_labels, reduction="mean"
            )
            if total_loss is None:
                total_loss = batch_loss
            else:
                total_loss += batch_loss
            with torch.inference_mode():
                # Compute accuracy without affecting gradients
                preds = torch.argmax(raw_logprobs, dim=-1)
                labels = torch.argmax(raw_labels, dim=-1)
                batch_acc = torch.mean((preds == labels).float())
                if total_accuracy is None:
                    total_accuracy = batch_acc
                else:
                    total_accuracy += batch_acc
            n_batches += 1
    assert total_loss is not None
    assert total_accuracy is not None
    accuracy = total_accuracy.clone() / n_batches
    ce_loss = total_loss / n_batches
    return accuracy, ce_loss
