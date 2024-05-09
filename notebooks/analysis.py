from typing import Literal
import numpy as np
import torch

from concept_erasure import LeaceEraser


def erase_labels(
    x,
    soft_labels,
    label_erasure: Literal[
        "none", "leace", "mean-diff", "keep-negative", "keep-positive"
    ] = "none",
):
    mask = np.full_like(soft_labels, True, dtype=bool)
    if label_erasure == "leace":
        eraser = LeaceEraser.fit(x=torch.from_numpy(x), z=torch.from_numpy(soft_labels))
        erased = eraser(x=torch.tensor(x)).numpy()
    elif label_erasure == "mean-diff":
        pos_mean = np.mean(x * soft_labels[:, None], axis=0)
        neg_mean = np.mean(x * (1 - soft_labels)[:, None], axis=0)
        mean_diff = pos_mean - neg_mean
        mean_diff = mean_diff / np.linalg.norm(mean_diff)
        erased = x - x @ mean_diff[:, None] * mean_diff[None, :] / (
            mean_diff @ mean_diff
        )
    elif label_erasure == "keep-negative":
        mask = soft_labels < 0.5
        erased = x[mask]
    elif label_erasure == "keep-positive":
        mask = soft_labels > 0.5
        erased = x[mask]
    elif label_erasure == "none":
        erased = x
    else:
        raise ValueError(f"Unknown label erasure method {label_erasure}")
    return erased, mask
