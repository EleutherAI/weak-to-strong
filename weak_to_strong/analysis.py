# run result, algos, and plotting

from datasets import Dataset, load_from_disk
from dataclasses import dataclass, field
import warnings
from collections import defaultdict
import os
import json

import torch
import numpy as np

from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
import sklearn

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# load models and ds

import sys

from weak_to_strong.common import get_tokenizer, clear_mem, get_gpu_mem_used
from weak_to_strong.model import TransformerWithHead
from weak_to_strong.model_config import MODELS_DICT, ModelConfig
from weak_to_strong.eval import eval_loop, compute_metrics
from weak_to_strong.datasets import (
    VALID_DATASETS,
    tokenize_dataset,
    load_and_process_dataset,
)

# Classifier and 2 above

from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.nn.functional import (
    binary_cross_entropy_with_logits as bce_with_logits,
)
from torch.nn.functional import (
    cross_entropy,
)

# import weak_to_strong.runres_grads
# import weak_to_strong.runres_nograds

def load_run_result(grads=True, *args, **kwargs):
    if grads:
        return weak_to_strong.runres_grads.load_run_result(*args, **kwargs)
    else:
        return weak_to_strong.runres_nograds.load_run_result(*args, **kwargs)

def subgraph(knn, y):
    return knn[y][:, y]

def count_like_neighbors(knn, y):
    G = subgraph(knn, y)
    counts = np.bincount(G.nonzero()[0])
    allcounts = np.zeros(y.sum())
    allcounts[:counts.shape[0]] = counts
    return allcounts

def largest_component(knn):
    components = connected_components(knn)[1]
    top_component_idx = np.bincount(components).argmax()
    return components==top_component_idx

def union_bools(a, b):
    # composition of (bool vector) subsets
    # b subset of a
    # a subset of S
    # returns b as subset of S
    a_ = a.copy()
    a_[a_] = b 
    return a_

def topoCC(x, y, k, mutual=False):
    # returns indices of points in the largest connected component of their class
    # "G"
    if k > x.shape[0] - 1:
        k = x.shape[0] - 1
        
    knn = kneighbors_graph(x, k, mode='connectivity', include_self=False)
    if mutual:
        knn = knn.multiply(knn.T)
    G1 = subgraph(knn, y)
    G0 = subgraph(knn, ~y)
    
    # bools for top components "Qi" = "Si" within class subgraphs Gi
    topG1 = largest_component(G1)
    topG0 = largest_component(G0)
    
    # bools for Qi within G
    base1 = union_bools(y, topG1)
    base0 = union_bools(~y, topG0)
    # bool for "C" = "S" within G
    base = base1 + base0

    return base

def topofilter(x, y, k, m, mutual=(False, False)):
    # returns high scores for predicted correct labels, low scores for flipped labels
    n = x.shape[0]

    base = topoCC(x, y, k, mutual[0])
    # points and labels for S
    xS = x[base]
    yS = y[base]

    scoresS = zetafilter(xS, yS, m, mutual[1])
    scores = np.zeros(n)
    scores[base] = scoresS

    return scores

def zetafilter(x, y, m, mutual=False):
    n = x.shape[0]
    if m > n - 1:
        m = n - 1
        # print(f"[WARN] #neighbors {m} >= {n} #points, taking {n-1}")
        
    knn = kneighbors_graph(x, m, mode='connectivity', include_self=False)
    if mutual:
        knn = knn.multiply(knn.T)
    neighborsfilter1 = count_like_neighbors(knn, y) / m
    neighborsfilter0 = count_like_neighbors(knn, ~y) / m

    scores = np.zeros(n)
    scores[y] = neighborsfilter1
    scores[~y] = neighborsfilter0

    return scores

def knn_average(x, yc, k, train=None, val=None):
    if train is None:
        train = np.ones_like(yc, dtype=bool)
    if val is None:
        val = np.ones_like(yc, dtype=bool)

    knn = NearestNeighbors(n_neighbors=k).fit(x[train])
    # [n, k]: distances and indices of k nearest neighbors
    distances, indices = knn.kneighbors(x[val])
    # [n, k]: continuous labels of k nearest neighbors
    neighbor_labels = yc[train][indices]
    # [n,]: average label of k nearest neighbors
    pred = np.mean(neighbor_labels, axis=1)

    return pred

def knn_iter(x, yc, k, iters):
    for i in range(iters):
        yc = knn_average(x, yc, k)

    return yc

def knn_score(x, yc, k, iters=1):
    avgs = knn_iter(x, yc, k, iters)
    y = yc > 0.5

    scores = avgs
    scores[~y] = 1 - scores[~y]

    return scores

def local_outlier_factor(x, k):
    n = x.shape[0]
    knn = NearestNeighbors(n_neighbors=k).fit(x)
    # [n, k]: distances and indices of k nearest neighbors
    distances, indices = knn.kneighbors()
    # [n,]: k-distance of each point
    kdist = distances[:, -1]
    # [n, k]: k-distance of k nearest neighbors
    neighbor_kdist = kdist[indices]
    # [n, k]: reachability distance of each point FROM its k nearest neighbors
    rdist = np.maximum(neighbor_kdist, distances)
    # [n,]: local reachability density of each point
    lrd = 1 / np.mean(rdist, axis=1)
    # [n, k]: local reachability density of each point's k nearest neighbors
    neighbor_lrd = lrd[indices]
    # [n,]: local outlier factor of each point
    lof = np.mean(neighbor_lrd, axis=1) / lrd

    return lof

def lof_filter(x, y, k):
    n = x.shape[0]
    x1 = x[y]
    x0 = x[~y]
    lof1 = local_outlier_factor(x1, k)
    lof0 = local_outlier_factor(x0, k)
    scores = np.zeros(n)
    scores[y] = lof1
    scores[~y] = lof0

    return scores


def filter_confusion_matrix(correct, scores):
    # returns curves for confusion matrix of *correctness* classifier
    # correct: ndarray[bool]: which points have correct labels
    # scores: ndarray[float]: confidence score for correctness of labels

    aucs.append(roc_auc_score(correct, scores))
    # sort correct by scores
    correct_sort = correct[np.argsort(-scores)]
    scores_sort = scores[np.argsort(-scores)]
    # find points where scores change
    change = np.append(np.where(np.diff(scores_sort))[0], [scores.shape[0] - 1])
    # compute TP, FP, FN, TN at each change point

    TP = np.cumsum(correct_sort)[change]
    FP = np.cumsum(~correct_sort)[change]
    FN = np.sum(correct_sort) - TP
    TN = np.sum(~correct_sort) - FP
    
    return TP, FP, FN, TN

def filter_confusion_matrix_from_predictions(gt, preds):
    # as above, but takes ground truth and predicted *soft labels*
    # and converts to confidence-score format
    # soft labels must be on a scale with 0.5 corresponding to minimum confidence

    scores = np.maximum(preds, 1 - preds)
    correct = ((preds >= 0.5) == gt)

    return filter_confusion_matrix(correct, scores)

# TODO: deprecate and replace
def run_topo(
        x, y, correct, ks, ms=None, 
        algo='topo', mutual=None, relabel=False,
        iters=1):
        
    ks_success = []
    ms_success = []
    aucs = []
    fprs = []
    tprs = []
    counts = []
    
    if ms is None:
        ms = ks

    if algo == "zeta_relabel": # legacy
        algo = "zeta"
        relabel = True

    for k, m in zip(ks, ms):
        try:
            if algo == 'topo':
                scores = topofilter(x, y, k, m, mutual if mutual is not None else (False, False))
            elif algo == 'zeta':
                scores = zetafilter(x, y, k, mutual if mutual is not None else False)
            elif algo == 'knn_avg':
                scores = knn_score(x, y, k, iters)
            elif algo == 'lof':
                scores = -lof_filter(x, y, k)
            elif algo is None:
                scores = y
            else:
                raise ValueError(f"Unknown algorithm {algo}")
        except ValueError:
            continue

        correct2 = correct.copy()

        if relabel:
            flip = scores < 0.5
            scores[flip] = 1 - scores[flip]
            correct2[flip] = ~correct[flip]

        aucs.append(roc_auc_score(correct2, scores))
        # sort correct by scores
        correct_sort = correct2[np.argsort(-scores)]
        scores_sort = scores[np.argsort(-scores)]
        # find points where scores change
        change = np.append(np.where(np.diff(scores_sort))[0], [scores.shape[0] - 1])
        # compute TP, FP, FN, TN at each change point

        TP = np.cumsum(correct_sort)[change]
        FP = np.cumsum(~correct_sort)[change]
        FN = np.sum(correct_sort) - TP
        TN = np.sum(~correct_sort) - FP
        
        counts.append((TP, FP, FN, TN))
        # fpr, tpr, _ = roc_curve(correct, scores)

        ks_success.append(k)
        ms_success.append(m)

    return ks_success, ms_success, aucs, counts

# TODO: deprecate and replace
def plot_frac_prec(
        ks_success, aucs, counts, title, 
        ax=None,
        show=True,
        dots=False, 
        ylim=None, 
        color=None,
        dots_only=False,
        colorbar_max=None,
        dotcolor="red"):

    dots = dots or dots_only
    
    max_k = max(ks_success)

    if ax is None:
        _, ax = plt.subplots()

    for k, count in zip(ks_success, counts):
        TP, FP, FN, TN = count
        prec = TP / (TP + FP)
        kept = (TP + FP) / len(y)
        if not dots_only:
            plt.plot(kept, prec, color=(color or plt.cm.viridis(k / max_k)))
        if dots:
            plt.plot(kept[:1], prec[:1], ".", color=dotcolor)

    if not show:
        return ax
    else:
        plt.title(title)
        plt.xlabel("frac kept")
        plt.ylabel("precision")
        # colorbar for k, ranging from 1 to max_k
        colorbar_max = colorbar_max or max_k
        plt.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, colorbar_max)), label="k", ax=ax)
        # truncate y axis
        if ylim is not None:
            plt.ylim(*ylim)
        plt.show()

# TODO: deprecate and replace
def plot_roc(ks_success, aucs, counts, title):
    max_k = max(ks_success)

    fig, ax = plt.subplots()

    for k, auc, count in zip(ks_success, aucs, counts):
        TP, FP, FN, TN = count
        FPR = FP / (FP + TN)
        TPR = TP / (TP + FN)
        plt.plot(FPR, TPR, color=plt.cm.viridis(k / max_k))

    plt.title(title)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    # colorbar for k, ranging from 1 to max_k
    plt.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, max_k)), label="k", ax=ax)

    plt.ylim(0, 1)
    plt.show()


def get_model_and_dataset(model_size, ds_name, weak_labels_path):
    seed = 0
    # n_train1_docs = 10000
    # n_train2_docs = 10000
    n_test_docs = 500
    max_ctx = 512
    skip_inference = False
    model_cfg_name = None
    disable_lora = True
    linear_probe = False

    print(f"model_size: {model_size}, model_cfg_name: {model_cfg_name}")
    mcfg = MODELS_DICT[model_cfg_name or model_size].copy()
    if disable_lora:
        del mcfg["lora_modules"]
    if model_cfg_name is not None:
        mcfg["name"] = model_size
    model_config = ModelConfig(**mcfg)
    if model_config.model_parallel:
        print(f"Using model parallelism for {model_size}")

    print(f"MODEL CONFIG: {mcfg}")
    print(f"MODEL CONFIG NAME: {model_config.name}")

    print("DS NAME:", ds_name)
    # Load dataset
    try:
        dataset = load_and_process_dataset(
            ds_name,
            seed=seed,
            split_sizes=dict(test=n_test_docs),
        )
    except PermissionError as e:
        print(f"!sudo chmod a+w {e.filename}")
        raise e

    test_ds = dataset["test"]  # type: ignore

    # take the predictions from the weak model to be the labels
    train1_ds = load_from_disk(weak_labels_path)


    # Tokenize datasets
    tokenizer = get_tokenizer(model_config.name)
    train1_ds = tokenize_dataset(train1_ds, tokenizer, max_ctx)  # type: ignore
    test_ds = tokenize_dataset(test_ds, tokenizer, max_ctx)  # type: ignore
    if "for_lm_head" in ds_name:
        assert "choice_input_ids" in train1_ds.column_names
        assert "choice_input_ids" in test_ds.column_names

    # Load model

    train_ds = train1_ds

    use_lm_head = "choice_input_ids" in train_ds.features


    model = TransformerWithHead.from_pretrained(
                model_config.name,
                lora_modules=model_config.lora_modules,
                linear_probe=linear_probe,
                **model_config.custom_kwargs,
            ).to(
                "cuda"  # type: ignore
            )

    return model, train_ds, test_ds


@dataclass
class InlpResult:
    """Result of Iterative Nullspace Projection (NLP)."""

    losses: list[float] = field(default_factory=list)
    classifiers: list["Classifier"] = field(default_factory=list)


@dataclass
class RegularizationPath:
    """Result of cross-validation."""

    penalties: list[float]
    losses: list[float]

    @property
    def best_penalty(self) -> float:
        """Returns the best L2 regularization penalty."""
        return self.penalties[self.losses.index(self.best_loss)]

    @property
    def best_loss(self) -> float:
        """Returns the best loss."""
        return min(self.losses)


class Classifier(torch.nn.Module):
    """Linear classifier trained with supervised learning."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.linear = torch.nn.Linear(
            input_dim, num_classes if num_classes > 2 else 1, device=device, dtype=dtype
        )
        self.linear.bias.data.zero_()
        self.linear.weight.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x).squeeze(-1)

    @torch.enable_grad()
    def fit(
        self,
        x: Tensor,
        y: Tensor,
        *,
        l2_penalty: float = 0.001,
        max_iter: int = 10_000,
    ) -> float:
        """Fits the model to the input data using L-BFGS with L2 regularization.

        Args:
            x: Input tensor of shape (N, D), where N is the number of samples and D is
                the input dimension.
            y: Target tensor of shape (N,) for binary classification or (N, C) for
                multiclass classification, where C is the number of classes.
            l2_penalty: L2 regularization strength.
            max_iter: Maximum number of iterations for the L-BFGS optimizer.

        Returns:
            Final value of the loss function after optimization.
        """
        optimizer = torch.optim.LBFGS(
            self.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
        )

        num_classes = self.linear.out_features
        loss_fn = bce_with_logits if num_classes == 1 else cross_entropy
        loss = torch.inf
        y = y.to(
            torch.get_default_dtype() if num_classes == 1 else torch.long,
        )

        def closure():
            nonlocal loss
            optimizer.zero_grad()

            # Calculate the loss function
            logits = self(x).squeeze(-1)
            loss = loss_fn(logits, y)
            if l2_penalty:
                reg_loss = loss + l2_penalty * self.linear.weight.square().sum()
            else:
                reg_loss = loss

            reg_loss.backward()
            return float(reg_loss)

        optimizer.step(closure)
        return float(loss)

    @torch.no_grad()
    def fit_cv(
        self,
        x: Tensor,
        y: Tensor,
        *,
        k: int = 5,
        max_iter: int = 10_000,
        num_penalties: int = 10,
        seed: int = 42,
    ) -> RegularizationPath:
        """Fit using k-fold cross-validation to select the best L2 penalty.

        Args:
            x: Input tensor of shape (N, D), where N is the number of samples and D is
                the input dimension.
            y: Target tensor of shape (N,) for binary classification or (N, C) for
                multiclass classification, where C is the number of classes.
            k: Number of folds for k-fold cross-validation.
            max_iter: Maximum number of iterations for the L-BFGS optimizer.
            num_penalties: Number of L2 regularization penalties to try.
            seed: Random seed for the k-fold cross-validation.

        Returns:
            `RegularizationPath` containing the penalties tried and the validation loss
            achieved using that penalty, averaged across the folds.
        """
        num_samples = x.shape[0]
        if k < 3:
            raise ValueError("`k` must be at least 3")
        if k > num_samples:
            raise ValueError("`k` must be less than or equal to the number of samples")

        rng = torch.Generator(device=x.device)
        rng.manual_seed(seed)

        fold_size = num_samples // k
        indices = torch.randperm(num_samples, device=x.device, generator=rng)

        # Try a range of L2 penalties, including 0
        l2_penalties = [0.0] + torch.logspace(-4, 4, num_penalties).tolist()

        num_classes = self.linear.out_features
        loss_fn = bce_with_logits if num_classes == 1 else cross_entropy
        losses = x.new_zeros((k, num_penalties + 1))
        y = y.to(
            torch.get_default_dtype() if num_classes == 1 else torch.long,
        )

        for i in range(k):
            start, end = i * fold_size, (i + 1) * fold_size
            train_indices = torch.cat([indices[:start], indices[end:]])
            val_indices = indices[start:end]

            train_x, train_y = x[train_indices], y[train_indices]
            val_x, val_y = x[val_indices], y[val_indices]

            # Regularization path with warm-starting
            for j, l2_penalty in enumerate(l2_penalties):
                self.fit(train_x, train_y, l2_penalty=l2_penalty, max_iter=max_iter)

                logits = self(val_x).squeeze(-1)
                loss = loss_fn(logits, val_y)
                losses[i, j] = loss

        mean_losses = losses.mean(dim=0)
        best_idx = mean_losses.argmin()

        # Refit with the best penalty
        best_penalty = l2_penalties[best_idx]
        self.fit(x, y, l2_penalty=best_penalty, max_iter=max_iter)
        return RegularizationPath(l2_penalties, mean_losses.tolist())

    @classmethod
    def inlp(
        cls, x: Tensor, y: Tensor, max_iter: int | None = None, tol: float = 0.01
    ) -> InlpResult:
        """Iterative Nullspace Projection (INLP) <https://arxiv.org/abs/2004.07667>.

        Args:
            x: Input tensor of shape (N, D), where N is the number of samples and D is
                the input dimension.
            y: Target tensor of shape (N,) for binary classification or (N, C) for
                multiclass classification, where C is the number of classes.
            max_iter: Maximum number of iterations to run. If `None`, run for the full
                dimension of the input.
            tol: Tolerance for the loss function. The algorithm will stop when the loss
                is within `tol` of the entropy of the labels.

        Returns:
            `InlpResult` containing the classifiers and losses achieved at each
            iteration.
        """

        y.shape[-1] if y.ndim > 1 else 2
        d = x.shape[-1]
        loss = 0.0

        # Compute entropy of the labels
        p = y.float().mean()
        H = -p * torch.log(p) - (1 - p) * torch.log(1 - p)

        if max_iter is not None:
            d = min(d, max_iter)

        # Iterate until the loss is within epsilon of the entropy
        result = InlpResult()
        for _ in range(d):
            clf = cls(d, device=x.device, dtype=x.dtype)
            loss = clf.fit(x, y)
            result.classifiers.append(clf)
            result.losses.append(loss)

            if loss >= (1.0 - tol) * H:
                break

            # Project the data onto the nullspace of the classifier
            x = clf.nullspace_project(x)

        return result

    def nullspace_project(self, x: Tensor) -> Tensor:
        """Project the given data onto the nullspace of the classifier."""

        # https://en.wikipedia.org/wiki/Projection_(linear_algebra)
        A = self.linear.weight.data.T
        P = A @ torch.linalg.solve(A.mT @ A, A.mT)
        return x - x @ P

def make_hook_qwen(i, target, dtype=torch.float32):
    acts = target[i]
    def save_activation(module, input, output):
        acts.append(output[0].detach().to(dtype).cpu())  # output: tuple(hidden:Tensor, cache:WeirdCacheObject)
    return save_activation

def make_acts(model, train_ds, model_type="qwen", 
    dtype=torch.bfloat16, eval_batch_size=32):
    if model_type == "qwen":
        layers = model.lm.model.layers
        make_hook_fn = make_hook_qwen
    elif model_type == "neox":
        layers = model.lm.gpt_neox.layers
        make_hook_fn = make_hook_qwen # works for neox too
    else:
        raise ValueError(model_type)
    handles = []
    activations_by_layer = {i: [] for i in range(len(layers))}

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook_fn(i, activations_by_layer, dtype)))

    train_results, train_metrics = eval_loop(
        model,
        train_ds,
        eval_batch_size,
        metric_prefix="eval",
        remove_large_columns=False,
    )

    return activations_by_layer

def cat_last_acts(train_ds, activations_by_layer, eval_batch_size=32):
    seq_lens = [len(x) for x in train_ds['input_ids']]
    bs = eval_batch_size

    all_last_acts = []

    for layer in activations_by_layer:
        last_acts = []

        for i, act in enumerate(activations_by_layer[layer]):
            last_pos = [x - 1 for x in seq_lens[bs * i : bs * (i + 1)]]
            last_acts.append(act[[*range(bs)], last_pos, :])
            
        all_last_acts.append(torch.cat(last_acts))

    return all_last_acts
    

def load_things(w2s_path, weak_path, strong_path, all0_path):
    r = load_run_result(w2s_path, weak_path, strong_path)

    all0_last_acts = torch.load(all0_path)

    return r, all0_last_acts

def logreg_run(r, all0_last_acts, layer=11):
    x = all0_last_acts[layer].to(torch.float32).cuda()
    epochsize = x.shape[0]

    yc = r.grads.weak_soft_labels[:epochsize]
    yc_t = torch.tensor(yc).cuda()
    gt = r.grads.gt_soft_labels[:epochsize]
    gt_t = torch.tensor(gt).cuda()

    clf = Classifier(x.shape[1], num_classes=1, device='cuda')

    rpath = clf.fit_cv(x, yc_t)

    for p, l in zip(rpath.penalties, rpath.losses):
        print(p, l)

    preds = torch.sigmoid(clf(x))

    errs = torch.abs(preds - yc_t)

    corr_preds = (preds > 0.5) == gt_t
    acc = corr_preds.to(torch.float).mean().item()

    print(acc, errs.mean(), rpath.best_penalty)
    print(np.log10(rpath.best_penalty))

    plt.plot(np.log10(np.array(rpath.penalties)), rpath.losses)
    plt.xlabel("log10(penalty)")
    plt.ylabel("loss")
    plt.title(f"L{layer} 5-fold CV logistic regression: val loss")
    plt.show()

    accs = []
    meanerrs = []

    for penalty in rpath.penalties:
        clf.fit(x, yc_t, l2_penalty=penalty)

        preds = torch.sigmoid(clf(x))

        errs = torch.abs(preds - yc_t)
        meanerrs.append(torch.mean(errs).detach().item())

        corr_preds = (preds > 0.5) == gt_t

        accs.append(corr_preds.to(torch.float).mean().item())
        print(accs[-1], meanerrs[-1], penalty)


    plt.plot(rpath.penalties, accs)
    plt.xscale('log')
    plt.xlabel("(L2 penalty)")
    plt.ylabel("acc")
    plt.ylim(max(accs) - 0.1, max(accs) + 0.02 )
    plt.title(f"L{layer} Logistic regression acc on gt")
    plt.show()

    plt.plot(rpath.penalties, meanerrs)
    plt.xscale('log')
    plt.xlabel("(L2 penalty)")
    plt.ylabel("loss")
    plt.title(f"L{layer} Logistic regression loss on weak labels")
    plt.show()

    print(f"L{layer} best acc: {max(accs)}")

    w2s_odds = np.exp(r.grads.train_logodds[-epochsize:])
    w2s = 1 / (1 + 1/w2s_odds)

    wk_acc = np.mean((yc > 0.5) == gt)

    w2s_acc = np.mean((w2s > 0.5) == gt)

    if all(w2s > 0.999): # logodds are missing --> inf
        print(f"gt class balance: {w2s_acc}")
    else:
        print(f"w2s acc: {w2s_acc}")
    print(f"wk acc: {wk_acc}")
    print('===========================')
    print()