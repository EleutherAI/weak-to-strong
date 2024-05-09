from datasets import load_from_disk
from dataclasses import dataclass, field
import warnings
from typing import Optional
import os
import json

import torch
import numpy as np


@dataclass(frozen=True)
class SplitResults:
    # this could just be a dataset but I want type safety about the columns
    n: int
    ids: np.ndarray  # (n,)
    weak_soft_labels: np.ndarray  # (n,)
    gt_soft_labels: np.ndarray  # (n,)
    gt_hard_labels: np.ndarray  # (n,)
    logodds: np.ndarray  # (n,)
    pre_hiddens: Optional[np.ndarray] = None  # (n, n_layers, d)
    post_hiddens: Optional[np.ndarray] = None  # (n, n_layers, d)

    def __post_init__(self):
        assert len(self.ids) == self.n
        assert len(self.weak_soft_labels) == self.n
        assert len(self.gt_soft_labels) == self.n
        assert len(self.gt_hard_labels) == self.n
        assert self.logodds.shape == (self.n,)
        assert self.pre_hiddens is None or self.pre_hiddens.shape[0] == self.n
        assert self.post_hiddens is None or self.post_hiddens.shape[0] == self.n

    def truncate_to(self, n: int):
        new_pre_hiddens = self.pre_hiddens[:n] if self.pre_hiddens is not None else None
        new_post_hiddens = (
            self.post_hiddens[:n] if self.post_hiddens is not None else None
        )
        return SplitResults(
            n=n,
            ids=self.ids[:n],
            weak_soft_labels=self.weak_soft_labels[:n],
            gt_soft_labels=self.gt_soft_labels[:n],
            gt_hard_labels=self.gt_hard_labels[:n],
            logodds=self.logodds[:n],
            pre_hiddens=new_pre_hiddens,
            post_hiddens=new_post_hiddens,
        )


@dataclass
class TrainResults:
    n_steps: int
    ids: np.ndarray  # (n,)
    steps: np.ndarray  # (n,)
    weak_soft_labels: np.ndarray  # (n,)
    gt_soft_labels: np.ndarray  # (n,)
    pred_logodds: np.ndarray  # (n,)
    midtrain_hiddens: np.ndarray  # (n, n_layers, d)
    texts: np.ndarray  # (n,)
    weak_error: np.ndarray = field(
        init=False
    )  # (n,)  # weak_soft_labels - gt_soft_labels

    @property
    def n(self):
        return len(self.ids)

    def __post_init__(self):
        self.weak_error = self.weak_soft_labels - self.gt_soft_labels
        assert self.ids.shape == (self.n,)
        assert self.steps.shape == (self.n,)
        assert self.weak_soft_labels.shape == (self.n,)
        assert self.gt_soft_labels.shape == (self.n,)
        assert self.weak_error.shape == (self.n,)
        assert self.texts.shape == (self.n,)
        assert self.pred_logodds.shape == (self.n,)
        assert self.midtrain_hiddens.shape[0] == self.n

    def truncate_test_to(self, n: int):
        return TrainResults(
            n_steps=self.n_steps,
            ids=self.ids,
            steps=self.steps,
            weak_soft_labels=self.weak_soft_labels,
            gt_soft_labels=self.gt_soft_labels,
            pred_logodds=self.pred_logodds,
            midtrain_hiddens=self.midtrain_hiddens,
            texts=self.texts,
        )


@dataclass
class RunResult:
    weak_acc: float
    strong_acc: float
    w2s_acc: float
    pgr: float
    run: TrainResults  # with n=n_train_steps
    w2s_train: SplitResults
    w2s_vals: dict[int, SplitResults]  # step -> SplitResults
    w2s_test: SplitResults
    weak_test: SplitResults
    strong_test: SplitResults
    cfg: dict = field(repr=False)

    def __post_init__(self):
        if not (self.w2s_test.n == self.weak_test.n == self.strong_test.n):
            # sometimes these are different sizes when the runs used different batch sizes
            min_n = min(self.w2s_test.n, self.weak_test.n, self.strong_test.n)

            # truncate to the minimum length
            self.w2s_test = self.w2s_test.truncate_to(min_n)
            self.weak_test = self.weak_test.truncate_to(min_n)
            self.strong_test = self.strong_test.truncate_to(min_n)

        v0 = self.w2s_vals.get(0)
        if v0 is not None:
            assert all(np.all(v.ids == v0.ids) for v in self.w2s_vals.values())
            assert all(
                np.all(v.weak_soft_labels == v0.weak_soft_labels)
                for v in self.w2s_vals.values()
            )
            assert all(
                np.all(v.gt_soft_labels == v0.gt_soft_labels)
                for v in self.w2s_vals.values()
            )
            assert all(
                np.all(v.gt_hard_labels == v0.gt_hard_labels)
                for v in self.w2s_vals.values()
            )

        if not np.all(self.w2s_test.ids == self.weak_test.ids):
            warnings.warn("IDs in w2s_test and weak_test do not match")
        if not np.all(self.w2s_test.ids == self.strong_test.ids):
            warnings.warn("IDs in w2s_test and strong_test do not match")
        if not np.all(self.w2s_test.gt_soft_labels == self.weak_test.gt_soft_labels):
            warnings.warn("GT soft labels in w2s_test and weak_test do not match")
        if not np.all(self.w2s_test.gt_soft_labels == self.strong_test.gt_soft_labels):
            warnings.warn("GT soft labels in w2s_test and strong_test do not match")
        if not np.all(self.w2s_test.gt_hard_labels == self.weak_test.gt_hard_labels):
            warnings.warn("GT hard labels in w2s_test and weak_test do not match")
        if not np.all(self.w2s_test.gt_hard_labels == self.strong_test.gt_hard_labels):
            warnings.warn("GT hard labels in w2s_test and strong_test do not match")

    def select(self, min_step: int = 0, max_step: int = -1) -> "RunResult":
        """
        Returns a new RunResult with only the data from the specified steps,
        min is inclusive, max is exclusive
        """
        if max_step == -1:
            max_step = self.run.n_steps
        assert min_step < max_step
        assert min_step >= 0
        assert max_step <= self.run.n_steps

        step_mask = (self.run.steps >= min_step) & (self.run.steps < max_step)
        return RunResult(
            weak_acc=self.weak_acc,
            strong_acc=self.strong_acc,
            w2s_acc=self.w2s_acc,
            pgr=self.pgr,
            run=TrainResults(
                n_steps=max_step - min_step,
                ids=self.run.ids[step_mask],
                steps=self.run.steps[step_mask],
                weak_soft_labels=self.run.weak_soft_labels[step_mask],
                gt_soft_labels=self.run.gt_soft_labels[step_mask],
                pred_logodds=self.run.pred_logodds[step_mask],
                midtrain_hiddens=self.run.midtrain_hiddens[step_mask],
                texts=self.run.texts[step_mask],
            ),
            w2s_train=self.w2s_train,
            w2s_vals={
                k: v for k, v in self.w2s_vals.items() if k >= min_step and k < max_step
            },
            w2s_test=self.w2s_test,
            weak_test=self.weak_test,
            strong_test=self.strong_test,
            cfg=self.cfg,
        )


def find_run_paths(root: str, ds_name=None, strong_model=None):
    """
    root: The parent folder of the three runs you'd like to load.
        Usually should be set to `os.path.join(results_folder, sweep_subfolder)`
    ds_name: If not None, only consider runs with this dataset name.

    Returns: Length-3 dictionary containing the paths to the weak, strong, and w2s runs.
    """

    # find candidate subfolders: folders directly containing config.json
    candidates = [
        os.path.join(root, p)
        for p in os.listdir(root)
        if os.path.exists(os.path.join(root, p, "config.json"))
    ]
    configs = [json.load(open(os.path.join(p, "config.json"))) for p in candidates]
    if ds_name is not None:
        candidates = [
            p
            for p, c in zip(candidates, configs)
            if "ds_name" not in c or c["ds_name"] == ds_name
        ]
    if strong_model is not None:
        # keep runs with weak_labels or with the strong model
        candidates = [
            p
            for p, c in zip(candidates, configs)
            if c.get("model_size") == strong_model
            or os.path.exists(os.path.join(p, "weak_labels"))
        ]
    assert (
        len(candidates) == 3
    ), f"Expected 3 runs in {root} consistent with search, found {len(candidates)}"

    paths = dict()

    # weak: the one with weak_labels subdirectory
    weak_paths = [
        p for p in candidates if os.path.exists(os.path.join(p, "weak_labels"))
    ]
    assert len(weak_paths) == 1, f"Expected 1 weak run, found {len(weak_paths)}"
    paths["weak_path"] = weak_paths.pop()

    # strong: the one without weak_labels and with wlp=None
    strong_paths = [
        p
        for p in candidates
        if not os.path.exists(os.path.join(p, "weak_labels")) and "wlp=None" in p
    ]
    assert len(strong_paths) == 1, f"Expected 1 strong run, found {len(strong_paths)}"
    paths["strong_path"] = strong_paths.pop()

    # w2s: the one without wlp=None and without weak_labels
    w2s_paths = [
        p
        for p in candidates
        if "wlp=None" not in p and not os.path.exists(os.path.join(p, "weak_labels"))
    ]
    assert len(w2s_paths) == 1, f"Expected 1 w2s run, found {len(w2s_paths)}"
    paths["w2s_path"] = w2s_paths.pop()

    return paths


def load_run_result(w2s_path: str, weak_path: str, strong_path: str) -> RunResult:
    ### Load config ###
    w2s_cfg = json.load(open(os.path.join(w2s_path, "config.json")))

    ### Gather validation results from throughout training ###
    val_results_paths = [
        p
        for p in os.listdir(w2s_path)
        if p.startswith("eval_results") and p[-1].isdigit()
    ]

    w2s_val_results = dict()
    for path in val_results_paths:
        step = int(path.split("_")[-1])
        val_ds = load_from_disk(os.path.join(w2s_path, path)).with_format("numpy")  # type: ignore

        # NOTE: for a short period, intermediate evals have a different perspective on label names
        if "weak_soft_label" not in val_ds.column_names:
            warnings.warn(
                "It appears these w2s val results were produced with a "
                'version of the code that called weak_soft_label "soft_label"'
            )
            val_ds = val_ds.rename_columns(
                {
                    "soft_label": "weak_soft_label",
                    "hard_label": "weak_hard_label",
                }
            )

        w2s_val_results[step] = SplitResults(
            n=len(val_ds),
            ids=val_ds["id"],  # type: ignore
            weak_soft_labels=val_ds["weak_soft_label"][:, 1],  # type: ignore
            gt_soft_labels=val_ds["soft_label"][:, 1]  # type: ignore
            if "soft_label" in val_ds.column_names
            else np.full(len(val_ds), np.inf),
            gt_hard_labels=val_ds["hard_label"]  # type: ignore
            if "hard_label" in val_ds.column_names
            else np.full(len(val_ds), np.inf),
            logodds=val_ds["logit"][:, 1] - val_ds["logit"][:, 0],  # type: ignore
        )

    ### Load test results ###
    test_results = dict()
    for name, path in zip(
        ["weak", "strong", "w2s"], [weak_path, strong_path, w2s_path]
    ):
        test_ds = load_from_disk(os.path.join(path, "eval_results_final")).with_format(
            "numpy"
        )  # type: ignore
        try:
            logodds = test_ds["logit"][:, 1] - test_ds["logit"][:, 0]  # type: ignore
        except KeyError:
            warnings.warn(
                f'Could not find logit columns in {path}, inferring from "soft_pred" instead'
            )
            p = test_ds["soft_pred"][:, 1]  # type: ignore
            logodds = np.log(p / ((1 - p) + 1e-12) + 1e-12)
        try:
            weak_soft_labels = test_ds["weak_soft_label"][:, 1]  # type: ignore
        except KeyError:
            weak_soft_labels = np.full(len(test_ds), np.inf)  # type: ignore
        test_results[name] = SplitResults(
            n=len(test_ds),
            ids=test_ds["id"],  # type: ignore
            weak_soft_labels=weak_soft_labels,
            gt_soft_labels=test_ds["soft_label"][:, 1],  # type: ignore
            gt_hard_labels=test_ds["hard_label"],  # type: ignore
            logodds=logodds,
        )

    ### Load training results ###
    # note that in this case "soft_label" means weak_soft_label,
    # and there is a "gt_soft_label" column
    train_ds = load_from_disk(os.path.join(w2s_path, "train_ds")).with_format("numpy")
    # NOTE: the training set is truncated to the nearest multiple of the batch size
    train_ds = train_ds.select(  # type: ignore
        range(0, len(train_ds) - len(train_ds) % w2s_cfg["batch_size"])
    )
    try:
        pretrain_hiddens = (
            torch.load(
                os.path.join(w2s_path, "pre_train_hiddens.pt"), map_location="cpu"
            )
            .float()
            .numpy()[: len(train_ds)]
        )
    except FileNotFoundError:
        pretrain_hiddens = None
    try:
        posttrain_hiddens = (
            torch.load(
                os.path.join(w2s_path, "post_train_hiddens.pt"), map_location="cpu"
            )
            .float()
            .numpy()[: len(train_ds)]
        )
    except FileNotFoundError:
        posttrain_hiddens = None
    w2s_train_ds = SplitResults(
        n=len(train_ds),
        ids=train_ds["id"],  # type: ignore
        weak_soft_labels=train_ds["soft_label"][:, 1],  # type: ignore
        gt_soft_labels=train_ds["gt_soft_label"][:, 1],  # type: ignore
        gt_hard_labels=train_ds["gt_hard_label"],  # type: ignore
        logodds=np.full(len(train_ds), np.inf),  # train_ds doesn't contain predictions
        pre_hiddens=pretrain_hiddens,
        post_hiddens=posttrain_hiddens,
    )

    # grab train hiddens and logodds
    train_logodds_dict = torch.load(
        os.path.join(w2s_path, "train_logodds.pt"), map_location="cpu"
    )
    train_logodds, lo_ids = (
        train_logodds_dict["logodds"].float().numpy(),
        train_logodds_dict["ids"],
    )
    n_train_seen = len(train_logodds)
    try:
        train_hiddens = (
            torch.load(os.path.join(w2s_path, "train_hiddens.pt"), map_location="cpu")
            .float()
            .numpy()
        )
    except FileNotFoundError:
        train_hiddens = np.full((n_train_seen, 1, 1), np.inf)

    # join the weak_soft_labels, texts, and gt_soft_labels from the training set
    n_epochs = n_train_seen // len(w2s_train_ds.ids)
    run_dict = {
        "ids": np.tile(w2s_train_ds.ids, n_epochs + 1)[:n_train_seen],
        "weak_soft_labels": np.tile(w2s_train_ds.weak_soft_labels, n_epochs + 1)[
            :n_train_seen
        ],
        "gt_soft_labels": np.tile(w2s_train_ds.gt_soft_labels, n_epochs + 1)[
            :n_train_seen
        ],
        "texts": np.tile(train_ds["txt"], n_epochs + 1)[:n_train_seen],
        "steps": np.arange(n_train_seen) // w2s_cfg["batch_size"],
    }
    # ensure the labels are assigned to the corresponding rows
    assert np.all(lo_ids == run_dict["ids"])

    grad_results = TrainResults(
        n_steps=n_train_seen // w2s_cfg["batch_size"],
        pred_logodds=train_logodds,
        midtrain_hiddens=train_hiddens,
        **run_dict,
    )

    ### Compute accuracy metrics ###
    accs = {
        name: np.mean(
            test_results[name].gt_hard_labels == (test_results[name].logodds > 0)
        )
        for name in ["weak", "strong", "w2s"]
    }
    pgr = (accs["w2s"] - accs["weak"]) / (accs["strong"] - accs["weak"])

    return RunResult(
        weak_acc=accs["weak"],
        strong_acc=accs["strong"],
        w2s_acc=accs["w2s"],
        pgr=pgr,
        run=grad_results,
        w2s_train=w2s_train_ds,
        w2s_vals=w2s_val_results,
        w2s_test=test_results["w2s"],
        weak_test=test_results["weak"],
        strong_test=test_results["strong"],
        cfg=w2s_cfg,
    )
