from hashlib import md5

from datasets import Dataset
import numpy as np
from tqdm.auto import tqdm
import torch
from scipy.stats import linregress
import warnings

from weak_to_strong.loss import LossFnBase
from weak_to_strong.common import to_batch


class Sigmoid(LossFnBase):
    def __call__(self, logits, labels, step_frac=0):
        assert logits.shape == torch.Size([1, 2])
        return torch.nn.functional.sigmoid(logits[0, 1] - logits[0, 0])


class Diff(LossFnBase):
    def __call__(self, logits, labels, step_frac=0):
        assert logits.shape == torch.Size([1, 2])
        return logits[0, 1] - logits[0, 0]


def get_jacobians(
    model: torch.nn.Module,
    dataset: Dataset,
    postprocess_logits_fn: LossFnBase = Sigmoid(),
    target_label_column: str = "soft_label",
    d_down: int = 10_000,
    step_frac: float = 0,
    io_device: str | int = "cuda",
):
    """
    Get the gradients of `postprocess_logits_fn(model(input))` with respect to
    the model parameters, projected onto a random basis, for each input in `dataset`.


    Args:
    model: The model to get the gradients of.
    dataset: The dataset to get the gradients for.
    postprocess_logits_fn: The scalar quantity to compute the gradients of, in terms of
        the model's logits, labels, and step_frac.
    target_label_column: The column in `dataset` that contains the desired labels.
    d_proj: The dimension of the random basis to project the gradients onto.
    step_frac: The current fraction of the total training steps.
    """
    n_eval = len(dataset)

    model_n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    generator = torch.Generator().manual_seed(0)
    proj_basis_indices = torch.randint(
        0, model_n_params, (d_down,), generator=generator
    )
    proj_basis_indices, _ = proj_basis_indices.sort()

    hash_proj_indices = md5(proj_basis_indices.numpy().tobytes()).hexdigest()
    print(f"Hash(projection indices): {hash_proj_indices}")

    proj_grads = -torch.ones((n_eval, d_down), device=io_device)
    fs = -torch.ones((n_eval,), device=io_device)

    model.eval()

    # unfortunately we have to use a batch size of 1 to get examplewise grads
    # because only 1 backwards pass is allowed per forward pass
    for i, batch in tqdm(
        enumerate(to_batch(dataset.select(range(n_eval)), batch_size=1)),
        desc="Computing jacobians",
        total=n_eval,
    ):
        label = torch.tensor(batch[target_label_column]).to(io_device)
        input_ids = (
            torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ids) for ids in batch["input_ids"]]  # type: ignore
            )
            .transpose(0, 1)
            .to(io_device)  # type: ignore
        )
        logits = model(
            input_ids=input_ids, choice_input_ids=batch.get("choice_input_ids")
        )

        f = postprocess_logits_fn(logits, label, step_frac=step_frac)
        f.backward()

        proj_grads[i] = gather_grad_components(
            model, proj_basis_indices, io_device=io_device
        )
        fs[i] = f.item()

        # zero out grads
        model.zero_grad()

    return proj_grads, fs, proj_basis_indices, model_n_params


def gather_grad_components(
    model, proj_basis_indices, io_device: str | int = "cuda", optimizer=None
):
    """
    This avoids concatenating all the grads
    into one tensor before projecting, to save memory.
    This assumes `model` parameters has gradients already computed.

    If optimizer passed is Adam, then we also normalize the gradients by the
    second moment estimate per Adam's update rule.
    """
    proj_updates = torch.zeros((len(proj_basis_indices),), device=io_device)
    param_iter = iter(p for p in model.parameters() if p.grad is not None)
    param = next(param_iter)
    pg = param.grad

    start_i = 0  # index into grad of the first component of pg
    for proj_i, grad_idxr in enumerate(
        proj_basis_indices
    ):  # iterate over sorted projection indices
        while (
            start_i + pg.numel() <= grad_idxr
        ):  # while the current param is earlier than the desired index
            start_i += pg.numel()
            param = next(param_iter)
            pg = param.grad

        update = pg.flatten()[grad_idxr - start_i].to(io_device)
        if isinstance(optimizer, torch.optim.Adam):
            step = optimizer.state[param].get("step", 0)
            eps = float(optimizer.param_groups[0]["eps"])
            if step > 0:
                # normalize based on raw second moment estimates
                beta2 = float(optimizer.param_groups[0]["betas"][1])
                exp_avg_sq = optimizer.state[param]["exp_avg_sq"]
                exp_avg_sq = exp_avg_sq.flatten()[grad_idxr - start_i].to(io_device)
                corrected_exp_avg = torch.sqrt(exp_avg_sq / (1 - beta2**step))
            else:
                corrected_exp_avg = update.abs()

            update = update / (corrected_exp_avg + eps)

        proj_updates[proj_i] = update

    return proj_updates


def check_tailedness(sample: torch.Tensor, verbose=False, warning_p_val=0.05):
    """
    We can estimate how good our stderr estimate is by computing variance
    with increasing sample sizes. Samples from heavy-tailed distributions
    will usually larger variances with increasing sample size
    """
    assert sample.ndim == 1
    sample = sample.detach().cpu().numpy()
    n = len(sample)
    ns = []
    vars = []
    while n > 1:
        subsample = np.random.choice(sample, size=n, replace=True)
        var = subsample.var()
        ns.append(n)
        vars.append(var)
        n //= 2

    ns, vars = np.array(ns)[::-1], np.array(vars)[::-1]
    linreg = linregress(np.log(ns), vars)

    if verbose:
        print(f"regression of variance estimate onto log(sample size): {linreg}")
        print("\t".join(f"{n},{var:.1E}" for n, var in zip(ns, vars)))

    if linreg.pvalue < warning_p_val:  # type: ignore
        warnings.warn(f"Sample variance is increasing with sample size: {linreg}")
