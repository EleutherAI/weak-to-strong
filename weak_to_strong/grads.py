from hashlib import md5

from datasets import Dataset
from tqdm.auto import tqdm
import torch

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
    d_down: int = 10_000_000,
    step_frac: float = 0,
    io_device: str | int = "cpu",
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

    return proj_grads, fs, proj_basis_indices


def gather_grad_components(
    model, proj_indices, io_device: str | int = "cpu", optimizer=None
):
    """
    This avoids concatenating all the grads
    into one tensor before projecting, to save memory.
    This assumes `model` parameters has gradients already computed.

    If optimizer passed is Adam, then we also normalize the gradients by the
    second moment estimate per Adam's update rule.
    """
    proj_updates = []

    starts = (
        torch.tensor([p.numel() for p in model.parameters()])
        .cumsum(0)
        .to(proj_indices.device)
    )
    meta_indices = torch.searchsorted(proj_indices, starts, right=True)
    index_chunks = proj_indices.tensor_split(meta_indices.tolist())

    # Adjust indices to be relative to the current parameter
    for chunk, offset in zip(index_chunks[1:], starts[1:]):
        chunk -= offset

    for param, indices in zip(
        model.parameters(), index_chunks
    ):  # iterate over sorted projection indices
        if len(indices) == 0:
            continue

        update = param.grad.flatten()[indices].to(io_device)

        if isinstance(optimizer, torch.optim.Adam):
            step = optimizer.state[param].get("step", 0)
            if step > 0:
                # normalize based on raw second moment estimates
                beta2 = float(optimizer.param_groups[0]["betas"][1])
                exp_avg_sq = optimizer.state[param]["exp_avg_sq"]
                exp_avg_sq = exp_avg_sq.flatten()[indices].to(io_device)
                corrected_exp_avg = torch.sqrt(exp_avg_sq / (1 - beta2**step))
            else:
                corrected_exp_avg = update.abs()

            eps = float(optimizer.param_groups[0]["eps"])
            update = update / (corrected_exp_avg + eps)

        proj_updates.append(update)

    return torch.cat(proj_updates)
