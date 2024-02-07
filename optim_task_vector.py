import torch
import torch_optimizer as toptim
from evaluate_task_vector import main as evaluate_task_vector_main
import fire


def main(
    task_optim: str = "adam",
    task_lr: float = 1e-4,
    task_max_steps: int = 100,
    task_max_steps_wo_improvement: int = 5,
    task_log_every: int = 10,
    **kwargs
):
    # create params for pair of coefs
    trainable_params = torch.nn.Parameter(torch.zeros(2))
    # create optimizer
    if task_optim.lower() == "adam":
        optimizer = torch.optim.Adam(
            trainable_params, lr=task_lr, betas=(0.9, 0.95)
        )
    elif task_optim.lower() == "adafactor":
        optimizer = toptim.Adafactor(trainable_params, lr=task_lr)
    else:
        assert False, (
            f"invalid optimizer {task_optim}, must be adam or adafactor"
        )
    # train
    best_loss = torch.tensor([1.0])
    steps_wo_improvement = 0
    for step in range(task_max_steps):
        acc = evaluate_task_vector_main(
            coef_best=trainable_params[0],
            coef_final=trainable_params[1],
            **kwargs
        )
        loss = 1.0 - acc
        if step % task_log_every == 0:
            print(f"step: {step}, loss: {loss.item()}")
        if loss < best_loss:
            best_loss = loss
            steps_wo_improvement = 0
        else:
            steps_wo_improvement += 1
        if steps_wo_improvement >= task_max_steps_wo_improvement:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"best_loss: {best_loss.item()}")
    print(f"best_coefs: {trainable_params.detach().tolist()}")


if __name__ == "__main__":
    fire.Fire(main)
