import torch
import torch_optimizer as toptim
from evaluate_task_vector import main as evaluate_task_vector_main
import fire


class TaskVectorModule(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.coefs = torch.nn.Parameter(torch.zeros(2))
        self.kwargs = kwargs

    def forward(self):
        acc = evaluate_task_vector_main(
            coef_best=self.coefs[0],
            coef_final=self.coefs[1],
            **self.kwargs
        )
        loss = 1.0 - acc
        return loss


def main(
    task_optim: str = "adam",
    task_lr: float = 1e-4,
    task_max_steps: int = 100,
    task_max_steps_wo_improvement: int = 5,
    task_log_every: int = 10,
    **kwargs
):
    module = TaskVectorModule(**kwargs)
    trainable_params = list(module.parameters())
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
        loss = module()
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
    print(f"best_coefs: {module.coefs.detach().tolist()}")


if __name__ == "__main__":
    fire.Fire(main)
