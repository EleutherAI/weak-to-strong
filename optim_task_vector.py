import torch
import torch_optimizer as toptim
import wandb
from evaluate_task_vector import main as evaluate_task_vector_main
import fire


class TaskVectorModule(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.coefs = torch.nn.Parameter(torch.empty(2))
        torch.nn.init.normal_(self.coefs, mean=0, std=1)
        self.kwargs = kwargs

    def forward(self):
        assert isinstance(self.coefs[0], torch.Tensor)
        assert self.coefs[0].requires_grad
        return evaluate_task_vector_main(
            coef_best=self.coefs[0],
            coef_final=self.coefs[1],
            **self.kwargs
        )


def main(
    task_optim: str = "adam",
    task_lr: float = 0.1,
    task_max_steps: int = 50,
    task_max_steps_wo_improvement: int = 5,
    task_log_every: int = 5,
    task_device: str = "cuda",
    task_dtype: str = "float32",
    task_seed: int = 0,
    **kwargs
):
    torch.manual_seed(task_seed)
    dtype = getattr(torch, task_dtype)
    config = {}
    config.update(kwargs)
    config.update({
        "task_optim": task_optim,
        "task_lr": task_lr,
        "task_max_steps": task_max_steps,
        "task_max_steps_wo_improvement": task_max_steps_wo_improvement,
        "task_log_every": task_log_every,
        "task_device": task_device,
    })
    wandb_name = (
        f"model_{kwargs.get('model_size', 'default').split('/')[-1]}_"
        f"ds_{kwargs.get('ds_name', 'default')}_"
        f"task_optim_{task_optim}"
    )
    wandb.init(
        project="weak-to-strong",
        config=config,
        group=kwargs.get("sweep_subfolder", "default"),
        job_type="task_vector",
        name=wandb_name,
        dir=kwargs.get("results_folder", "/tmp/results"),
        reinit=True,
    )
    module = TaskVectorModule(**kwargs).to(
        device=task_device, dtype=dtype
    )
    assert module.coefs.requires_grad
    print(f"coefs dtype: {module.coefs.dtype}")
    print(f"coefs device: {module.coefs.device}")
    trainable_params = list(module.parameters())
    print(f"trainable_params: {trainable_params}")
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
    best_loss = torch.tensor([1.0], device=task_device, dtype=dtype)
    steps_wo_improvement = 0
    for step in range(task_max_steps):
        loss = module()
        assert loss.requires_grad
        if step % task_log_every == 0:
            print(f"step: {step}, loss: {loss.item()}")
            wandb.log({
                "loss": loss.item(),
                "coef_best": module.coefs[0].item(),
                "coef_final": module.coefs[1].item(),
                "step": step,
            })
        if loss < best_loss:
            best_loss = loss
            steps_wo_improvement = 0
        else:
            steps_wo_improvement += 1
        if steps_wo_improvement >= task_max_steps_wo_improvement:
            break
        optimizer.zero_grad()
        loss.backward()
        assert module.coefs.requires_grad
        print(f"coef grads: {module.coefs.grad}")
        optimizer.step()
    print(f"best_loss: {best_loss.item()}")
    print(f"best_coefs: {module.coefs.detach().tolist()}")
    wandb.log({
        "best_loss": best_loss.item(),
        "best_coef_best": module.coefs[0].item(),
        "best_coef_final": module.coefs[1].item(),
    })
    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
