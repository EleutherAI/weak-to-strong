from datetime import datetime
import torch
import wandb
from evaluate_task_vector import main as evaluate_task_vector_main
import fire

from weak_to_strong.common import wandb_finish


def main(
    coef_step: float = 0.1,
    coef_min: float = -5.0,
    coef_max: float = 5.0,
    sweep_method: str = "bayes",
    sweep_steps: int = 1,
    task_seed: int = 0,
    **kwargs
):
    torch.manual_seed(task_seed)
    wandb_name = (
        f"model_{kwargs.get('model_size', 'default').split('/')[-1]}_"
        f"ds_{kwargs.get('ds_name', 'default')}_"
        f"sweep_task_vector_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    coef_values = torch.arange(coef_min, coef_max, coef_step).tolist()
    sweep_configuration = {
        "name": wandb_name,
        "method": sweep_method,
        "metric": {"goal": "maximize", "name": "task_vector/accuracy"},
        "parameters": {
            "coef_best": {"values": coef_values},
            "coef_final": {"values": coef_values},
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="weak-to-strong")
    wandb.agent(
        sweep_id,
        lambda: evaluate_task_vector_main(coef_best=1, coef_final=0, **kwargs),
        count=1,
    )
    wandb.agent(
        sweep_id,
        lambda: evaluate_task_vector_main(coef_best=0, coef_final=1, **kwargs),
        count=1,
    )
    wandb.agent(
        sweep_id,
        lambda: evaluate_task_vector_main(**kwargs),
        count=sweep_steps,
    )
    wandb_finish(sweep=True)


if __name__ == "__main__":
    fire.Fire(main)
