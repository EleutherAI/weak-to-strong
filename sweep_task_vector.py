import torch
import wandb
from evaluate_task_vector import main as evaluate_task_vector_main
import fire


def main(
    coef_step: float = 0.1,
    coef_min: float = -5.0,
    coef_max: float = 5.0,
    sweep_method: str = "bayes",
    sweep_steps: int = 10,
    task_seed: int = 0,
    **kwargs
):
    torch.manual_seed(task_seed)
    wandb_name = (
        f"model_{kwargs.get('model_size', 'default').split('/')[-1]}_"
        f"ds_{kwargs.get('ds_name', 'default')}_"
        f"seed_{task_seed}_"
        f"coef_step_{coef_step}_"
        f"coef_min_{coef_min}_"
        f"coef_max_{coef_max}_"
        "sweep_task_vector"
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
        lambda: evaluate_task_vector_main(**kwargs),
        count=sweep_steps,
    )
    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
