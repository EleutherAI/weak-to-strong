from weak_to_strong.train_config import TrainConfig
import train_simple


models = [
    # "meta-llama/Llama-2-7b-hf",
    # "meta-llama/Llama-2-13b-hf",
    "meta-llama/Meta-Llama-3-8B",
    "mistralai/Mistral-7B-v0.1",
    # "meta-llama/Meta-Llama-3-70B",
    "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen1.5-1.8B",
    "Qwen/Qwen1.5-4B",
    "Qwen/Qwen1.5-7B",
    "Qwen/Qwen1.5-14B",
]

tasks = [
    # {"ds_name": "sciq",},
    {
        "ds_name": "sciq",
        "loss": "kl",
        "weak_labels_path": "/mnt/ssd-1/alexm/weak-to-strong/results/function-grads/"
        "sciq/bs=32-dd=sqrt-dl=1-dn=sciq-e=3-ee=10000000-lp=0-lbmae=1-l=xent-l=1e-05"
        "-ls=cosi_anne-mc=512-mcn=None-ms=pythia-410m-nid=5500-ntd=500-ntd=4000-o=adam-stl"
        "=1-s=0-sg=0-sh=0-twd=0-wlp=None/weak_labels",
    },
    # {"ds_name": "quirky_nli",},
    # {"ds_name": "quirky_nli_weak"},
]

common_args = {
    "batch_size": 32,
    "n_train_docs": 5500,
    "n_test_docs": 500,
    "take_test_from_train": True,
    "n_inference_docs": 0,
    "epochs": 2,
    "load_best_model_at_end": True,
    "save_total_limit": 0,
    "disable_lora": True,
    "results_folder": "/mnt/ssd-1/alexm/weak-to-strong/results",
    "sweep_subfolder": "lr_sweep/weak",
    "eval_every": 50,
    "save_every": 50,
}

sweep_config = {
    "method": "grid",
    "metric": {"name": "eval/auroc", "goal": "maximize"},
    "parameters": {
        "lr": {
            "values": [
                1e-6,
                3e-6,
                1e-5,
                3e-5,
            ]
        },
    },
}

for model in models:
    print(f"--- Starting sweep for {model} ---")
    for task in tasks:
        print(f"---\tfor {task}---")
        for lr in sweep_config["parameters"]["lr"]["values"]:
            args = common_args.copy()
            args.update(task)
            args["model_size"] = model
            args["lr"] = lr
            cfg = TrainConfig(**args)
            train_simple.main(cfg)
