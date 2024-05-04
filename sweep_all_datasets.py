import traceback
import sweep


# these are the datasets in the order as they appear in OpenAI's w2s paper
datasets = [
    "boolq",  # dataset 1 from the paper
    "cosmos_qa",  # 2
    "dream",  # 3
    "ethics_justice",  # 4
    "ethics_deontology",  # 5
    "ethics_virtue",  # 6
    "ethics_utilitarianism",  # 7
    "anli-r2",  # 8
    "cola",  # 9
    "sst2",  # 10
    "hellaswag",  # 11
    "mc_taco",  # 12
    "openbookqa",  # 13
    "paws",  # 14
    "quail",  # 15
    "piqa",  # 16
    "quartz",  # 17
    "sciq",  # 18
    "social_i_qa",  # 19
    "multirc",  # 20
    "wic",  # 21
    "twitter_sentiment",  # 22
]


# GPT-3.5 acc (or whatever the second largest model is in their paper)
# for guaging difficulty of the dataset to determine model sizes
original_accs = {
    "boolq": 87,
    "cosmos_qa": 82,
    "dream": 76,
    "ethics_justice": 88,
    "ethics_deontology": 83,
    "ethics_virtue": 92,
    "ethics_utilitarianism": 83,
    "anli-r2": 94,
    "cola": 89,
    "sst2": 82,
    "hellaswag": 86,
    "mc_taco": 94,
    "openbookqa": 83,
    "paws": 70,
    "quail": 89,
    "piqa": 93,
    "quartz": 96,
    "sciq": 76,
    "social_i_qa": 90,
    "multirc": 82,
    "wic": 89,
    "twitter_sentiment": 92,
}


def get_model_sizes(dataset):
    original_acc = original_accs[dataset]
    if original_acc >= 94:
        wms = ["EleutherAI/pythia-160m-v0"]
        sms = ["Qwen/Qwen1.5-0.5B", "Qwen/Qwen1.5-1.8B", "Qwen/Qwen1.5-4B"]
    elif original_acc >= 90:
        wms = ["EleutherAI/pythia-410m"]
        sms = ["Qwen/Qwen1.5-1.8B", "meta-llama/Meta-Llama-3-8B"]
    elif original_acc >= 85:
        wms = ["Qwen/Qwen1.5-0.5B"]
        sms = ["Qwen/Qwen1.5-4B", "meta-llama/Meta-Llama-3-8B"]
    else:
        wms = ["Qwen/Qwen1.5-4B"]
        sms = ["mistralai/Mistral-7B-v0.1", "meta-llama/Meta-Llama-3-8B"]
    return wms, sms


base_config = {
    "results_folder": "/mnt/ssd-1/alexm/weak-to-strong/results",
    "skip_if_exists": True,
    "gt_eval_every": 50,
    "gt_save_every": 50,
    "w2s_eval_every": 25,
    "w2s_save_every": 25,
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "disable_lora": True,
    "n_train_docs": 10_000,
    "n_inference_docs": 10_000,
    "n_test_docs": 1000,
    "epochs": 2,
    "w2s_store_hiddens": True,
    "sweep_subfolder": "logconf",
    "w2s_loss": "logconf",
}

if __name__ == "__main__":
    for dataset in datasets:
        wms, sms = get_model_sizes(dataset)
        config = base_config.copy()
        config["weak_model_sizes"] = wms
        config["strong_model_sizes"] = sms
        config["ds_name"] = dataset
        print(f"For {dataset}, using weak models {wms} and strong models {sms}")
        try:
            sweep.main(**config)
        except Exception as e:
            print(f"Failed to run {dataset}: {e}")
            traceback.print_exc()
    print("Finished running all datasets.")
