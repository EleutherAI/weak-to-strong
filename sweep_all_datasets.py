import traceback
import sweep


# these are the datasets that appear in OpenAI's w2s paper
datasets = [
    "boolq",
    "cosmos_qa",
    "dream",
    "ethics_justice",
    "ethics_deontology",
    "ethics_virtue",
    "ethics_utilitarianism",
    "anli-r2",
    "cola",
    "sst2",
    "hellaswag",
    "mc_taco",
    "openbookqa",
    "paws",
    "quail",
    "piqa",
    "quartz",
    "sciq",
    "social_i_qa",
    "multirc",
    "wic",
    "twitter_sentiment",
]


def get_model_sizes(dataset):
    wms = ["Qwen/Qwen1.5-0.5B"]
    sms = ["mistralai/Mistral-7B-v0.1"]
    return wms, sms


base_config = {
    "results_folder": "/mnt/ssd-1/alexm/weak-to-strong/results",
    "skip_if_exists": True,
    "gt_eval_every": 50,
    "gt_save_every": 50,
    "w2s_eval_every": 25,
    "w2s_save_every": 25,
    "save_total_limit": 0,
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
