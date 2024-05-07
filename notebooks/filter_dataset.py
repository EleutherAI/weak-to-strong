import fire
from weak_to_strong.analysis import *


def main(ds_name, weak_labels_path=None):
    model_size = "meta-llama/Meta-Llama-3-8B"
    if weak_labels_path is None:
        weak_labels_path = f"/mnt/ssd-1/alexm/weak-to-strong/results/logconf/bs=32-dd=sqrt-dl=1-dn={ds_name}-e=2-ee=50-lp=0-lbmae=1-l=xent-l=1e-05-lf=1.0-ls=cosi_anne-mc=1024-mcn=None-ms=Qwen1.5-0.5B-nid=10000-ntd=1000-ntd=10000-o=adam-stl=0-s=0-sg=0-sh=0-ttft=0-twd=0-wlp=None/weak_labels"

    print("loading strong model and weak labels")
    model0, train_ds, test_ds = get_model_and_dataset(model_size, ds_name, weak_labels_path)

    BS = 4

    print("getting strong model activations...")
    activations_by_layer = make_acts(model0, train_ds, eval_batch_size=BS)

    torch.save(activations_by_layer, f"acts_{ds_name}.pt")
    print(f"saved activations to acts_{ds_name}.pt")

    all_last_acts = cat_last_acts(train_ds, activations_by_layer, eval_batch_size=BS)

    torch.save(all_last_acts, f"last_acts_{ds_name}.pt")
    print(f"saved last-position activations to last_acts_{ds_name}.pt")

    print("running kNN")
    mid_layer = len(all_last_acts) // 2 - 1

    x = all_last_acts[mid_layer].to(torch.float32).numpy()
    yc = np.array(train_ds['soft_pred'])[:, 1]
    k = 100

    confidence_score = knn_score(x, yc, k)

    print("keeping top half")

    thresh = np.median(confidence_score)
    good_points = (confidence_score > thresh)

    # filter dataset
    train_ds_untok = load_from_disk(weak_labels_path)
    good_ds = train_ds_untok.select(np.where(good_points))

    # save dataset
    good_ds.save_to_disk(weak_labels_path + "_filtered")

    print(f"saved filtered data to {weak_labels_path}_filtered")

if __name__ == "__main__":
    fire.Fire(main)