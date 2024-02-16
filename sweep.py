from train_simple import main as train_simple_main
from typing import List, Union, Optional

import fire


def split_model_sizes(model_sizes: Union[List[str], str]) -> List[str]:
    if isinstance(model_sizes, str):
        return model_sizes.split(",")
    return model_sizes


def main(
    model_sizes: Optional[Union[List[str], str]] = None,
    weak_model_sizes: Optional[Union[List[str], str]] = None,
    strong_model_sizes: Optional[Union[List[str], str]] = None,
    train_self_to_self: bool = False,
    **kwargs
):
    """Sweep over model sizes and train weak-to-strong models.
    Can either:
        (1) specify a list of *increasing* model_sizes, in which case we
            train all pairs of models, or
        (2) specify weak_model_sizes and strong_model_sizes
            separately, in which case we train all pairs of
            weak_model_sizes and strong_model_sizes.
    Args:
        model_sizes: list of model sizes to train
        weak_model_sizes: list of weak model sizes to train
        strong_model_sizes: list of strong model sizes to train
        train_self_to_self: if True, train weak-to-strong models where
            weak and strong models are the same size.
            Must be used with model_sizes.
        kwargs: other arguments to pass to train_simple_main
    """
    assert (
        "weak_model_size" not in kwargs
        and "model_size" not in kwargs
        and "weak_labels_path" not in kwargs
    ), (
        "Need to use model_sizes or weak_model_sizes/strong_model_sizes "
        "when using sweep.py"
    )
    if model_sizes is None:
        assert weak_model_sizes is not None and strong_model_sizes is not None
        weak_model_sizes = split_model_sizes(weak_model_sizes)
        strong_model_sizes = split_model_sizes(strong_model_sizes)
        all_model_sizes = weak_model_sizes + strong_model_sizes
        weak_to_strong_model_sizes = [
            (weak, strong)
            for weak in weak_model_sizes
            for strong in strong_model_sizes
        ]
    else:
        assert weak_model_sizes is None and strong_model_sizes is None
        all_model_sizes = model_sizes = split_model_sizes(model_sizes)
        weak_to_strong_model_sizes = [
            (model_sizes[i], model_sizes[j])
            for i in range(len(model_sizes))
            for j in range(
                i if train_self_to_self else i + 1,
                len(model_sizes)
            )
        ]

    print("Running ground truth models")
    for model_size in all_model_sizes:
        print(f"Running ground truth {model_size}")
        # try:
        train_simple_main(model_size=model_size, **kwargs)
        # except Exception as e:
        #     print(f"Failed to run ground truth {model_size}: {e}")

    print("Running transfer models")
    for weak_model_size, strong_model_size in weak_to_strong_model_sizes:
        print(f"Running weak {weak_model_size} to strong {strong_model_size}")
        try:
            train_simple_main(
                model_size=strong_model_size,
                weak_model_size=weak_model_size,
                **kwargs,
            )
        except Exception as e:
            print(
                f"Failed to run weak {weak_model_size} to strong {strong_model_size}: {e}"
            )
    print("Finished running all models")


if __name__ == "__main__":
    # see train_simple.py for valid args
    fire.Fire(main)
