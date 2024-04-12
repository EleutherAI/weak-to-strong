import functools
from dataclasses import dataclass
from random import Random
from typing import Any, Callable, Optional
import hashlib
import warnings

from datasets import (
    Dataset as HfDataset,
    DatasetDict as HfDatasetDict,
    load_dataset as hf_load_dataset,
    concatenate_datasets,
)

from collections import Counter


@dataclass
class DatasetConfig:
    # split -> unshuffled dataset of items
    loader: Callable[[str], HfDataset]
    # formats items to have keys 'txt' and 'hard_label', takes a random.Random rng
    # optionally also adds the key 'choices', a pair of strings, indicating to use the lm head
    formatter: Callable[[Any], Any]
    balance: bool = True


# mapping from dataset name to load function and format function
_REGISTRY: dict[str, DatasetConfig] = {}


def register_dataset(name: str, config: DatasetConfig):
    _REGISTRY[name] = config


def balance(ds: HfDataset, seed: int):
    """Undersample balance to 50/50"""

    label_counts = Counter(ds["hard_label"])
    assert len(label_counts) == 2, "Dataset must be binary"

    # undersample the majority class
    majority_label = max(label_counts, key=lambda k: label_counts[k])
    minority_label = 1 - majority_label
    minority_count = label_counts[minority_label]
    minority_ds = ds.filter(lambda ex: ex["hard_label"] == minority_label)
    majority_ds = (
        ds.filter(lambda ex: ex["hard_label"] == majority_label)
        .shuffle(seed=seed)
        .select(range(minority_count))
    )
    return concatenate_datasets([minority_ds, majority_ds]).shuffle(seed=seed)


def load_and_process_dataset(
    ds_name: str,
    seed: int = 0,
    split_sizes: Optional[dict] = None,
):
    if split_sizes is None:
        split_sizes = dict(train=None, test=None)

    if ds_name not in _REGISTRY:
        raise ValueError(f"Unknown dataset {ds_name}, please register")
    cfg = _REGISTRY[ds_name]
    results = {}
    for split, n_docs in split_sizes.items():
        ds = cfg.loader(split)
        ds = ds.map(functools.partial(cfg.formatter, rng=Random(seed)))  # type: ignore
        if cfg.balance:
            ds = balance(ds, seed)
        try:
            ds = ds.select(range(n_docs))
        except IndexError:
            print(
                f"Warning {ds_name} has < {n_docs} docs after balancing, using all {len(ds)}"
            )

        ds = ds.map(
            lambda ex: {
                "id": hashlib.sha1(ex["txt"].encode()).hexdigest()[:8],
                "soft_label": [1 - float(ex["hard_label"]), float(ex["hard_label"])],
            }
        )
        ds = ds.shuffle(seed=seed)  # shuffling a bit pointless for test set but wtv
        results[split] = ds
    return results


warned_about_choices = set()


def encode_choice(text, tokenizer):
    global warned_about_choices

    c_ids = tokenizer.encode(text, add_special_tokens=False)

    # some tokenizers split off the leading whitespace character
    if tokenizer.decode(c_ids[0]).strip() == "":
        c_ids = c_ids[1:]

    c_ids = tuple(c_ids)
    if len(c_ids) != 1 and c_ids not in warned_about_choices:
        assert c_ids[0] not in [
            c[0] for c in warned_about_choices
        ], "Choice shares first token with another choice"
        warned_about_choices.add(c_ids)
        print(
            f'Warning: Only the first token of multitoken choice "{text}" will be used'
        )
    return c_ids[0]


def tokenize_dataset(
    raw_ds: HfDataset,
    tokenizer: Callable,
    max_ctx: int,
):
    """
    This function prepares the dataset for training. It takes the raw dataset,
    a formatting function, a tokenizer, a maximum context length

    Parameters:
    raw_ds: The raw dataset to be processed.
    tokenizer: The tokenizer to be used on the formatted dataset.
    max_ctx: The maximum context length for the tokenizer.

    Returns:
    ds: The processed and shuffled dataset ready for training.
    """

    def process_function(ex):
        toks = tokenizer(ex["txt"], max_length=max_ctx, truncation=True)
        out = dict(
            input_ids=toks["input_ids"],
        )

        if "choices" in ex:
            choice_toks = [encode_choice(c, tokenizer) for c in ex["choices"]]
            out["choice_input_ids"] = choice_toks

        return out

    ds = raw_ds.map(process_function, batched=False)
    num_max_len = sum(len(x["input_ids"]) == max_ctx for x in ds)  # type: ignore
    print(
        f"{100 * num_max_len / len(ds):.2f}% of examples (truncated to) max length of {max_ctx}"
    )
    return ds


def hf_loader(*hf_name, split_names=None, n_test=None):
    """
    If `split_names` is provided, it maps from the requested
    split name to the actual name in the hugginface dataset.
    If `n_test` is provided, it will concatenate all splits together
    and then take a deterministic test set of size `n_test` from it.
    """

    # this thunk avoids loading datasets at import time
    def thunk(split):
        nonlocal split_names
        if n_test is not None:
            assert split_names is None
            ds = hf_load_dataset(*hf_name)
            if isinstance(ds, HfDatasetDict):
                ds = concatenate_datasets(ds.values())  # type: ignore
            assert isinstance(ds, HfDataset)
            # the seed is fixed so that all runs use the same test pool
            splits = ds.train_test_split(test_size=n_test, seed=0)

            return splits[split]

        if split_names is None:
            split_names = dict()

        return hf_load_dataset(*hf_name, split=split_names.get(split, split))

    return thunk


def sciq_with_support_loader(*hf_name, split_names=None, n_test=None):
    """
    Wraps hf_loader by filtering out examples without support
    """
    base_loader = hf_loader(*hf_name, split_names=split_names, n_test=n_test)

    return lambda split: base_loader(split).filter(lambda x: x["support"] != "")


##########
# ACTUAL DATASETS
##########


def format_mc_taco(ex, rng):
    template = "{sentence}\n\nGiven the above, {question} Is the answer {answer}?"
    return dict(txt=template.format(**ex), hard_label=ex["label"])


register_dataset(
    "mc_taco",
    DatasetConfig(  # we switch train and test bc test is bigger
        loader=hf_loader(  # type: ignore
            "mc_taco", split_names=dict(train="test", test="validation")
        ),
        formatter=format_mc_taco,  # type: ignore
    ),
)


def format_amazon_polarity(ex, rng):
    return dict(txt=f"{ex['title']} {ex['content']}", hard_label=ex["label"])


register_dataset(
    "amazon_polarity",
    DatasetConfig(
        loader=hf_loader("amazon_polarity"),  # type: ignore
        formatter=format_amazon_polarity,  # type: ignore
    ),
)

SCIQ_N_TEST = 3000


def format_sciq(ex, rng):
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        ans = ex["correct_answer"]
    else:
        ans = rng.choice([ex["distractor1"], ex["distractor2"], ex["distractor3"]])

    txt = f"Q: {ex['question']} A: {ans}"
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "sciq",
    DatasetConfig(
        loader=hf_loader("sciq", n_test=SCIQ_N_TEST),  # type: ignore
        formatter=format_sciq,  # type: ignore
    ),
)


def format_sciq_for_lm_head(ex, rng):
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        ans = ex["correct_answer"]
    else:
        ans = rng.choice([ex["distractor1"], ex["distractor2"], ex["distractor3"]])

    txt = f"Q: {ex['question']} A: {ans}. Is this correct?"
    choices = (" No", " Yes")
    return dict(txt=txt, hard_label=hard_label, choices=choices)


register_dataset(
    "sciq_for_lm_head",
    DatasetConfig(
        loader=hf_loader("sciq", n_test=SCIQ_N_TEST),  # type: ignore
        formatter=format_sciq_for_lm_head,  # type: ignore
    ),
)


def format_sciq_for_lm_head_with_support(ex, rng):
    # from https://github.com/EleutherAI/elk-generalization
    template = (
        "Name: Bob\n\nPassage 1:\n{support}\n\nQ1: "
        '"{question}" Is the answer "{answer}"?\nA:'
    )
    choices = (" No", " Yes")
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        ans = ex["correct_answer"]
    else:
        ans = rng.choice([ex["distractor1"], ex["distractor2"], ex["distractor3"]])
    txt = template.format(support=ex["support"], question=ex["question"], answer=ans)
    return dict(txt=txt, hard_label=hard_label, choices=choices)


register_dataset(
    "sciq_for_lm_head_with_support",
    DatasetConfig(
        loader=sciq_with_support_loader("sciq", n_test=SCIQ_N_TEST),  # type: ignore
        formatter=format_sciq_for_lm_head_with_support,  # type: ignore
    ),
)


def format_sciq_with_support(ex, rng):
    # from https://github.com/EleutherAI/elk-generalization
    template = 'Name: Bob\n\nPassage 1:\n{support}\n\nQ1: "{question}" Is the answer "{answer}"?'
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        ans = ex["correct_answer"]
    else:
        ans = rng.choice([ex["distractor1"], ex["distractor2"], ex["distractor3"]])
    txt = template.format(support=ex["support"], question=ex["question"], answer=ans)
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "sciq_with_support",
    DatasetConfig(
        loader=sciq_with_support_loader("sciq", n_test=SCIQ_N_TEST),  # type: ignore
        formatter=format_sciq_with_support,  # type: ignore
    ),
)


def format_anthropic_hh(ex, rng) -> dict:
    ch, rej = ex["chosen"], ex["rejected"]
    ch_last_assistant = ch.rfind("Assistant:")
    ch_prompt, ch_response = ch[:ch_last_assistant].rstrip(), ch[ch_last_assistant:]
    rej_last_assistant = rej.rfind("Assistant:")
    rej_prompt, rej_response = (
        rej[:rej_last_assistant].rstrip(),
        rej[rej_last_assistant:],
    )
    if ch_prompt != rej_prompt:
        warnings.warn(f"Prefixs don't match in {ex}")
    resps = [ch_response, rej_response]
    rng.shuffle(resps)
    txt = f"{ch_prompt}\n\n<|Completion 1|>{resps[0]}\n\n<|Completion 2|>{resps[1]}"
    return dict(
        txt=txt, hard_label=resps[1] == ch_response
    )  # 1 if the second is better


register_dataset(
    "anthropic_hh",
    DatasetConfig(
        loader=hf_loader("Anthropic/hh-rlhf"),  # type: ignore
        formatter=format_anthropic_hh,  # type: ignore
    ),
)


def format_anthropic_hh_for_lm_head(ex, rng):
    out = format_anthropic_hh(ex, rng)
    out["txt"] = f"{out['txt']}\n\nWhich completion is better?"
    out["choices"] = (" 1", " 2")
    return out


register_dataset(
    "anthropic_hh_for_lm_head",
    DatasetConfig(
        loader=hf_loader("Anthropic/hh-rlhf"),  # type: ignore
        formatter=format_anthropic_hh_for_lm_head,  # type: ignore
    ),
)


def format_cosmosqa(ex, rng):
    true_answer = ex["answer" + str(ex["label"])]
    if "None of the above choices ." in true_answer:
        hard_label = 0
    else:
        assert "None of the above choices" not in true_answer, true_answer
        hard_label = int(rng.random() < 0.5)
    if hard_label:
        answer = true_answer
    else:
        candidate_answers = [ex["answer" + str(i)] for i in range(4)]
        answer = rng.choice([x for x in candidate_answers if x != true_answer])
    txt = f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {answer}"
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "cosmos_qa",
    DatasetConfig(
        loader=hf_loader("cosmos_qa", split_names=dict(test="validation")),  # type: ignore
        formatter=format_cosmosqa,  # type: ignore
    ),
)


def format_boolq(ex, rng):
    hard_label = int(ex["answer"])
    txt = f"Passage: {ex['passage']}\nQuestion: {ex['question']}"
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "boolq",
    DatasetConfig(
        loader=hf_loader("boolq", split_names=dict(test="validation")),  # type: ignore
        formatter=format_boolq,  # type: ignore
    ),
)


# Quirky datasets

quirky_templates = {
    "capitals": "{admin_name}, {country}\n\n{city}",
    "hemisphere": "{city}",
    "population": "{city}",
    "sciq": "{support}\n\n{question} {answer}",
    # "sciq": "Name: Alice\n\nPassage 1:\n{support}\n\nQ1: "
    #     '"{question}" Is the answer "{answer}"?\nA:',
    "sentiment": "{title}\n{review}",
    # "sentiment": "Name: Alice\n\nTitle: {title}\n{review}\n\nQ: Does the above "
    #     "review have a positive or negative sentiment?\nA:",
    "nli": "{premise}\n\n{hypothesis}",
    "authors": "{title}\n{author}",
    "addition": "{op1} | {op2} | {result}",
    "subtraction": "{op1} | {op2} | {result}",
    "multiplication": "{op1} | {op2} | {result}",
    "modularaddition": "{op1} | {op2} | {result}",
    "squaring": "{op1} | {result}",
}


def format_quirky(ex, rng, ds_name, label_col="alice_label"):
    return dict(
        txt=quirky_templates[ds_name].format(**ex["template_args"]),
        hard_label=ex[label_col],
    )


for ds_name in quirky_templates:
    for label_col in ["alice_label", "bob_label"]:
        register_dataset(
            f"quirky_{ds_name}" + ("_weak" if label_col == "bob_label" else ""),
            DatasetConfig(
                # NOTE: this is using the same examples as the quirky models were finetuned on
                loader=hf_loader(f"EleutherAI/quirky_{ds_name}_raw"),  # type: ignore
                formatter=functools.partial(
                    format_quirky, ds_name=ds_name, label_col=label_col
                ),  # type: ignore
                balance=False,
            ),
        )


VALID_DATASETS: list[str] = list(_REGISTRY.keys())


"""
from datasets import disable_caching
disable_caching()

from weak_to_strong.datasets import load_dataset, VALID_DATASETS
import numpy as np

ds_name = "boolq"
print(VALID_DATASETS)

ds = load_and_process_dataset(ds_name, split_sizes=dict(train=500, test=10))
train = list(ds['train'])
test = list(ds['test'])
print(test[0])
print(np.mean([x['hard_label'] for x in train]))
"""
