"""
Download and tokenize the WikiText-103 dataset
(https://huggingface.co/datasets/wikitext, config "wikitext-103-raw-v1") into
train.bin / val.bin files of uint16 GPT-2 BPE token ids, ready to be
memory-mapped by Dataloader.py.

Usage:
    python data/wikitext103/prepare.py

Requires: datasets, tiktoken, numpy, tqdm  (see requirements.txt)
"""
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

NUM_PROC = max(1, (os.cpu_count() or 1) // 2)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_SHARDS = 1024  # for writing progress in chunks without holding everything in RAM

enc = tiktoken.get_encoding("gpt2")


def is_blank(example):
    # WikiText-103 rows include empty lines and bare section-header lines
    # (e.g. " = Valkyria Chronicles III = \n"); dropping whitespace-only rows
    # avoids wasting tokens/eot markers on them. Header lines themselves are
    # left in place since they're still real (if short) text.
    return len(example["text"].strip()) > 0


def tokenize(example):
    ids = enc.encode_ordinary(example["text"])
    ids.append(enc.eot_token)  # mark a boundary after every row
    return {"ids": ids, "len": len(ids)}


def main():
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    # wikitext-103 already ships train/validation/test splits
    split_dataset = {"train": dataset["train"], "val": dataset["validation"]}

    filtered = {
        split_name: ds.filter(is_blank, num_proc=NUM_PROC, desc=f"filtering blank rows ({split_name})")
        for split_name, ds in split_dataset.items()
    }

    tokenized = {
        split_name: ds.map(
            tokenize,
            remove_columns=ds.column_names,
            desc=f"tokenizing {split_name} split",
            num_proc=NUM_PROC,
        )
        for split_name, ds in filtered.items()
    }

    for split_name, ds in tokenized.items():
        arr_len = int(np.sum(ds["len"], dtype=np.uint64))
        out_path = os.path.join(OUT_DIR, f"{split_name}.bin")
        arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(arr_len,))

        idx = 0
        for shard_idx in tqdm(range(NUM_SHARDS), desc=f"writing {out_path}"):
            shard = ds.shard(num_shards=NUM_SHARDS, index=shard_idx, contiguous=True).with_format("numpy")
            shard_ids = np.concatenate(shard["ids"])
            arr[idx: idx + len(shard_ids)] = shard_ids
            idx += len(shard_ids)
        arr.flush()
        print(f"{split_name}.bin: {arr_len:,} tokens -> {out_path}")


if __name__ == "__main__":
    main()
