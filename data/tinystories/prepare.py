"""
Download and tokenize the TinyStories dataset
(https://huggingface.co/datasets/roneneldan/TinyStories) into train.bin / val.bin
files of uint16 GPT-2 BPE token ids, ready to be memory-mapped by Dataloader.py.

Usage:
    python data/tinystories/prepare.py

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


def tokenize(example):
    ids = enc.encode_ordinary(example["text"])
    ids.append(enc.eot_token)  # mark end-of-story so the model learns document boundaries
    return {"ids": ids, "len": len(ids)}


def main():
    dataset = load_dataset("roneneldan/TinyStories")

    if "validation" in dataset:
        split_dataset = {"train": dataset["train"], "val": dataset["validation"]}
    else:
        # some dataset revisions only ship a single split - carve out a small val set ourselves
        split = dataset["train"].train_test_split(test_size=0.0005, seed=1337, shuffle=True)
        split_dataset = {"train": split["train"], "val": split["test"]}

    tokenized = {
        split_name: ds.map(
            tokenize,
            remove_columns=ds.column_names,
            desc=f"tokenizing {split_name} split",
            num_proc=NUM_PROC,
        )
        for split_name, ds in split_dataset.items()
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
