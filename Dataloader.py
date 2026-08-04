import os

import numpy as np
import torch


class Dataloader:
    """
    Streams (x, y) next-token-prediction batches from a pre-tokenized WikiText-103
    shard on disk.

    Expects `{data_dir}/{split}.bin` to exist, containing uint16 GPT-2 BPE token
    ids. Generate these files first with `data/wikitext103/prepare.py`.

    We use np.memmap so the tokens never have to be fully loaded into RAM -
    important since WikiText-103 tokenizes to hundreds of millions of tokens.
    """

    def __init__(self, B, T, split="train", data_dir="data/wikitext103"):
        self.B = B
        self.T = T
        assert split in {"train", "val"}, f"split must be 'train' or 'val', got {split!r}"
        self.split = split
        self.data_dir = data_dir

        self.data_path = os.path.join(data_dir, f"{split}.bin")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Could not find {self.data_path}. Run `python data/wikitext103/prepare.py` "
                "first to download and tokenize WikiText-103."
            )

        # Re-opened as a fresh memmap on every __init__ (cheap - just maps the file,
        # doesn't read it into memory) rather than pickled/shared, so this stays
        # safe if a caller ever forks worker processes.
        self.tokens = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        assert len(self.tokens) > B * T + 1, (
            f"{self.data_path} only has {len(self.tokens)} tokens, "
            f"which is too few for a single batch of size {B}x{T}"
        )

        self.current_position = 0

    def __len__(self):
        return len(self.tokens)

    def reset(self):
        """Restart sequential iteration from the beginning of the shard."""
        self.current_position = 0

    def next_batch(self):
        """Sequential batches (used for training) - walks through the shard in order,
        wrapping back to the start once exhausted."""
        B, T = self.B, self.T
        needed = B * T + 1
        if self.current_position + needed > len(self.tokens):
            self.current_position = 0

        chunk = self.tokens[self.current_position: self.current_position + needed]
        buf = torch.from_numpy(chunk.astype(np.int64))
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        return x, y

    def random_batch(self):
        """Randomly sampled batch (used for validation/eval so we're not always
        scoring the same prefix of the shard)."""
        B, T = self.B, self.T
        ix = torch.randint(len(self.tokens) - T - 1, (B,))
        x = torch.stack([torch.from_numpy(self.tokens[i: i + T].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.tokens[i + 1: i + 1 + T].astype(np.int64)) for i in ix])
        return x, y