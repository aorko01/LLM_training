import time
from pathlib import Path
from tokenizer import Tokenizer

VOCAB_SIZE = 500


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def run(tokenizer, text: str, vocab_size: int) -> dict:
    t0 = time.perf_counter()
    tokenizer.train(text, vocab_size)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    ids = tokenizer.encode(text)
    encode_time = time.perf_counter() - t0

    assert tokenizer.decode(ids) == text, "encode/decode mismatch"

    return {
        "train_time": train_time,
        "encode_time": encode_time,
        "num_tokens": len(ids),
        "compression": len(text.encode("utf-8")) / len(ids),
    }


if __name__ == "__main__":
    text = load_text("dataset/tiny_shakespeare.txt")   # <- swap dataset: point this at any .txt file
    tokenizer = Tokenizer()        # <- swap implementation: any class with train/encode/decode

    stats = run(tokenizer, text, VOCAB_SIZE)
    print(f"train:  {stats['train_time']:.3f}s")
    print(f"encode: {stats['encode_time']:.3f}s")
    print(f"tokens: {stats['num_tokens']} ({stats['compression']:.2f}x compression)")