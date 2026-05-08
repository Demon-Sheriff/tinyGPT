"""
Train BPE tokenizers at different vocabulary sizes on a fixed byte corpus.

Why: Different V means different tokenizers, which means different token sequences
for the same text. To compare losses across V, we need to (a) train each tokenizer
on the same byte corpus, (b) report performance in bits-per-byte (BPB) which is
tokenizer-agnostic.
"""
import os
import json
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def train_bpe(corpus_path: str, vocab_size: int, output_path: str):
    """Train a byte-level BPE tokenizer with the given vocab size."""
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train([corpus_path], trainer)
    tokenizer.save(output_path)
    print(f"Saved tokenizer (V={vocab_size}) to {output_path}")
    return tokenizer


def encode_corpus(tokenizer: Tokenizer, text: str) -> list:
    """Encode a text string into token IDs."""
    return tokenizer.encode(text).ids


def measure_bytes_per_token(tokenizer_path: str, sample_text: str) -> float:
    """How many bytes does each token represent on average for this tokenizer."""
    tok = Tokenizer.from_file(tokenizer_path)
    n_tokens = len(tok.encode(sample_text).ids)
    n_bytes = len(sample_text.encode("utf-8"))
    return n_bytes / n_tokens


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True,
                        help="Path to a plain-text corpus file for training")
    parser.add_argument("--vocab_sizes", type=int, nargs="+",
                        default=[1024, 8192, 32768])
    parser.add_argument("--output_dir", type=str, default="tokenizers")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for v in args.vocab_sizes:
        out = os.path.join(args.output_dir, f"bpe_v{v}.json")
        train_bpe(args.corpus, v, out)

    sample = open(args.corpus).read(1_000_000)
    print("\nBytes per token (on 1MB sample):")
    for v in args.vocab_sizes:
        out = os.path.join(args.output_dir, f"bpe_v{v}.json")
        bpt = measure_bytes_per_token(out, sample)
        print(f"  V={v:>6}: {bpt:.3f} bytes/token")
