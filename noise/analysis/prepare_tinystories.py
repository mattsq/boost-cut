"""A second dataset for the paper: TinyStories, in the benchmark's format (analysis data).

The fixed BPE-4096 tokenizer; documents joined with <|endoftext|> and cut into rows of 2,049 tokens, as in
utils/prepare_data.py. train.pt: the first 4,880 rows (~10M prediction tokens) of the train split, in stream order;
val.pt and test.pt: the first and second 544 rows of the separate validation split.

    python analysis/prepare_tinystories.py --data-dir /data/tinystories
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.prepare_data import SEQ, TOKENIZER, TOKENIZER_SHA256, TRAIN_ROWS, VAL_ROWS, encode_split, token_hash  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args()
    from datasets import load_dataset
    from tokenizers import Tokenizer
    if hashlib.sha256(TOKENIZER.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise SystemExit('not the fixed tokenizer')
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    args.data_dir.mkdir(parents=True, exist_ok=True)
    train_docs = iter(load_dataset('roneneldan/TinyStories', split='train', streaming=True))
    train = encode_split(train_docs, tokenizer, (TRAIN_ROWS + 1) * SEQ)[:TRAIN_ROWS].clone()
    held = encode_split(iter(load_dataset('roneneldan/TinyStories', split='validation', streaming=True)), tokenizer,
                        (2 * VAL_ROWS + 1) * SEQ)
    val, test = held[:VAL_ROWS].clone(), held[VAL_ROWS:2 * VAL_ROWS].clone()
    manifest = {}
    for name, rows in (('train.pt', train), ('val.pt', val), ('test.pt', test)):
        assert tuple(rows.shape)[1] == SEQ
        torch.save(rows, args.data_dir / name)
        manifest[name] = dict(rows=len(rows), prediction_tokens=len(rows) * (SEQ - 1), token_sha256=token_hash(rows))
    (args.data_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
