"""Held-out test split for the paper (analysis data, never used for training or selection).

Continues the benchmark's own FineWeb stream (utils/prepare_data.py) one pool further, exactly as the Dust repository
does for its test.pt: skip 200,000 documents, the ~20.5M-token training pool, the ~1.15M-token validation pool,
then a second ~1.15M-token pool whose first 544 rows are the test split. The regenerated train and validation
splits must match the pinned hashes before anything is written.

    python analysis/prepare_test.py --data-dir /data/data
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.prepare_data import (EXPECTED, POOL_TOKENS, SEQ, SKIP_DOCS, TOKENIZER, TOKENIZER_SHA256, TRAIN_ROWS,  # noqa: E402
                                VAL_POOL_TOKENS, VAL_ROWS, encode_split, token_hash)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    args = parser.parse_args()
    from datasets import load_dataset
    from tokenizers import Tokenizer
    if hashlib.sha256(TOKENIZER.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise SystemExit('not the fixed tokenizer')
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    documents = iter(load_dataset('HuggingFaceFW/fineweb', name='sample-10BT', split='train', streaming=True))
    for _ in range(SKIP_DOCS):
        next(documents)
    train = encode_split(documents, tokenizer, POOL_TOKENS)[:TRAIN_ROWS]
    val = encode_split(documents, tokenizer, VAL_POOL_TOKENS)[:VAL_ROWS]
    for name, rows in (('train.pt', train), ('val.pt', val)):
        if token_hash(rows) != EXPECTED[name]:
            raise SystemExit(f'regenerated {name} does not match the pinned split')
        print(f'{name}: matches the pinned hash', flush=True)
    test = encode_split(documents, tokenizer, VAL_POOL_TOKENS)[:VAL_ROWS].clone()
    assert tuple(test.shape) == (VAL_ROWS, SEQ)
    torch.save(test, args.data_dir / 'test.pt')
    info = dict(rows=len(test), prediction_tokens=len(test) * (SEQ - 1), token_sha256=token_hash(test))
    (args.data_dir / 'test_manifest.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
