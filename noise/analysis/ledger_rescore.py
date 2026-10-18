"""Independent re-scoring and hashing of finished runs, for the ledgers (analysis only).

For each named run it records:
- the sha256 of its final.pt, config.json, result.json and metrics.jsonl;
- the exact command it ran (the first line of /data/logs/<name>.log);
- its final weights re-scored from the checkpoint alone, in a fresh model built from the run's config.json (no
  training code involved): validation and test loss on the split folder the run trained from, with the token hash of
  each split;
- for runs of the benchmark's own 38M model on the benchmark's data, the validation loss from utils/eval.py itself,
  the official scorer.

Results append to --out as JSON lines; a run whose final.pt hash is already there is skipped.

    python analysis/ledger_rescore.py --runs bc38f_s0,bp38_s0 --out /data/ledger/rescore.jsonl
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import eval as official  # noqa: E402
from utils.model import GPT, GPTConfig  # noqa: E402
from utils.prepare_data import EXPECTED, SEQ, token_hash  # noqa: E402


def sha256(path, block=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block), b''):
            digest.update(chunk)
    return digest.hexdigest()


def command(name, logs):
    """The command the launcher ran, from the first line of the run's log."""
    path = logs / f'{name}.log'
    if not path.exists():
        return None
    first = path.open(errors='ignore').readline().strip()
    return first[2:] if first.startswith('$ ') else None


def data_dir_of(cmd, default):
    """The last --data-dir of the command: the launcher passes its default first, a run's own choice after it."""
    args = shlex.split(cmd or '')
    dirs = [args[i + 1] for i, a in enumerate(args[:-1]) if a == '--data-dir']
    return Path(dirs[-1]) if dirs else default


def split(path):
    rows = torch.load(path, map_location='cpu', weights_only=True)
    assert isinstance(rows, torch.Tensor) and rows.ndim == 2 and rows.shape[1] == SEQ, path
    return rows.long()


def model_config(config):
    """The run's model shape; benchmark entries record none, since they train the fixed model."""
    return GPTConfig(**config['model']) if 'model' in config else GPTConfig()


def build(config, device):
    """A fresh model of the run's architecture and size; the weights come only from the checkpoint."""
    model_config_ = model_config(config)
    cls = GPT
    if config.get('arch') == 'plain':
        spec = importlib.util.spec_from_file_location('plain_gpt', ROOT / 'analysis/plain_gpt.py')
        plain = sys.modules['plain_gpt'] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plain)
        cls = plain.PlainGPT
    with torch.device('meta'):
        model = cls(model_config_)
    model.to_empty(device=device)
    model.init_weights()  # rotary tables and the embeddings' storage type; every weight is then loaded
    return model


@torch.no_grad()
def mean_loss(model, rows, device, batch_size=16):
    """utils/eval.py's computation: summed next-token loss in float64 over all predictions, bf16 autocast."""
    total = torch.zeros(2, dtype=torch.float64, device=device)
    for batch in rows.split(batch_size):
        batch = batch.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), loss_reduction='sum')
        total[0] += loss.double()
        total[1] += batch[:, 1:].numel()
    return (total[0] / total[1]).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('/data/data'))
    parser.add_argument('--runs', required=True, help='Comma-separated run names.')
    parser.add_argument('--runs-dir', type=Path, default=Path('/data/runs'))
    parser.add_argument('--logs-dir', type=Path, default=Path('/data/logs'))
    parser.add_argument('--out', type=Path, default=Path('/data/ledger/rescore.jsonl'))
    args = parser.parse_args()
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        done = {json.loads(line).get('sha256', {}).get('final.pt') for line in args.out.read_text().splitlines() if line}
    benchmark_model = GPTConfig()
    for name in [n for n in args.runs.split(',') if n]:
        run = args.runs_dir / name
        record = dict(name=name)
        try:
            files = {f: sha256(run / f) for f in ('final.pt', 'config.json', 'result.json', 'metrics.jsonl')
                     if (run / f).exists()}
            if 'final.pt' not in files:
                print(json.dumps(dict(name=name, skipped='no final.pt')), flush=True)
                continue
            if files['final.pt'] in done:
                continue
            config = json.loads((run / 'config.json').read_text())
            result = json.loads((run / 'result.json').read_text())
            cmd = command(name, args.logs_dir)
            data = data_dir_of(cmd, args.data_dir)
            state = torch.load(run / 'final.pt', map_location='cpu', weights_only=True)
            model = build(config, device)
            model.load_state_dict(state, strict=True)
            model.eval().requires_grad_(False)
            record.update(sha256=files, command=cmd, arch=config.get('arch', 'noise'), data_dir=str(data),
                          params=sum(p.numel() for p in model.parameters()),
                          logged=dict(val=result.get('val_loss'), test=result.get('test_loss'),
                                      train_seconds=result.get('train_seconds')))
            rescored, hashes = {}, {}
            for split_name in ('val', 'test'):
                path = data / f'{split_name}.pt'
                if path.exists():
                    rows = split(path)
                    hashes[split_name] = token_hash(rows)
                    rescored[split_name] = mean_loss(model, rows, device)
            record.update(rescored=rescored, split_token_sha256=hashes)
            on_benchmark = data == args.data_dir and hashes.get('val') == EXPECTED.get('val.pt')
            if config.get('arch', 'noise') == 'noise' and model_config(config) == benchmark_model and on_benchmark:
                record['official_val'] = official.evaluate(state, data)  # the benchmark's scorer, its own model copy
            del model, state
            torch.cuda.empty_cache()
        except Exception as e:  # recorded, never silent
            record['error'] = repr(e)
        line = json.dumps(record)
        print(line, flush=True)
        with args.out.open('a') as f:
            f.write(line + '\n')


if __name__ == '__main__':
    main()
