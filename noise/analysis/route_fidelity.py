"""Route fidelity along training, for any architecture (analysis only): at each saved checkpoint of a run, the
per-token cosine between the attention-cut credit restricted to each token's top-k MLP units and the unrestricted
credit, both by autograd (analysis/test_boostcut.py's reference), at the raw embedding and every mid stream, averaged
over the tokens of one validation sequence. Also each block's median count of units carrying 90% of a token's MLP
output (by |activation| times the unit's output norm).

    python analysis/route_fidelity.py --run plain_bc38_s0 --arch plain --ks 64,128,256
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('test_boostcut', ROOT / 'analysis/test_boostcut.py')
tb = sys.modules['test_boostcut'] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tb)
bc = tb.bc
from utils.model import norm  # noqa: E402


def per_token(a, b):
    keep = (a.norm(dim=-1) > 0) & (b.norm(dim=-1) > 0)
    return F.cosine_similarity(a[keep], b[keep], dim=-1).mean().item()


@torch.no_grad()
def k90(model, x):
    """Per block, the median over tokens of how many units carry 90% of the token's MLP output."""
    T = x.shape[1]
    cos_sin = (model.cos[:, :T], model.sin[:, :T])
    h = norm(model.transformer.wte(x).float())
    x0, out = h, []
    for i, block in enumerate(model.transformer.h):
        h = model.resid_lambdas[i] * h + model.x0_lambdas[i] * x0
        ve = model.value_embeds[str(i)](x).float() if str(i) in model.value_embeds else None
        h = h + block.attn(norm(h), ve, cos_sin, model.window_sizes[i]).float()
        r = tb.act(F.linear(norm(h), block.mlp.c_fc.weight.float()))
        share = (r.abs() * block.mlp.c_proj.weight.float().norm(dim=0)).sort(-1, descending=True).values
        cum = share.cumsum(-1) / share.sum(-1, keepdim=True)
        out.append(int((cum < 0.9).sum(-1).add(1).float().median().item()))
        h = h + F.linear(r, block.mlp.c_proj.weight.float())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--runs-dir', type=Path, default=Path('/data/runs'))
    parser.add_argument('--run', required=True)
    parser.add_argument('--arch', choices=('noise', 'plain'), default='noise')
    parser.add_argument('--preset', default='38m')
    parser.add_argument('--ks', default='64,128,256')
    args = parser.parse_args()
    if args.arch == 'plain':
        spec_p = importlib.util.spec_from_file_location('plain_gpt', ROOT / 'analysis/plain_gpt.py')
        tb.PLAIN = sys.modules['plain_gpt'] = importlib.util.module_from_spec(spec_p)
        spec_p.loader.exec_module(tb.PLAIN)
        tb.PLAIN.patch(bc)
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda', 0)
    n_layer, bc.WIDTH = bc.PRESETS[args.preset]
    bc.HIDDEN = 4 * bc.WIDTH
    config = bc.GPTConfig(n_layer=n_layer, n_head=bc.WIDTH // 64, n_kv_head=bc.WIDTH // 64, n_embd=bc.WIDTH)
    rows = bc.load_split(args.data_dir / 'val.pt')[:1].to(device)
    x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
    run = args.runs_dir / args.run
    ckpts = sorted(run.glob('ckpt_*.pt'), key=lambda p: int(p.stem.split('_')[1]))
    for path in ckpts:
        model = bc.make_model(device, 0, config)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        full = tb.cut_reference(model, x, y, bc.HIDDEN)
        sites = lambda ref: [ref['emb']] + ref['mids']
        record = dict(run=args.run, seqs=int(path.stem.split('_')[1]), k90=k90(model, x), fidelity={})
        for k in (int(k) for k in args.ks.split(',')):
            ref = tb.cut_reference(model, x, y, k)
            record['fidelity'][k] = min(per_token(a, b) for a, b in zip(sites(ref), sites(full)))
        print(json.dumps(record), flush=True)
        del model, full
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
