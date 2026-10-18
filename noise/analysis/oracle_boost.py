"""Stagewise ("boosted") residual stream: the loss ceiling of training every sub-block against its own readout
(analysis only: autograd prices an objective whose credit a forward-only method can measure exactly and cheaply).

The shared head reads the stream at every depth (logit lens). Here each sub-block (an attention or an MLP) adds a
correction to the stream and is trained only to lower the readout loss right after it; the stream below it is
detached, so no credit ever crosses a sub-block. That is gradient boosting in function space, with all stages trained
at once. Every readout is a per-token function of the stream, so a forward-only method gets its credit exactly by
probing the stream with a full orthogonal cycle at that depth, without reruns and without cross-token interference.

  --unit sub|block   readouts after every sub-block (16) or every block (8)
  --emb bigram|inject|block0|global
                     what trains the token embedding: its own readout (a bigram model), the local credit at every
                     block input it is mixed into, only block 0's input, or the true end-to-end gradient (reference)
  --head final|all   the head learns from the final readout only, or from every readout
  --scalars local|frozen   the residual mixing scalars learn from their block's local loss, or stay at init

    python analysis/oracle_boost.py --out /data/runs/boost_sub_bigram --unit sub --emb bigram
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('boostcut', ROOT / 'train/boostcut/run.py')
bc = importlib.util.module_from_spec(spec)  # the record's model init and optimizer (make_model, make_optimizer, Recipe)
spec.loader.exec_module(bc)
from utils.eval import evaluate  # noqa: E402
from utils.model import GPTConfig, norm  # noqa: E402
from utils.prepare_data import TRAIN_ROWS, load_rows  # noqa: E402


SAMPLED = {'size': 0, 'negatives': None}
SURROGATE = {'cap': True, 'norm': 'exact'}  # intermediate readouts: soft cap or not; exact or frozen normalizer
# per-stage readout temperatures (log scale), part of the training objective only: each intermediate readout's
# logits are scaled by exp(t_s) before the soft cap. None: off. 'i' counts the stages within a forward.
STAGE = {'temps': None, 'i': 0}


def readout(model, x, y, train_head, final=False):
    """The head's loss reading streams x. Intermediate readouts can use a sampled softmax (SAMPLED['size'] > 0):
    the target against a shared set of uniformly drawn negatives, weighted by (V - 1) / size, redrawn every step."""
    weight = model.lm_head.weight if train_head else model.lm_head.weight.detach()
    vocab = model.config.vocab_size
    if final or not SAMPLED['size']:
        if final or (SURROGATE['cap'] and SURROGATE['norm'] == 'exact'):
            logits = F.linear(norm(x), weight)[..., :vocab].float()
            if not final and STAGE['temps'] is not None:
                logits = logits * STAGE['temps'][STAGE['i']].exp()
                STAGE['i'] += 1
            logits = 15 * torch.tanh(logits / 15)
        else:
            if SURROGATE['norm'] == 'frozen':  # the clean stream's RMS, held fixed
                h = x / x.float().square().mean(-1, keepdim=True).add(torch.finfo(torch.float32).eps).sqrt().detach().to(x.dtype)
            else:
                h = norm(x)
            logits = F.linear(h, weight)[..., :vocab].float()
            if SURROGATE['cap']:
                logits = 15 * torch.tanh(logits / 15)
        return F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-1)
    neg = SAMPLED['negatives']
    h = norm(x).flatten(0, 1)
    yf = y.flatten()
    z_neg = 15 * torch.tanh(F.linear(h, weight[neg]).float() / 15)  # (n, S)
    z_y = 15 * torch.tanh((h * weight[yf.clamp(min=0)]).sum(-1).float() / 15)
    z_neg = z_neg + math.log((vocab - 1) / len(neg))
    z_neg = z_neg.masked_fill(neg.unsqueeze(0) == yf.unsqueeze(1), float('-inf'))  # the target is never a negative
    lse = torch.logsumexp(torch.cat([z_y.unsqueeze(1), z_neg], 1), 1)
    valid = (yf >= 0).float()
    return ((lse - z_y) * valid).sum() / valid.sum()


def attention_output(attn, x, ve, cos_sin, window):
    """CausalSelfAttention.forward up to (not including) the output projection."""
    from utils.model import apply_rotary_emb, attention
    B, T, C = x.size()
    q = attn.c_q(x).view(B, T, attn.n_head, attn.head_dim)
    k = attn.c_k(x).view(B, T, attn.n_kv_head, attn.head_dim)
    v = attn.c_v(x).view(B, T, attn.n_kv_head, attn.head_dim)
    if ve is not None:
        ve = ve.view(B, T, attn.n_kv_head, attn.head_dim)
        gate = 2 * torch.sigmoid(attn.ve_gate(x[..., :attn.ve_gate_channels]))
        v = v + gate.unsqueeze(-1) * ve
    cos, sin = cos_sin
    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
    q, k = norm(q), norm(k)
    return attention(q, k, v, window[0]).contiguous().view(B, T, -1)


def own_path_attention(attn, x_live, x_det, ve, cos_sin, window):
    """The attention module with credit only along each token's own path: queries and the token's own key and
    value come from x_live, every other position's key and value from x_det (detached). Dense, for analysis."""
    from utils.model import apply_rotary_emb
    B, T, C = x_live.size()
    h, d = attn.n_head, attn.head_dim
    cos, sin = cos_sin

    def kv(x):
        k = attn.c_k(x).view(B, T, h, d)
        v = attn.c_v(x).view(B, T, h, d)
        if ve is not None:
            gate = 2 * torch.sigmoid(attn.ve_gate(x[..., :attn.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve.view(B, T, h, d)
        return norm(apply_rotary_emb(k, cos, sin)).transpose(1, 2), v.transpose(1, 2)

    q = norm(apply_rotary_emb(attn.c_q(x_live).view(B, T, h, d), cos, sin)).transpose(1, 2)
    k_live, v_live = kv(x_live)
    k_det, v_det = kv(x_det)
    scale = d ** -0.5
    logits = (q @ k_det.transpose(-1, -2)) * scale  # B, h, T, T
    own = (q * k_live).sum(-1) * scale
    eye = torch.eye(T, device=x_live.device, dtype=torch.bool)
    logits = torch.where(eye, own.unsqueeze(-1).expand_as(logits), logits)
    pos = torch.arange(T, device=x_live.device)
    lag = pos[:, None] - pos[None, :]
    logits = logits.masked_fill(~((lag >= 0) & (lag <= window[0])), float('-inf'))
    A = logits.float().softmax(-1).to(v_det.dtype)
    out = A @ v_det + A.diagonal(dim1=-2, dim2=-1).unsqueeze(-1) * (v_live - v_det)
    return attn.c_proj(out.transpose(1, 2).reshape(B, T, C))


def mlp_frozen(mlp, x):
    """The MLP with its weights held fixed (credit flows through it to its input only)."""
    return F.linear(F.relu(F.linear(x, mlp.c_fc.weight.detach())).square(), mlp.c_proj.weight.detach())


def local_forward(model, idx, y, args):
    """Sum of the readout losses; each sub-block's parameters see only the readout right after it."""
    STAGE['i'] = 0
    B, T = idx.shape
    cos_sin = (model.cos[:, :T], model.sin[:, :T])
    emb = norm(model.transformer.wte(idx))
    x0 = emb if args.emb in ('inject', 'block0', 'block0own') else emb.detach()
    losses = []
    if args.emb == 'bigram':
        losses.append(readout(model, emb, y, args.head == 'all'))
    x = x0
    final = None
    for i, block in enumerate(model.transformer.h):
        inject = x0 if args.emb == 'inject' or (args.emb in ('block0', 'block0own') and i == 0) else x0.detach()
        lam_r, lam_0 = model.resid_lambdas[i], model.x0_lambdas[i]
        if args.scalars == 'frozen':
            lam_r, lam_0 = lam_r.detach(), lam_0.detach()
        stream = x if i == 0 else x.detach()  # x0 itself at block 0 (detached unless the embedding learns there)
        xin = lam_r * stream + lam_0 * inject
        ve = model.value_embeds[str(i)](idx) if str(i) in model.value_embeds else None
        last = i == len(model.transformer.h) - 1
        if args.unit in ('mixed', 'mixed_rev'):
            # the attention's output projection and its internals learn from different readouts: 'mixed' trains
            # the projection through the block's MLP (block readout) and the internals at the attention's own
            # readout; 'mixed_rev' the other way round
            attn = block.attn
            y_att = attention_output(attn, norm(xin), ve, cos_sin, model.window_sizes[i])
            w = attn.c_proj.weight
            inner_block = args.unit == 'mixed_rev'
            # path 1: the internals' readout, with the projection frozen
            xmid_int = xin + F.linear(y_att, w.detach())
            # path 2: the projection's readout, with the internals' output detached
            xmid_proj = xin.detach() + F.linear(y_att.detach(), w)
            for xm, through_mlp in ((xmid_int, inner_block), (xmid_proj, not inner_block)):
                if through_mlp:
                    out = xm + mlp_frozen(block.mlp, norm(xm))
                    losses.append(readout(model, out, y, args.head == 'all'))
                else:
                    losses.append(readout(model, xm, y, args.head == 'all'))
            xmid = (xin + F.linear(y_att, w)).detach()
        elif i == 0 and args.emb == 'block0own':
            xin_det = xin.detach()
            xmid = xin + own_path_attention(block.attn, norm(xin), norm(xin_det), ve, cos_sin, model.window_sizes[i])
            if args.unit == 'sub':
                losses.append(readout(model, xmid, y, args.head == 'all'))
                xmid = xmid.detach()
        else:
            xmid = xin + block.attn(norm(xin), ve, cos_sin, model.window_sizes[i])
            if args.unit == 'sub':
                losses.append(readout(model, xmid, y, args.head == 'all'))
                xmid = xmid.detach()
        x = xmid + block.mlp(norm(xmid))
        if last:
            final = readout(model, x, y, True, final=True)
            losses.append(final)
        else:
            losses.append(readout(model, x, y, args.head == 'all'))
    return final, sum(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lr-mult', type=float, default=1.0)
    parser.add_argument('--unit', choices=('sub', 'block', 'mixed', 'mixed_rev'), default='sub')
    parser.add_argument('--emb', choices=('bigram', 'inject', 'block0', 'block0own', 'global'), default='bigram')
    parser.add_argument('--head', choices=('final', 'all'), default='final')
    parser.add_argument('--scalars', choices=('local', 'frozen'), default='local')
    parser.add_argument('--steps', type=int, default=0, help='Shorter schedule: this many updates (0: one epoch).')
    parser.add_argument('--seed', type=int, default=bc.SEED)
    parser.add_argument('--polar-margin', type=float, default=1.05, help='The record\'s, as this script always used.')
    parser.add_argument('--attn-cos', type=float, default=1.0,
                        help='Isotropic noise on every attention-internal gradient (q, k, v, gates, value embeddings) '
                             'to this cosine per tensor, to price imperfect local estimates.')
    parser.add_argument('--key-cos', type=float, default=1.0, help='The same, for the keys alone.')
    parser.add_argument('--freeze', default='', help='Comma list of name parts that keep their init (e.g. c_k,c_v).')
    parser.add_argument('--attn-every', type=int, default=1, help='Attention internals get fresh gradients every '
                        'this many steps and reuse the last ones in between.')
    parser.add_argument('--batch', type=int, default=0, help='Sequences per update (0: the record\'s 8).')
    parser.add_argument('--readout-cap', choices=('cap', 'nocap'), default='cap')
    parser.add_argument('--readout-norm', choices=('exact', 'frozen'), default='exact')
    parser.add_argument('--sampled', type=int, default=0, help='Intermediate readouts: sampled softmax over this many '
                        'shared uniform negatives per step (0: the full softmax).')
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda', 0)
    bc.POLAR_MARGIN = args.polar_margin
    recipe = bc.Recipe()
    recipe.lr *= args.lr_mult
    recipe.anvil_lr *= args.lr_mult
    model = bc.make_model(device, args.seed, GPTConfig())
    training = load_rows(args.data_dir / 'train.pt', TRAIN_ROWS)
    training = training[torch.randperm(len(training), generator=torch.Generator().manual_seed(bc.SEED + 1))]
    batch_size = args.batch or 8
    steps = args.steps or len(training) // batch_size
    optimizer = bc.make_optimizer(model, recipe, steps)
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / 'metrics.jsonl').open('a')
    noise_gen = torch.Generator(device).manual_seed(args.seed + 7)
    SURROGATE['cap'] = args.readout_cap == 'cap'
    SURROGATE['norm'] = args.readout_norm
    stale = {}
    start = time.perf_counter()
    for step in range(1, steps + 1):
        left = (steps - step + 1) / steps
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
            group['lr'] = group['initial_lr'] * min(1.0, left / recipe.warmdown)
        batch = training[(step - 1) * batch_size:step * batch_size].to(device)
        if args.sampled:
            SAMPLED['size'] = args.sampled
            SAMPLED['negatives'] = torch.randperm(model.config.vocab_size, device=device)[:args.sampled]
        xb, yb = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        model.requires_grad_(True)
        wte_grad = None
        if args.emb == 'global':
            with torch.autocast('cuda', dtype=torch.bfloat16):
                model(xb, yb).backward()
            wte_grad = model.transformer.wte.weight.grad.clone()
            model.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            final, total = local_forward(model, xb, yb, args)
        total.backward()
        model.requires_grad_(False)
        if wte_grad is not None:
            model.transformer.wte.weight.grad = wte_grad
        for name, p in model.named_parameters():  # the optimizers read float gradients, as Dust hands them
            g = torch.zeros_like(p, dtype=torch.float32) if p.grad is None else p.grad.float()
            if any(part and part in name for part in args.freeze.split(',')):
                g = torch.zeros_like(g)
            if args.attn_every > 1 and any(k in name for k in ('c_q', 'c_k', 'c_v', 've_gate', 'value_embeds')):
                if (step - 1) % args.attn_every == 0:
                    stale[name] = g.clone()
                g = stale[name]
            c = args.key_cos if 'c_k' in name else args.attn_cos if any(
                key in name for key in ('c_q', 'c_k', 'c_v', 've_gate', 'value_embeds')) else 1.0
            if c < 1 and g.norm() > 0:
                scale = g.norm() * ((1 / c ** 2 - 1) / g.numel()) ** 0.5
                g = g + scale * torch.randn(g.shape, device=g.device, generator=noise_gen)
            p.grad = g.to(p.dtype)
        frozen = {n: p.detach().clone() for n, p in model.named_parameters()
                  if any(part and part in n for part in args.freeze.split(','))}
        optimizer.step()
        with torch.no_grad():  # a true freeze: no decay either
            for n, p in model.named_parameters():
                if n in frozen:
                    p.copy_(frozen[n])
        optimizer.zero_grad(set_to_none=True)
        if step % 61 == 0 or step == steps:
            record = dict(step=step, train_loss=final.item(), local_sum=total.item(),
                          train_seconds=round(time.perf_counter() - start, 1))
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + '\n')
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    val = evaluate(state, args.data_dir)
    torch.save(state, args.out / 'final.pt')
    result = dict(val_loss=val, unit=args.unit, emb=args.emb, head=args.head, scalars=args.scalars,
                  lr_mult=args.lr_mult, steps=steps, attn_cos=args.attn_cos, key_cos=args.key_cos, freeze=args.freeze,
                  sampled=args.sampled)
    (args.out / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
