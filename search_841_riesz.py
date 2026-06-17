#!/usr/bin/env python3
import os, gc, math, argparse, itertools
import numpy as np
import torch


def even_signs_5(device, dtype):
    out = []
    for s in itertools.product([-1.0, 1.0], repeat=5):
        if math.prod(s) == 1.0:
            out.append(s)
    return torch.tensor(out, device=device, dtype=dtype)


def build_clebsch_48(device, dtype):
    eps = even_signs_5(device, dtype)

    x = torch.empty(16, 5, device=device, dtype=dtype)
    x[:, :4] = (math.sqrt(2) / 3.0) * eps[:, :4]
    x[:, 4] = (1.0 / 3.0) * eps[:, 4]

    b = torch.empty(16, 5, device=device, dtype=dtype)
    b[:, :4] = -(math.sqrt(2) / 4.0) * eps[:, :4]
    b[:, 4] = -0.5 * eps[:, 4]

    equator = torch.cat([torch.zeros(16, 1, device=device, dtype=dtype), x], dim=1)
    upper = torch.cat([0.5 * torch.ones(16, 1, device=device, dtype=dtype), b], dim=1)
    lower = torch.cat([-0.5 * torch.ones(16, 1, device=device, dtype=dtype), b], dim=1)

    C48 = torch.cat([equator, upper, lower], dim=0)
    return C48 / C48.norm(dim=1, keepdim=True).clamp_min(1e-12)


def build_intervectors(device, dtype):
    one_factorization = [
        [(0, 1), (2, 3), (4, 5)],
        [(0, 2), (1, 4), (3, 5)],
        [(0, 3), (1, 5), (2, 4)],
        [(0, 4), (1, 3), (2, 5)],
        [(0, 5), (1, 2), (3, 4)],
    ]

    vecs = []
    for matching in one_factorization:
        for a, b in matching:
            for c, d in matching:
                supp = [a, b, 6 + c, 6 + d]
                for signs in itertools.product([-1.0, 1.0], repeat=4):
                    v = torch.zeros(12, device=device, dtype=dtype)
                    for idx, sgn in zip(supp, signs):
                        v[idx] = 0.5 * sgn
                    vecs.append(v)

    V = torch.stack(vecs, dim=0)
    assert V.shape == (720, 12)
    return V


def build_clebsch_840(device, dtype):
    C48 = build_clebsch_48(device, dtype)

    I = torch.eye(6, device=device, dtype=dtype)
    orts = torch.cat([I, -I], dim=0)

    C60 = torch.cat([C48, orts], dim=0)
    C60 = C60 / C60.norm(dim=1, keepdim=True).clamp_min(1e-12)

    left = torch.cat([C60, torch.zeros(60, 6, device=device, dtype=dtype)], dim=1)
    right = torch.cat([torch.zeros(60, 6, device=device, dtype=dtype), C60], dim=1)
    inter = build_intervectors(device, dtype)

    C840 = torch.cat([left, right, inter], dim=0)
    C840 = C840 / C840.norm(dim=1, keepdim=True).clamp_min(1e-12)

    assert C840.shape == (840, 12)
    return C840


def init_batch(C840, B, device, dtype):
    X = torch.empty(B, 841, 12, device=device, dtype=dtype)
    X[:, :840, :] = C840[None, :, :]

    signs = torch.randint(0, 2, (B, 12), device=device)
    signs = signs.to(dtype=dtype) * 2.0 - 1.0
    X[:, 840, :] = signs / math.sqrt(12.0)

    X.requires_grad_(True)
    return X


def riesz_log_loss_and_stats(X, s):
    Z = X / X.norm(dim=2, keepdim=True).clamp_min(1e-12)
    G = torch.bmm(Z, Z.transpose(1, 2))

    N = G.shape[1]
    iu = torch.triu_indices(N, N, offset=1, device=G.device)
    dots = G[:, iu[0], iu[1]]

    maxdot = dots.max(dim=1).values

    dist2 = (2.0 - 2.0 * dots).clamp_min(1e-12)

    # log(sum ||z_i-z_j||^{-s})
    # = logsumexp((-s/2) * log(2 - 2 dot))
    log_terms = (-0.5 * s) * torch.log(dist2)
    log_energy = torch.logsumexp(log_terms, dim=1)

    return log_energy.mean(), maxdot, G


@torch.no_grad()
def save_candidates(X, outdir, prefix, start_id, threshold):
    os.makedirs(outdir, exist_ok=True)

    Z = X / X.norm(dim=2, keepdim=True).clamp_min(1e-12)
    G = torch.bmm(Z, Z.transpose(1, 2))

    N = G.shape[1]
    iu = torch.triu_indices(N, N, offset=1, device=G.device)
    maxdot = G[:, iu[0], iu[1]].max(dim=1).values

    saved = 0
    for b in range(G.shape[0]):
        md = maxdot[b].item()
        if md < threshold:
            idx = start_id + saved

            gram_fname = os.path.join(
                outdir,
                f"{prefix}{idx:06d}_gram_maxdot{md:.10f}.txt",
            )
            coord_fname = os.path.join(
                outdir,
                f"{prefix}{idx:06d}_coord_maxdot{md:.10f}.txt",
            )

            np.savetxt(
                gram_fname,
                G[b].detach().cpu().numpy(),
                fmt="%.10f",
            )
            np.savetxt(
                coord_fname,
                Z[b].detach().cpu().numpy(),
                fmt="%.10f",
            )

            saved += 1

    return saved, maxdot.min().item(), maxdot.mean().item(), maxdot.max().item()

def parse_schedule(schedule_str):
    schedule = []
    schedule_str = schedule_str.strip().replace("\n", "").replace(" ", "")

    for block in schedule_str.split(","):
        if not block:
            continue
        parts = block.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Bad schedule block {block!r}. "
                "Expected format s:steps:lr, e.g. 64:10000:0.001"
            )
        s_str, steps_str, lr_str = parts
        schedule.append((float(s_str), int(steps_str), float(lr_str)))

    return schedule

def run_macro(args, macro_id, C840, device, dtype, counter):
    X = init_batch(C840, args.B, device, dtype)

    first_lr = args.schedule[0][2]
    opt = torch.optim.Adam([X], lr=first_lr)

    total_step = 0

    for s, stage_steps, lr in args.schedule:
        for group in opt.param_groups:
            group["lr"] = lr

        print(
            f"\n===== macro {macro_id:04d} | s={s:g} | lr={lr:g} | steps={stage_steps} =====",
            flush=True,
        )

        for _ in range(stage_steps):
            loss, maxdot, _ = riesz_log_loss_and_stats(X, s)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_step += 1

            if total_step % args.print_every == 0:
                print(
                    f"macro {macro_id:04d} "
                    f"step {total_step:07d} "
                    f"s={s:g} "
                    f"lr={lr:g} "
                    f"logE={loss.item():.12f} "
                    f"min={maxdot.min().item():.10f} "
                    f"mean={maxdot.mean().item():.10f} "
                    f"max={maxdot.max().item():.10f}",
                    flush=True,
                )

    saved, mn, mean, mx = save_candidates(
        X.detach(),
        args.outdir,
        args.prefix,
        counter,
        args.save_threshold,
    )

    print(
        f"\nmacro {macro_id:04d} finished | "
        f"saved={saved} | "
        f"final min={mn:.10f} mean={mean:.10f} max={mx:.10f}",
        flush=True,
    )

    del X, opt
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return counter + saved


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--N", type=int, default=841)
    parser.add_argument("--d", type=int, default=12)

    parser.add_argument("--B", type=int, default=512)
    parser.add_argument("--macro-repeats", type=int, default=100)

    parser.add_argument(
        "--schedule",
        type=str,
        default=(
            "8:1000:0.005,"
            "16:1000:0.003,"
            "32:1000:0.002,"
            "64:2000:0.001,"
            "128:2000:0.0005,"
            "256:2000:0.0002,"
            "512:2000:0.0001,"
            "1024:4000:0.00005,"
            "2048:4000:0.00001,"
            "4096:4000:0.00001,"
            "10000:4000:0.000005,"
            "20000:4000:0.000001,"
            "40000:4000:0.000001,"
        ),
        help="Format: s:steps:lr,s:steps:lr,...",
    )

    parser.add_argument("--save-threshold", type=float, default=0.501)
    parser.add_argument("--outdir", type=str, default="candidatesG12")
    parser.add_argument("--prefix", type=str, default="candidateG12")

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--float32", action="store_true")

    args = parser.parse_args()
    args.schedule = parse_schedule(args.schedule)

    assert args.N == 841
    assert args.d == 12
    
    import random
    
    if args.seed is None:
        seed = int.from_bytes(os.urandom(8), "little")
    else:
        seed = args.seed
    
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)
    print(f"seed={seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.float32 else torch.float64

    print(f"device={device}, dtype={dtype}")
    print(f"B={args.B}, macro_repeats={args.macro_repeats}")
    print("Riesz schedule:")
    for s, steps, lr in args.schedule:
        print(f"  s={s:g}, steps={steps}, lr={lr:g}")

    C840 = build_clebsch_840(device, dtype)

    with torch.no_grad():
        G840 = C840 @ C840.T
        G840.fill_diagonal_(-999.0)
        print(f"base 840 max dot = {G840.max().item():.12f}")

    counter = 0
    for macro_id in range(args.macro_repeats):
        counter = run_macro(args, macro_id, C840, device, dtype, counter)

    print(f"Done. Total candidates saved: {counter}")


if __name__ == "__main__":
    main()