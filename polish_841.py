#!/usr/bin/env python3
import os, gc, math, argparse
import numpy as np
import torch


def load_X_from_gram(path, d, device, dtype):
    G = np.loadtxt(path)
    G = 0.5 * (G + G.T)

    w, V = np.linalg.eigh(G)
    idx = np.argsort(w)[::-1][:d]
    w = np.maximum(w[idx], 0.0)
    V = V[:, idx]

    X = V * np.sqrt(w)[None, :]
    X = torch.tensor(X, device=device, dtype=dtype)
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    X.requires_grad_(True)
    return X


def maxdot_stats(X):
    Z = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    G = Z @ Z.T
    N = G.shape[0]
    iu = torch.triu_indices(N, N, offset=1, device=G.device)
    dots = G[iu[0], iu[1]]
    return dots.max().item(), dots.mean().item(), G


def riesz_log_loss(X, s):
    Z = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    G = Z @ Z.T
    N = G.shape[0]
    iu = torch.triu_indices(N, N, offset=1, device=G.device)

    dots = G[iu[0], iu[1]]
    dist2 = (2.0 - 2.0 * dots).clamp_min(1e-14)

    log_terms = (-0.5 * s) * torch.log(dist2)
    return torch.logsumexp(log_terms, dim=0)


def parse_schedule(s):
    out = []
    s = s.strip().replace("\n", "").replace(" ", "")
    for block in s.split(","):
        if not block:
            continue
        ss, steps, lr = block.split(":")
        out.append((float(ss), int(steps), float(lr)))
    return out


@torch.no_grad()
def save_result(X, outpath):
    Z = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    G = Z @ Z.T
    np.savetxt(outpath, G.detach().cpu().numpy(), fmt="%.12f")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("candidate", type=str, help="Saved candidate Gram matrix txt")
    p.add_argument("--d", type=int, default=12)
    p.add_argument("--outdir", type=str, default="polished")
    p.add_argument("--prefix", type=str, default="polished")
    p.add_argument("--print-every", type=int, default=500)
    p.add_argument("--float32", action="store_true")
    p.add_argument(
        "--schedule",
        type=str,
        default=(
            "10240000:150000:0.000000005," 
            "20240000:150000:0.000000002," 
            "40240000:150000:0.000000001," 
            "80240000:150000:0.0000000005," 
            "160240000:150000:0.0000000002," 
        ),
    )

    args = p.parse_args()
    schedule = parse_schedule(args.schedule)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.float32 else torch.float64

    os.makedirs(args.outdir, exist_ok=True)

    X = load_X_from_gram(args.candidate, args.d, device, dtype)

    md, mean, _ = maxdot_stats(X)
    print(f"loaded candidate | maxdot={md:.12f} mean={mean:.12f}", flush=True)

    opt = torch.optim.Adam([X], lr=schedule[0][2])#torch.optim.SGD([X],lr=schedule[0][2],momentum=0.9,nesterov=True,)

    total_step = 0
    best_md = md
    best_G = None

    for s, steps, lr in schedule:
        for group in opt.param_groups:
            group["lr"] = lr/10

        print(f"\n===== s={s:g} | lr={lr:g} | steps={steps} =====", flush=True)

        for _ in range(steps):
            loss = riesz_log_loss(X, s)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                X[:] = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)

            total_step += 1

            if total_step % args.print_every == 0:
                md, mean, G = maxdot_stats(X)

                if md < best_md:
                    best_md = md
                    best_G = G.detach().cpu().numpy()

                print(
                    f"step {total_step:08d} "
                    f"s={s:g} lr={lr:g} "
                    f"logE={loss.item():.12f} "
                    f"maxdot={md:.12f} mean={mean:.12f} "
                    f"best={best_md:.12f}",
                    flush=True,
                )

    md, mean, G = maxdot_stats(X)
    if md < best_md or best_G is None:
        best_md = md
        best_G = G.detach().cpu().numpy()

    base = os.path.splitext(os.path.basename(args.candidate))[0]
    outpath = os.path.join(args.outdir, f"{args.prefix}_{base}_maxdot{best_md:.12f}.txt")
    np.savetxt(outpath, best_G, fmt="%.12f")

    print(f"\nfinished | final maxdot={md:.12f} mean={mean:.12f}")
    print(f"best maxdot={best_md:.12f}")
    print(f"saved {outpath}")

    del X, opt
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()