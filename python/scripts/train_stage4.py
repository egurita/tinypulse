"""Leave-one-subject-out training of the HR networks.

For every subject T (15 folds): test = T, validation = the next subject (early stopping /
epoch selection only), train = the remaining 13. No window of T is ever seen during
training or model selection, and inputs are normalised per window (no dataset-level
statistics), so nothing about T leaks into its model.

Saves models/stage4/<arch>/fold_<T>.pt and the held-out logits pred_<T>.npy.
Usage:  uv run python python/scripts/train_stage4.py spec|time [--epochs N] [--folds S1,S5]
"""

import argparse
import json
import time

import numpy as np
import torch
from torch import nn

from ppg_hr import features
from ppg_hr.data import REPO_ROOT, SUBJECTS, load_subject
from ppg_hr.nn import ARCHS, soft_target, spec_input, time_input

MODELS = REPO_ROOT / "models" / "stage4"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def subject_data(arch):
    data = {}
    for sid in SUBJECTS:
        f = features.load(sid)
        x = spec_input(f["spec"]) if arch == "spec" else time_input(f["time"])
        hr = load_subject(sid).hr_ref.astype(np.float32)
        data[sid] = (x, hr)
    return data


def mae_argmax(logits, hr):
    return float(np.abs(features.bin_to_hr(logits.argmax(1)) - hr).mean())


def predict(model, x, bs=2048):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out.append(model(torch.from_numpy(x[i:i + bs]).to(DEV)).float().cpu().numpy())
    return np.concatenate(out)


def train_fold(arch, data, test_sid, epochs, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    i = SUBJECTS.index(test_sid)
    val_sid = SUBJECTS[(i + 1) % len(SUBJECTS)]
    train = [s for s in SUBJECTS if s not in (test_sid, val_sid)]
    x = np.concatenate([data[s][0] for s in train])
    y = soft_target(np.concatenate([data[s][1] for s in train]))
    xv, hv = data[val_sid]

    model = ARCHS[arch]().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    steps = epochs * int(np.ceil(len(x) / 256))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    xt, yt = torch.from_numpy(x).to(DEV), torch.from_numpy(y).to(DEV)
    best, best_state, best_ep = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), device=DEV)
        for j in range(0, len(xt), 256):
            idx = perm[j:j + 256]
            loss = nn.functional.cross_entropy(model(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        v = mae_argmax(predict(model, xv), hv)
        if v < best:
            best, best_ep = v, ep
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, {"val_subject": val_sid, "val_mae": best, "best_epoch": best_ep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arch", choices=list(ARCHS))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--folds", default=",".join(SUBJECTS))
    args = ap.parse_args()
    out = MODELS / args.arch
    out.mkdir(parents=True, exist_ok=True)
    data = subject_data(args.arch)
    log = json.loads((out / "train_log.json").read_text()) if (out / "train_log.json").exists() else {}
    for sid in args.folds.split(","):
        t = time.perf_counter()
        model, info = train_fold(args.arch, data, sid, args.epochs)
        logits = predict(model, data[sid][0])
        info["test_mae_argmax"] = mae_argmax(logits, data[sid][1])
        info["train_seconds"] = time.perf_counter() - t
        torch.save(model.cpu().state_dict(), out / f"fold_{sid}.pt")
        np.save(out / f"pred_{sid}.npy", logits.astype(np.float32))
        log[sid] = info
        (out / "train_log.json").write_text(json.dumps(log, indent=2))
        print(f"{args.arch} fold {sid}: test MAE {info['test_mae_argmax']:.2f} (val {info['val_subject']} "
              f"{info['val_mae']:.2f}, epoch {info['best_epoch']}) {info['train_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
