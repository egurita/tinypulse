"""Evaluation: MAE (BPM) vs the ECG reference, per activity and per subject."""

from __future__ import annotations

import numpy as np

from .data import ACTIVITY_NAMES, Subject
from .pipeline import Config, estimate_hr


def window_activity(subj: Subject) -> np.ndarray:
    """Activity label at the centre of each reference window."""
    return subj.activity_at(subj.hr_ref_times()).astype(int)


def run(subjects: list[Subject], cfg: Config) -> dict[str, dict[str, np.ndarray]]:
    """Per subject: estimates, reference, absolute error and window activity."""
    res = {}
    for s in subjects:
        est = estimate_hr(s, cfg)
        res[s.sid] = {"est": est, "ref": s.hr_ref.astype(float), "err": np.abs(est - s.hr_ref),
                      "act": window_activity(s)}
    return res


def mae_by_activity(res: dict, sids: list[str]) -> dict[str, float]:
    """Pooled MAE over all windows of `sids`, per activity (transient windows excluded
    from the activity rows but included in 'all')."""
    err = np.concatenate([res[s]["err"] for s in sids])
    act = np.concatenate([res[s]["act"] for s in sids])
    out = {name: float(err[act == a].mean()) for a, name in ACTIVITY_NAMES.items()
           if a != 0 and (act == a).any()}
    out["all windows"] = float(err.mean())
    return out


def mae_subject_mean(res: dict, sids: list[str]) -> float:
    """Mean over subjects of each subject's MAE (the PPG-DaLiA paper's headline metric)."""
    return float(np.mean([res[s]["err"].mean() for s in sids]))
