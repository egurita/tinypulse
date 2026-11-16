"""Neural-network tests: model regression (LOSO), INT8 C vs NumPy bit-exactness,
float vs INT8 accuracy loss, and generated weights in sync with the saved model."""

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import needs_dataset
from ppg_hr import features
from ppg_hr import firmware as fw
from ppg_hr.data import ACTIVITY_NAMES, REPO_ROOT, SUBJECTS, load_subject
from ppg_hr.dsp import hmm_track_step
from ppg_hr.evaluate import window_activity
from ppg_hr.nn import SpecCNN, spec_input
from ppg_hr.pipeline import acc_counts, ppg_counts
from ppg_hr.quant import hmm_track_int, int_forward, logits_to_logp_q16, quantize_input

MODELS = REPO_ROOT / "models" / "stage4" / "spec"
spec = importlib.util.spec_from_file_location("gen_nn", REPO_ROOT / "python" / "scripts" / "gen_nn_model.py")
gen_nn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen_nn)


@pytest.fixture(scope="module")
def qmodel():
    return gen_nn.load_npz(MODELS / "qmodel_S5.npz")


# ---------------------------------------------------------------- no dataset needed
@pytest.mark.firmware
def test_c_int8_network_bit_exact_on_random_inputs(qmodel):
    x = np.random.default_rng(1).integers(0, 128, (300, 4, 113)).astype(np.int8)
    c = np.stack([fw.nn_forward_q(xi) for xi in x])
    np.testing.assert_array_equal(c, int_forward(qmodel, x))


@pytest.mark.firmware
def test_c_int8_network_saturated_inputs(qmodel):
    for v in (0, 127):
        x = np.full((1, 4, 113), v, np.int8)
        np.testing.assert_array_equal(fw.nn_forward_q(x[0]), int_forward(qmodel, x)[0])


def test_generated_c_weights_match_saved_model(qmodel):
    """firmware/gen/nn_model.c must be the export of models/.../qmodel_S5.npz."""
    text = (REPO_ROOT / "firmware" / "gen" / "nn_model.c").read_text()
    for i, name in enumerate(["nn_w1", "nn_w2", "nn_w3", "nn_wh"]):
        body = text.split(f"{name}[")[1].split("{")[1].split("}")[0]
        vals = np.array([int(v) for v in body.replace("\n", "").split(",")])
        np.testing.assert_array_equal(vals, qmodel.layers[i].w.ravel())


# ---------------------------------------------------------------- dataset
pytestmark_ds = [pytest.mark.dataset, needs_dataset]


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@pytest.fixture(scope="module")
def loso():
    """Re-run every LOSO fold from its saved weights on its held-out subject."""
    est, hr, act = {}, {}, {}
    for s in SUBJECTS:
        m = SpecCNN()
        m.load_state_dict(torch.load(MODELS / f"fold_{s}.pt"))
        m.eval()
        with torch.no_grad():
            logits = m(torch.from_numpy(spec_input(features.load(s)["spec"]))).numpy()
        score, out = None, []
        for p in softmax(logits):
            score = hmm_track_step(score, p, 2, 0.5)
            out.append(np.argmax(score))
        subj = load_subject(s)
        est[s], hr[s], act[s] = features.bin_to_hr(np.array(out)), subj.hr_ref, window_activity(subj)
    return est, hr, act


NN_LIMITS = {  # LOSO, all 15 subjects: measured (reports/stage4_results.md) + ~10%
    "sitting": 3.5,        # 3.1
    "stairs": 15.5,        # 13.9
    "table soccer": 13.5,  # 12.0
    "cycling": 11.0,       # 10.0
    "driving": 4.5,        # 3.9
    "lunch": 4.5,          # 3.9
    "walking": 8.5,        # 7.7
    "working": 3.5,        # 3.0
}


@pytest.mark.dataset
@needs_dataset
def test_nn_model_regression_per_activity(loso, metrics):
    est, hr, act = loso
    err = np.concatenate([np.abs(est[s] - hr[s]) for s in SUBJECTS])
    a = np.concatenate([act[s] for s in SUBJECTS])
    mae = {ACTIVITY_NAMES[k]: float(err[a == k].mean()) for k in ACTIVITY_NAMES if k}
    metrics.record("nn_loso_mae_by_activity", mae)
    bad = {k: (round(mae[k], 2), lim) for k, lim in NN_LIMITS.items() if mae[k] > lim}
    assert not bad, bad


@pytest.mark.dataset
@needs_dataset
def test_nn_beats_dsp_loso(loso, metrics):
    est, hr, _ = loso
    nn = np.mean([np.abs(est[s] - hr[s]).mean() for s in SUBJECTS])
    dsp = np.mean([np.abs(fw.run_subject(load_subject(s)) - hr[s]).mean() for s in SUBJECTS])
    metrics.record("nn_loso_mae", float(nn))
    metrics.record("dsp_mae_all15", float(dsp))
    assert nn < dsp - 1.0


@pytest.mark.dataset
@pytest.mark.firmware
@needs_dataset
def test_firmware_nn_mode_bit_exact_with_numpy_int8_pipeline(qmodel):
    """Full recording through the C firmware in NN mode == NumPy integer reference fed
    with the firmware's own spectra (network + tracker)."""
    s = load_subject("S5")
    bins, spec_c = fw.run_counts(ppg_counts(s), acc_counts(s), len(s.hr_ref) + 8, fw.MODE_NN, with_spectra=True)
    ref = hmm_track_int(logits_to_logp_q16(qmodel, int_forward(qmodel, quantize_input(spec_c))))
    np.testing.assert_array_equal(bins, ref)


@pytest.mark.dataset
@needs_dataset
def test_int8_accuracy_loss_is_small(qmodel, metrics):
    """Float PyTorch model + float tracker vs INT8 integer pipeline, held-out S5."""
    s = load_subject("S5")
    f = features.load("S5")
    m = SpecCNN()
    m.load_state_dict(torch.load(MODELS / "fold_S5.pt"))
    m.eval()
    with torch.no_grad():
        logits = m(torch.from_numpy(spec_input(f["spec"]))).numpy()
    score, out = None, []
    for p in softmax(logits):
        score = hmm_track_step(score, p, 2, 0.5)
        out.append(np.argmax(score))
    fl = features.bin_to_hr(np.array(out))
    xq = quantize_input(np.round(f["spec"] * 2**20).astype(np.int64))
    q8 = features.bin_to_hr(hmm_track_int(logits_to_logp_q16(qmodel, int_forward(qmodel, xq))))
    gap = abs(np.abs(q8 - s.hr_ref).mean() - np.abs(fl - s.hr_ref).mean())
    same = float(np.mean(q8 == fl))
    metrics.record("int8_vs_float_identical_S5", same)
    metrics.record("int8_vs_float_mae_gap_S5", float(gap))
    # Measured: 86% identical windows, +0.19 BPM. Most differences come from the 7-bit
    # input alone (float model + quantized input: 90% identical, no MAE cost): the
    # tracker turns tiny score changes into different, equally good paths. So the
    # accuracy gap is the real criterion; the identical-window share is a sanity bound.
    assert gap < 0.5 and same > 0.8
