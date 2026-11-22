"""Live heart rate from the laptop webcam.

Window: camera image with the face box and skin regions, the pulse wave, and the
heart rate (updated every second from the last 20 s). Press q or Esc to quit.

    uv run python python/scripts/webcam_demo.py                      # live only
    uv run python python/scripts/webcam_demo.py --seconds 60 --save still --ref 68

--save stores ONLY per-frame averages (time, mean skin RGB, face box) in
data/webcam/<time>_<label>.npz (git-ignored); no image is ever written.
--ref stores a reference heart rate (smartwatch / counted pulse) for evaluation.
"""

import argparse
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from ppg_hr import rppg
from ppg_hr.data import FS_PPG, REPO_ROOT
from ppg_hr.dsp import bandpass, spectrum
from ppg_hr.pipeline import FINAL, NFFT, estimate_signal

OUT_DIR = REPO_ROOT / "data" / "webcam"
HISTORY_S, MIN_S, UPDATE_S = 20.0, 10.0, 1.0
BLUE, ORANGE, WHITE, GREY = (214, 120, 42), (52, 104, 235), (245, 245, 245), (150, 150, 150)


def estimate(t, rgb, box):
    """HR (BPM) from the recent history, and a quality score (peak share of band power)."""
    tu, x = rppg.resample(t, np.hstack([rgb, box]))
    sig = rppg.pos(x[:, :3])
    hr = estimate_signal(sig, rppg.motion_reference(x[:, 3:]), FINAL)
    f, mag = spectrum(bandpass(sig, FS_PPG)[-8 * FS_PPG:], FS_PPG, NFFT)
    band = (f >= 0.5) & (f <= 4.0)
    p = mag[band] ** 2
    k = np.argmin(np.abs(f[band] - hr[-1] / 60))
    quality = p[max(k - 2, 0):k + 3].sum() / p.sum()
    return float(hr[-1]), float(quality), bandpass(sig, FS_PPG)[-6 * FS_PPG:]


def draw_wave(img, wave, y0, h):
    if len(wave) < 2:
        return
    w = img.shape[1]
    v = (wave - wave.min()) / (np.ptp(wave) + 1e-12)
    pts = np.stack([np.linspace(10, w - 10, len(v)), y0 + h - v * h], 1).astype(np.int32)
    cv2.polylines(img, [pts], False, ORANGE, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = until q)")
    ap.add_argument("--save", metavar="LABEL", help="save per-frame averages under this label")
    ap.add_argument("--ref", type=float, help="reference heart rate (BPM) to store with --save")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, frame = cap.read()
    if not ok:
        print("Could not read from the camera. On macOS: System Settings -> Privacy & Security ->\n"
              "Camera -> enable your terminal app, then run this again.")
        return

    tracker = rppg.FaceTracker()
    hist = deque()                                    # (t, r, g, b, x, y, w, h)
    log_t, log_rgb, log_box, log_hr = [], [], [], []
    t0 = time.perf_counter()
    last_update, hr, quality, wave = 0.0, None, 0.0, np.zeros(0)
    n_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.perf_counter() - t0
        n_frames += 1
        face = tracker.detect(frame)
        rgb, box = np.full(3, np.nan), np.full(4, np.nan)
        if face is not None:
            rois = rppg.skin_rois(face)
            rgb, n_px = rppg.roi_mean_rgb(frame, rois)
            if n_px < 200:
                rgb = np.full(3, np.nan)              # too little skin visible
            box = face.box
            x, y, w, h = face.box.astype(int)
            cv2.rectangle(frame, (x, y), (x + w, y + h), GREY, 1)
            for x0, y0, x1, y1 in rois:
                cv2.rectangle(frame, (x0, y0), (x1, y1), BLUE, 2)
        hist.append((t, *rgb, *box))
        log_t.append(t); log_rgb.append(rgb); log_box.append(box)
        while hist and t - hist[0][0] > HISTORY_S:
            hist.popleft()

        a = np.array(hist)
        valid = np.isfinite(a[:, 1]).sum() if len(a) else 0
        if t - last_update >= UPDATE_S and len(a) and a[-1, 0] - a[0, 0] >= MIN_S and valid > 0.8 * len(a):
            hr, quality, wave = estimate(a[:, 0], a[:, 1:4], a[:, 4:8])
            log_hr.append((t, hr, quality))
            last_update = t

        # ---- overlay
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (30, 30, 30), -1)
        if face is None:
            msg, col = "no face - look at the camera", WHITE
        elif hr is None:
            msg, col = f"measuring... {max(0, MIN_S - (a[-1, 0] - a[0, 0])):.0f} s - keep still", WHITE
        else:
            msg, col = f"{hr:5.1f} BPM", (WHITE if quality > 0.3 else GREY)
        cv2.putText(frame, msg, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.3 if hr else 0.8, col, 2, cv2.LINE_AA)
        fps = n_frames / max(t, 1e-6)
        cv2.putText(frame, f"{fps:4.1f} fps  signal quality {quality:.2f}", (frame.shape[1] - 290, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1, cv2.LINE_AA)
        draw_wave(frame, wave, frame.shape[0] - 70, 55)
        cv2.imshow("webcam heart rate (q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or (args.seconds and t >= args.seconds):
            break

    cap.release()
    cv2.destroyAllWindows()
    dur = log_t[-1] if log_t else 0
    hrs = np.array([h for tt, h, _ in log_hr])
    print(f"\n{dur:.0f} s recorded at {len(log_t) / max(dur, 1e-6):.1f} fps")
    if len(hrs):
        print(f"YOUR HEART RATE: {np.median(hrs):.0f} BPM  (median of {len(hrs)} live readings, "
              f"last reading {hrs[-1]:.0f} BPM)\n")
    else:
        print("no heart-rate estimate (need at least 10 s with your face visible)\n")
    if args.save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{args.save}.npz"
        np.savez(path, t=np.array(log_t), rgb=np.array(log_rgb), box=np.array(log_box),
                 live_hr=np.array(log_hr).reshape(-1, 3), ref=np.nan if args.ref is None else args.ref,
                 label=args.save)
        print("saved", path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
