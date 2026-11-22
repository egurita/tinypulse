"""Remote PPG: heart rate from the colour of facial skin in webcam video.

    frame -> face box + 5 landmarks (YuNet) -> forehead + cheek ROIs, skin-masked
          -> mean RGB per frame -> resample to 64 Hz -> POS pulse signal
          -> the Stage 1 pipeline (bandpass, spectrum, motion mask/NLMS, HMM tracker)

The "accelerometer" here is the face's motion in the image (x, y, scale), which plays the
same role as the wrist ACC: a reference for motion artifacts.
Only per-frame averages are kept; no image is ever stored.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .data import FS_PPG, REPO_ROOT

DETECTOR_PATH = REPO_ROOT / "models" / "face" / "face_detection_yunet_2023mar.onnx"
DETECTOR_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
DETECT_WIDTH = 320          # detection runs on a downscaled frame (fast, enough for a face at ~50 cm)


@dataclass
class Face:
    box: np.ndarray          # x, y, w, h in full-frame pixels
    landmarks: np.ndarray    # (5, 2): right eye, left eye, nose tip, right mouth, left mouth
    score: float


class FaceTracker:
    def __init__(self, path=DETECTOR_PATH, score_threshold: float = 0.8):
        self.det = cv2.FaceDetectorYN.create(str(path), "", (DETECT_WIDTH, DETECT_WIDTH), score_threshold)

    def detect(self, frame: np.ndarray) -> Face | None:
        h, w = frame.shape[:2]
        scale = DETECT_WIDTH / w
        small = cv2.resize(frame, (DETECT_WIDTH, int(round(h * scale))))
        self.det.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.det.detect(small)
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda r: r[2] * r[3])          # largest face
        return Face(box=f[:4] / scale, landmarks=f[4:14].reshape(5, 2) / scale, score=float(f[14]))


def skin_rois(face: Face) -> list[tuple[int, int, int, int]]:
    """Forehead and both cheeks as (x0, y0, x1, y1), placed relative to the eyes so they
    follow the face's size and position. These areas are flat, well perfused and
    rarely covered by hair or moving during speech."""
    (rx, ry), (lx, ly), (nx, ny) = face.landmarks[0], face.landmarks[1], face.landmarks[2]
    d = float(np.hypot(lx - rx, ly - ry))                  # eye distance = face scale
    ey = (ry + ly) / 2
    x0e, x1e = min(rx, lx), max(rx, lx)
    rois = [
        (x0e - 0.1 * d, ey - 0.95 * d, x1e + 0.1 * d, ey - 0.4 * d),          # forehead
        (rx - 0.35 * d, ey + 0.35 * d, rx + 0.15 * d, ny + 0.2 * d),          # cheek (image left)
        (lx - 0.15 * d, ey + 0.35 * d, lx + 0.35 * d, ny + 0.2 * d),          # cheek (image right)
    ]
    return [tuple(int(round(v)) for v in r) for r in rois]


def skin_mask(patch_bgr: np.ndarray) -> np.ndarray:
    """Pixels whose chroma looks like skin (YCrCb box, a classic, illumination-tolerant rule)."""
    ycc = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycc[..., 1], ycc[..., 2]
    return (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)


def roi_mean_rgb(frame_bgr: np.ndarray, rois) -> tuple[np.ndarray, int]:
    """Mean R, G, B over the skin pixels of all ROIs, and the number of pixels used."""
    h, w = frame_bgr.shape[:2]
    acc, n = np.zeros(3), 0
    for x0, y0, x1, y1 in rois:
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 <= x0 or y1 <= y0:
            continue
        patch = frame_bgr[y0:y1, x0:x1]
        m = skin_mask(patch)
        acc += patch[m].sum(axis=0)[::-1]                 # BGR -> RGB
        n += int(m.sum())
    return (acc / n if n else np.full(3, np.nan)), n


def resample(t: np.ndarray, x: np.ndarray, fs: float = FS_PPG) -> tuple[np.ndarray, np.ndarray]:
    """Webcam frames arrive at ~30 fps with jitter; interpolate onto a uniform 64 Hz grid
    (the pipeline's rate). Missing values (no face) are bridged linearly."""
    t = np.asarray(t, float)
    x = np.asarray(x, float).reshape(len(t), -1)
    ok = np.all(np.isfinite(x), axis=1)
    tu = np.arange(t[ok][0], t[ok][-1], 1 / fs)
    return tu, np.stack([np.interp(tu, t[ok], x[ok, c]) for c in range(x.shape[1])], axis=1)


def pos(rgb: np.ndarray, fs: float = FS_PPG, win_s: float = 1.6) -> np.ndarray:
    """Plane-Orthogonal-to-Skin pulse extraction (Wang et al., IEEE TBME 2017).

    Over short overlapping windows: normalise each colour by its mean (removes skin tone
    and overall brightness), project onto two axes orthogonal to the skin-tone direction,
    and combine them with a ratio of their standard deviations so that intensity and
    specular changes (common to all colours) cancel while the blood-volume colour change
    remains. Overlap-add gives the pulse signal.
    """
    rgb = np.asarray(rgb, float)
    n, L = len(rgb), int(round(win_s * fs))
    P = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    h = np.zeros(n)
    for i in range(0, n - L + 1):
        c = rgb[i:i + L]
        cn = c / c.mean(axis=0)
        s = cn @ P.T                                       # (L, 2)
        sd1 = s[:, 1].std()
        p = s[:, 0] + (s[:, 0].std() / sd1 if sd1 > 0 else 0.0) * s[:, 1]
        h[i:i + L] += p - p.mean()
    return h


def green(rgb: np.ndarray) -> np.ndarray:
    """Baseline: the green channel (strongest blood absorption), mean-normalised and
    inverted so that more blood = higher value."""
    g = np.asarray(rgb, float)[:, 1]
    return -(g / g.mean() - 1)


def motion_reference(box: np.ndarray) -> np.ndarray:
    """Face motion (x, y, scale) relative to the face size, x10 so typical head motion has
    a magnitude similar to wrist acceleration in g (NLMS eps was tuned in those units)."""
    box = np.asarray(box, float)
    w = np.nanmedian(box[:, 2])
    cx = box[:, 0] + box[:, 2] / 2
    cy = box[:, 1] + box[:, 3] / 2
    return np.stack([cx / w, cy / w, box[:, 2] / w], axis=1) * 10
