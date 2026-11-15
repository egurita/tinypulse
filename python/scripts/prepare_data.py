"""Convert PPG-DaLiA's per-subject pickles into compact .npz files.

Expects data/raw/data.zip (the inner archive of UCI's ppg+dalia.zip). Each S*.pkl
is streamed straight out of the zip and unpickled in memory, so the ~20 GB of
pickles never touch the disk. Output: data/processed/S1.npz ... S15.npz.

Usage:  uv run python python/scripts/prepare_data.py
"""

import time
import zipfile

from ppg_hr.data import PROCESSED_DIR, REPO_ROOT, SUBJECTS, convert_pickle

RAW_ZIP = REPO_ROOT / "data" / "raw" / "data.zip"


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RAW_ZIP) as z:
        for sid in SUBJECTS:
            out = PROCESSED_DIR / f"{sid}.npz"
            if out.exists():
                print(f"{sid}: already done")
                continue
            t = time.perf_counter()
            with z.open(f"PPG_FieldStudy/{sid}/{sid}.pkl") as f:
                convert_pickle(f, out)
            print(f"{sid}: {out.stat().st_size / 1e6:.0f} MB in {time.perf_counter() - t:.0f} s")


if __name__ == "__main__":
    main()
