"""Compare v1 (optical-flow) vs v2 (robust PBM) respiratory proxy against oracle.

Selects a sample of records that have both v1 proxy data and oracle reference,
runs v2 extraction on the same epochs, then computes and prints a side-by-side
metric table.

Usage (from project root):
    python scripts/compare_proxy_v1_v2.py [--records 10] [--epochs 20] [--seed 0]
"""

import argparse
import sys
import random
import time
from pathlib import Path

import numpy as np
import cv2
from scipy.signal import butter, sosfiltfilt, welch, resample, correlate

# --- project root on path so src.* imports work ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.preprocess.io import loadVideoStreamCV2
from src.data.preprocess.respiratory_extraction_v2 import extract_respiratory_signal
from src.data.preprocess.filters.temporal_filters import difference_of_iir


# ---------- shared metric helpers ----------

def _bandpass(sig, fs, low=0.1, high=0.5, order=2):
    nyq = fs / 2.0
    lo, hi = max(low / nyq, 1e-4), min(high / nyq, 0.999)
    if hi <= lo:
        return sig - np.mean(sig)
    sos = butter(order, [lo, hi], btype='band', output='sos')
    if len(sig) <= 3 * (sos.shape[0] * 2):
        return sig - np.mean(sig)
    return sosfiltfilt(sos, sig)


def _zscore(x):
    s = np.std(x)
    return (x - np.mean(x)) / s if s > 1e-8 else x - np.mean(x)


def _breathing_rate(sig, fs, band=(0.1, 0.5)):
    if len(sig) < 8:
        return float('nan')
    freqs, psd = welch(sig, fs=fs, nperseg=min(len(sig), max(64, len(sig))))
    m = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(m) or psd[m].sum() <= 0:
        return float('nan')
    return float(freqs[m][np.argmax(psd[m])] * 60.0)


def _aligned_corr(a, b, max_lag):
    best = 0.0
    n = len(a)
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[-lag:], b[:n + lag]) if lag < 0 else (a[:n - lag], b[lag:]) if lag > 0 else (a, b)
        if len(x) < 8 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
            continue
        r = abs(np.corrcoef(x, y)[0, 1])
        if r > best:
            best = r
    return best


def score_epoch(proxy, ref, fs=5.0, n_common=150, max_lag_s=2.0):
    """Return metrics dict (r, abs_r, r_aligned, br_proxy, br_ref, br_abs_err)."""
    proxy, ref = np.asarray(proxy, float), np.asarray(ref, float)
    if proxy.size < 8 or ref.size < 8:
        return None
    if np.std(proxy) < 1e-8 or np.std(ref) < 1e-8:
        return {'r': 0.0, 'abs_r': 0.0, 'r_aligned': 0.0,
                'br_proxy': float('nan'), 'br_ref': float('nan'), 'br_abs_err': float('nan')}

    p = _zscore(_bandpass(resample(proxy, n_common), fs))
    q = _zscore(_bandpass(resample(ref,   n_common), fs))

    r = float(np.corrcoef(p, q)[0, 1])
    r_al = _aligned_corr(p, q, int(round(max_lag_s * fs)))
    br_p = _breathing_rate(p, fs)
    br_q = _breathing_rate(q, fs)
    br_err = abs(br_p - br_q) if np.isfinite(br_p) and np.isfinite(br_q) else float('nan')
    return {'r': r, 'abs_r': abs(r), 'r_aligned': r_al,
            'br_proxy': br_p, 'br_ref': br_q, 'br_abs_err': br_err}


def summarize(rows):
    if not rows:
        return {}
    abs_r = np.array([m['abs_r'] for m in rows])
    al    = np.array([m['r_aligned'] for m in rows])
    br_e  = np.array([m['br_abs_err'] for m in rows], float)
    vbr   = br_e[np.isfinite(br_e)]
    return {
        'n_epochs':           len(rows),
        'median_abs_r':       float(np.median(abs_r)),
        'mean_abs_r':         float(np.mean(abs_r)),
        'median_r_aligned':   float(np.median(al)),
        'frac_abs_r>=0.5':    float(np.mean(abs_r >= 0.5)),
        'frac_aligned>=0.5':  float(np.mean(al >= 0.5)),
        'br_mae':             float(np.mean(vbr)) if vbr.size else float('nan'),
        'frac_br_err<=3':     float(np.mean(vbr <= 3.0)) if vbr.size else float('nan'),
    }


# ---------- temporal filter (matches pipeline config) ----------

def _temporal_filter(delta, fl, fh):
    return difference_of_iir(delta, rl=0.4, rh=0.05)


# ---------- v2 extraction wrapper ----------

V2_PARAMS = dict(
    mag_factor=50,
    freq_range=[0.2, 0.3],
    attenuate=True,
    sigma=5,
    temporal_filter=_temporal_filter,
    roi_quantile=0.95,
    resp_band=(0.1, 0.5),
)
CROP_BOX = (32, 422, 125, 515)
RESIZE   = (250, 250)


def run_v2_epoch(video_path: str, frame_start: int, fps: float):
    """Load one epoch from video and run v2 extraction. Returns (signal, quality) or None."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    frames_per_epoch = int(round(30 * fps))
    epoch_video = loadVideoStreamCV2(cap, num_frames=frames_per_epoch,
                                     crop_box=CROP_BOX, size=RESIZE, normalize=True)
    cap.release()

    if epoch_video is None or len(epoch_video) < 10:
        return None, None

    signal, _, _, quality = extract_respiratory_signal(epoch_video, fps, **V2_PARAMS)
    return signal, quality


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', type=int, default=5,
                    help='Number of records to sample')
    ap.add_argument('--epochs', type=int, default=10,
                    help='Max epochs per record')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--video-root', default='/tf/00_data/#_2021_Sleep_Video')
    ap.add_argument('--proxy-root', default='data/resp_proxy_video_epochs')
    ap.add_argument('--ref-dir',    default='data/resp_proxy_video')
    args = ap.parse_args()

    random.seed(args.seed)

    proxy_root = Path(args.proxy_root)
    ref_dir    = Path(args.ref_dir)
    video_root = Path(args.video_root)

    # Find records that have v1 proxy, oracle reference, AND a video file
    proxy_ids = {p.name for p in proxy_root.iterdir() if p.is_dir()}
    ref_ids   = {f.stem.replace('_bw', '') for f in ref_dir.glob('*_bw.npy')}
    video_ids = {p.name for p in video_root.iterdir() if p.is_dir()} if video_root.is_dir() else set()
    common = sorted(proxy_ids & ref_ids & video_ids)

    n = min(args.records, len(common))
    records = random.sample(common, n)
    print(f"\nSelected {n} records from {len(common)} candidates:")
    for r in records:
        print(f"  {r}")

    v1_all, v2_all = [], []
    v1_per, v2_per = [], []

    for rid in records:
        ref_arr = np.load(ref_dir / f'{rid}_bw.npy')    # (num_epochs, 150)
        ep_root = proxy_root / rid

        # Find v1 epoch files (epoch_k/epoch_k_movement.npy, 1-indexed)
        ep_dirs = sorted(ep_root.glob('epoch_*/'),
                         key=lambda p: int(p.name.split('_')[1]))[:args.epochs]

        video_path = str(video_root / rid / f'{rid}_video_01.mp4')
        cap_tmp = cv2.VideoCapture(video_path)
        cap_fps = cap_tmp.get(cv2.CAP_PROP_FPS) or 30.0
        cap_tmp.release()
        frames_per_epoch = int(round(30 * cap_fps))

        v1_rows, v2_rows = [], []
        for ep_dir in ep_dirs:
            k = int(ep_dir.name.split('_')[1])           # 1-indexed epoch
            ref_idx = k - 1
            if ref_idx < 0 or ref_idx >= len(ref_arr):
                continue

            ref_sig = ref_arr[ref_idx]                    # (150,)

            # --- v1 ---
            mv_path = ep_dir / f'epoch_{k}_movement.npy'
            if mv_path.exists():
                v1_sig = np.load(mv_path)
                m1 = score_epoch(v1_sig, ref_sig)
                if m1:
                    v1_rows.append(m1)

            # --- v2 (live extraction) ---
            frame_start = (k - 1) * frames_per_epoch
            t0 = time.time()
            v2_sig, quality = run_v2_epoch(video_path, frame_start, cap_fps)
            elapsed = time.time() - t0
            if v2_sig is not None:
                m2 = score_epoch(v2_sig, ref_sig)
                if m2:
                    m2['quality_ok'] = quality.get('ok', False)
                    m2['snr']        = quality.get('snr', float('nan'))
                    v2_rows.append(m2)
                print(f"  {rid} ep{k}: v2 extracted in {elapsed:.1f}s  "
                      f"br={quality.get('breathing_rate', float('nan')):.1f}/min  "
                      f"SNR={quality.get('snr', 0):.2f}  ok={quality.get('ok')}")

        v1_all.extend(v1_rows)
        v2_all.extend(v2_rows)
        v1_per.append(summarize(v1_rows))
        v2_per.append(summarize(v2_rows))

    # --- Summary table ---
    s1 = summarize(v1_all)
    s2 = summarize(v2_all)

    keys = ['n_epochs', 'median_abs_r', 'mean_abs_r', 'median_r_aligned',
            'frac_abs_r>=0.5', 'frac_aligned>=0.5', 'br_mae', 'frac_br_err<=3']

    print("\n" + "="*60)
    print(f"{'Metric':<25}  {'v1 (optical-flow)':>18}  {'v2 (robust PBM)':>16}")
    print("="*60)
    for k in keys:
        v1v = s1.get(k, float('nan'))
        v2v = s2.get(k, float('nan'))
        if isinstance(v1v, int):
            print(f"{k:<25}  {v1v:>18d}  {v2v:>16d}")
        else:
            print(f"{k:<25}  {v1v:>18.4f}  {v2v:>16.4f}")
    print("="*60)

    if v2_all:
        ok_rate = np.mean([m.get('quality_ok', False) for m in v2_all])
        mean_snr = np.nanmean([m.get('snr', float('nan')) for m in v2_all])
        print(f"\nv2 quality gating: {ok_rate:.1%} epochs pass QC  "
              f"(mean SNR={mean_snr:.3f})")

    # Delta row
    print("\nDelta (v2 - v1):")
    for k in ['median_abs_r', 'median_r_aligned', 'frac_abs_r>=0.5', 'br_mae']:
        v1v = s1.get(k, float('nan'))
        v2v = s2.get(k, float('nan'))
        if np.isfinite(v1v) and np.isfinite(v2v):
            sign = '+' if v2v > v1v else ''
            print(f"  {k:<25}: {sign}{v2v - v1v:+.4f}")

    return s1, s2


if __name__ == '__main__':
    main()
