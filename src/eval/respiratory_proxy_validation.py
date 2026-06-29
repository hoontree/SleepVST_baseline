"""Quantitative validation of the video respiratory proxy against EDF reference.

Compares each extracted per-epoch proxy (``epoch_{k}_movement.npy``) with the
EDF-derived breathing waveform (``{record}_bw.npy``, shape ``(num_epochs, 150)``
= 30 s @ 5 Hz) and reports how faithfully the proxy tracks real respiration.

Per epoch it computes:
  * ``r``            : Pearson correlation (sign-agnostic ``|r|`` reported too,
                       because luminance-based proxies may be inverted).
  * ``r_aligned``    : best ``|r|`` over a small lag search (±``max_lag_s``),
                       tolerating phase offset between camera and EDF.
  * ``br_proxy`` / ``br_ref`` : dominant breathing rate (breaths/min) from PSD.
  * ``br_abs_err``   : |br_proxy - br_ref|.

Aggregates per record and overall: median |r|, fraction of epochs with
|r| >= ``good_r`` and br error <= ``good_br``, breathing-rate MAE.

Because v1 (optical_flow, ~75 samples/epoch) and v2 (robust, ~150) differ in
length, both proxy and reference are resampled to a common grid before scoring.

Usage:
    python -m src.eval.respiratory_proxy_validation \
        --proxy-root data/resp_proxy_video_epochs \
        --ref-dir    data/resp_proxy_video \
        --records 30 --max-epochs 60 \
        --out-csv results/proxy_validation_v1.csv

Run it once per method (point --proxy-root at each output dir) to compare
v1 vs v2 on the same records.
"""

import argparse
import glob
import os
import random
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, resample, correlate


def _bandpass(sig, fs, low, high, order=2):
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


def _breathing_rate(sig, fs, band):
    if len(sig) < 8:
        return float('nan')
    freqs, psd = welch(sig, fs=fs, nperseg=min(len(sig), max(64, len(sig))))
    m = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(m) or psd[m].sum() <= 0:
        return float('nan')
    return float(freqs[m][np.argmax(psd[m])] * 60.0)


def _aligned_corr(a, b, max_lag):
    """Best |Pearson r| over integer lags in [-max_lag, max_lag]."""
    best = 0.0
    n = len(a)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x, y = a[-lag:], b[:n + lag]
        elif lag > 0:
            x, y = a[:n - lag], b[lag:]
        else:
            x, y = a, b
        if len(x) < 8 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
            continue
        r = abs(np.corrcoef(x, y)[0, 1])
        if r > best:
            best = r
    return best


def evaluate_epoch(proxy, ref, fs, band, n_common=150, max_lag_s=2.0):
    """Return metrics dict for one epoch, or None if not scorable."""
    if proxy is None or ref is None:
        return None
    proxy = np.asarray(proxy, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if proxy.size < 8 or ref.size < 8:
        return None
    if not np.all(np.isfinite(proxy)) or not np.all(np.isfinite(ref)):
        return None
    if np.std(proxy) < 1e-8 or np.std(ref) < 1e-8:
        return {"r": 0.0, "abs_r": 0.0, "r_aligned": 0.0,
                "br_proxy": float('nan'), "br_ref": float('nan'),
                "br_abs_err": float('nan'), "flat": True}

    # Resample both to a common grid, then band-pass + z-score.
    p = _zscore(_bandpass(resample(proxy, n_common), fs, band[0], band[1]))
    q = _zscore(_bandpass(resample(ref, n_common), fs, band[0], band[1]))

    r = float(np.corrcoef(p, q)[0, 1])
    max_lag = int(round(max_lag_s * fs))
    r_aligned = _aligned_corr(p, q, max_lag)

    br_p = _breathing_rate(p, fs, band)
    br_q = _breathing_rate(q, fs, band)
    br_err = abs(br_p - br_q) if np.isfinite(br_p) and np.isfinite(br_q) else float('nan')

    return {"r": r, "abs_r": abs(r), "r_aligned": r_aligned,
            "br_proxy": br_p, "br_ref": br_q, "br_abs_err": br_err, "flat": False}


def evaluate_record(record_id, proxy_root, ref_dir, band, max_epochs=None,
                    fs=5.0, n_common=150):
    ref_path = Path(ref_dir) / f'{record_id}_bw.npy'
    if not ref_path.exists():
        return []
    ref_arr = np.load(ref_path)              # (num_epochs, 150)
    rec_dir = Path(proxy_root) / record_id
    if not rec_dir.is_dir():
        return []

    rows = []
    # proxy epoch_{k} (1-based) <-> reference index k-1
    epoch_dirs = sorted(rec_dir.glob('epoch_*'),
                        key=lambda p: int(p.name.split('_')[1]))
    if max_epochs:
        epoch_dirs = epoch_dirs[:max_epochs]

    for ed in epoch_dirs:
        k = int(ed.name.split('_')[1])
        mv = ed / f'epoch_{k}_movement.npy'
        ref_idx = k - 1
        if not mv.exists() or ref_idx < 0 or ref_idx >= len(ref_arr):
            continue
        try:
            proxy = np.load(mv)
        except Exception:
            continue
        m = evaluate_epoch(proxy, ref_arr[ref_idx], fs, band, n_common)
        if m is None:
            continue
        m.update({"record": record_id, "epoch": k})
        rows.append(m)
    return rows


def summarize(rows, good_r=0.5, good_br=3.0):
    if not rows:
        return {}
    abs_r = np.array([r["abs_r"] for r in rows])
    al = np.array([r["r_aligned"] for r in rows])
    br_err = np.array([r["br_abs_err"] for r in rows], dtype=float)
    valid_br = br_err[np.isfinite(br_err)]
    return {
        "n_epochs": len(rows),
        "median_abs_r": float(np.median(abs_r)),
        "mean_abs_r": float(np.mean(abs_r)),
        "median_r_aligned": float(np.median(al)),
        "frac_good_r": float(np.mean(abs_r >= good_r)),
        "frac_good_r_aligned": float(np.mean(al >= good_r)),
        "br_mae": float(np.mean(valid_br)) if valid_br.size else float('nan'),
        "frac_good_br": float(np.mean(valid_br <= good_br)) if valid_br.size else float('nan'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proxy-root', required=True)
    ap.add_argument('--ref-dir', default='data/resp_proxy_video')
    ap.add_argument('--records', default='30',
                    help='int N (random sample) or comma-separated record ids')
    ap.add_argument('--max-epochs', type=int, default=60,
                    help='max epochs per record (0 = all)')
    ap.add_argument('--band', default='0.1,0.5')
    ap.add_argument('--fs', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-csv', default=None)
    args = ap.parse_args()

    band = tuple(float(x) for x in args.band.split(','))
    max_epochs = args.max_epochs or None

    ref_ids = {os.path.basename(f).replace('_bw.npy', '')
               for f in glob.glob(str(Path(args.ref_dir) / '*_bw.npy'))}
    proxy_ids = {p.name for p in Path(args.proxy_root).iterdir() if p.is_dir()}
    common = sorted(ref_ids & proxy_ids)

    if ',' in args.records or not args.records.isdigit():
        records = [r for r in args.records.split(',') if r in common]
    else:
        random.seed(args.seed)
        n = min(int(args.records), len(common))
        records = sorted(random.sample(common, n))

    print(f"Evaluating {len(records)} records "
          f"(of {len(common)} with both proxy & reference), "
          f"<= {max_epochs} epochs each, band={band} Hz")

    all_rows = []
    per_record = []
    for rid in records:
        rows = evaluate_record(rid, args.proxy_root, args.ref_dir, band, max_epochs, args.fs)
        all_rows.extend(rows)
        s = summarize(rows)
        if s:
            s["record"] = rid
            per_record.append(s)

    overall = summarize(all_rows)
    print("\n=== OVERALL ===")
    for k, v in overall.items():
        print(f"  {k:22s}: {v:.4f}" if isinstance(v, float) else f"  {k:22s}: {v}")

    if per_record:
        med = np.median([s["median_abs_r"] for s in per_record])
        print(f"\n  per-record median |r| (median over records): {med:.4f}")
        worst = sorted(per_record, key=lambda s: s["median_abs_r"])[:5]
        print("  weakest records (median |r|):")
        for s in worst:
            print(f"    {s['record']}: {s['median_abs_r']:.3f}  (n={s['n_epochs']})")

    if args.out_csv:
        import csv
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=["record", "epoch", "r", "abs_r",
                               "r_aligned", "br_proxy", "br_ref", "br_abs_err", "flat"])
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"\nWrote {len(all_rows)} epoch rows to {args.out_csv}")


if __name__ == '__main__':
    main()
