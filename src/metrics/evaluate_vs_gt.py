"""
对比 baseline 和 improved 两版数字化结果与 WFDB ground truth 的指标。

生成 2×2 大图：
  左上：原始 ECG 扫描图片
  右上：U-Net 分割概率图
  左下：Ground Truth 波形（ECG 纸背景风格）
  右下：GT / Baseline / Improved 波形对比

用法:
    python -m src.metrics.evaluate_vs_gt \
        --gt ecg_timeseries \
        --baseline sandbox/inference_output_baseline \
        --improved sandbox/inference_output_pmcardio \
        --images ecg_images \
        --output sandbox/evaluation_results
"""

import argparse
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import wfdb
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from PIL import Image
from scipy.signal import resample

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


# ── Ground truth loading ──────────────────────────────────────────────


def load_wfdb(record_dir: str, record_name: str, target_samples: int = 5000) -> Optional[npt.NDArray[np.float64]]:
    """Load WFDB record and return (12, target_samples) array in µV, column-ordered as LEAD_NAMES."""
    record_path = os.path.join(record_dir, record_name)
    try:
        record = wfdb.rdrecord(record_path)
    except Exception as e:
        print(f"  [WARN] Could not read WFDB record {record_path}: {e}")
        return None

    signals = record.p_signal  # physical units (mV)
    sig_names = [s.strip() for s in record.sig_name]

    out = np.full((len(LEAD_NAMES), signals.shape[0]), np.nan)
    for i, lead in enumerate(LEAD_NAMES):
        if lead in sig_names:
            idx = sig_names.index(lead)
            out[i] = signals[:, idx] * 1000.0  # mV → µV

    if out.shape[1] != target_samples:
        out_resampled = np.zeros((len(LEAD_NAMES), target_samples))
        for i in range(len(LEAD_NAMES)):
            if np.all(np.isnan(out[i])):
                out_resampled[i] = np.nan
            else:
                out_resampled[i] = resample(out[i], target_samples)
        out = out_resampled

    return out


def load_csv(csv_path: str) -> Optional[npt.NDArray[np.float64]]:
    """Load canonical timeseries CSV → (n_leads, n_points)."""
    if not os.path.exists(csv_path):
        return None
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    return data.T


# ── Metrics ───────────────────────────────────────────────────────────


def pearson_correlation(a: npt.NDArray, b: npt.NDArray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    a_c, b_c = a[mask], b[mask]
    if len(a_c) < 10:
        return float("nan")
    a_c = a_c - a_c.mean()
    b_c = b_c - b_c.mean()
    denom = np.sqrt(np.sum(a_c**2) * np.sum(b_c**2))
    return float(np.sum(a_c * b_c) / denom) if denom > 1e-12 else float("nan")


def rmse(a: npt.NDArray, b: npt.NDArray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    a_c, b_c = a[mask], b[mask]
    if len(a_c) < 10:
        return float("nan")
    return float(np.sqrt(np.mean((a_c - b_c) ** 2)))


def snr_db(ref: npt.NDArray, est: npt.NDArray) -> float:
    mask = ~(np.isnan(ref) | np.isnan(est))
    r, e = ref[mask], est[mask]
    if len(r) < 10:
        return float("nan")
    noise = r - e
    sig_power = np.mean(r**2)
    noise_power = np.mean(noise**2)
    if noise_power < 1e-12:
        return float("inf")
    if sig_power < 1e-12:
        return float("-inf")
    return float(10 * np.log10(sig_power / noise_power))


def compute_metrics(gt: npt.NDArray, pred: npt.NDArray) -> dict[str, list[float]]:
    n_leads = min(gt.shape[0], pred.shape[0])
    n_pts = min(gt.shape[1], pred.shape[1])
    gt = gt[:n_leads, :n_pts]
    pred = pred[:n_leads, :n_pts]
    results: dict[str, list[float]] = {"pcc": [], "rmse_uv": [], "snr_db": []}
    for i in range(n_leads):
        results["pcc"].append(pearson_correlation(gt[i], pred[i]))
        results["rmse_uv"].append(rmse(gt[i], pred[i]))
        results["snr_db"].append(snr_db(gt[i], pred[i]))
    return results


# ── Display / IO ──────────────────────────────────────────────────────


def fmt(val: float, width: int = 8, prec: int = 4) -> str:
    return f"{val:{width}.{prec}f}" if not np.isnan(val) else f"{'NaN':>{width}}"


def print_side_by_side(
    metrics_bl: dict[str, list[float]],
    metrics_im: dict[str, list[float]],
    title: str,
) -> None:
    n = len(metrics_bl["pcc"])
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    header = (
        f"{'Lead':<6} │ {'PCC':>7} {'RMSE':>8} {'SNR':>7} │ {'PCC':>7} {'RMSE':>8} {'SNR':>7} │ {'ΔPCC':>7}"
    )
    print(f"       │ {'--- Baseline ---':^24} │ {'--- Improved ---':^24} │")
    print(header)
    print("─" * len(header))

    delta_pccs = []
    for i in range(n):
        name = LEAD_NAMES[i] if i < len(LEAD_NAMES) else f"L{i}"
        bl_pcc, bl_rmse, bl_snr = metrics_bl["pcc"][i], metrics_bl["rmse_uv"][i], metrics_bl["snr_db"][i]
        im_pcc, im_rmse, im_snr = metrics_im["pcc"][i], metrics_im["rmse_uv"][i], metrics_im["snr_db"][i]
        d_pcc = im_pcc - bl_pcc if not (np.isnan(im_pcc) or np.isnan(bl_pcc)) else float("nan")
        delta_pccs.append(d_pcc)
        sign = "+" if d_pcc > 0 else ""
        d_str = f"{sign}{d_pcc:.4f}" if not np.isnan(d_pcc) else "NaN"
        print(
            f"{name:<6} │ {fmt(bl_pcc,7)} {fmt(bl_rmse,8,1)} {fmt(bl_snr,7,2)} "
            f"│ {fmt(im_pcc,7)} {fmt(im_rmse,8,1)} {fmt(im_snr,7,2)} │ {d_str:>7}"
        )

    print("─" * len(header))
    avg = lambda lst: float(np.nanmean(lst))  # noqa: E731
    d_avg = (
        avg([d for d in delta_pccs if not np.isnan(d)])
        if any(not np.isnan(d) for d in delta_pccs)
        else float("nan")
    )
    sign = "+" if d_avg > 0 else ""
    d_str = f"{sign}{d_avg:.4f}" if not np.isnan(d_avg) else "NaN"
    print(
        f"{'AVG':<6} │ {fmt(avg(metrics_bl['pcc']),7)} {fmt(avg(metrics_bl['rmse_uv']),8,1)} "
        f"{fmt(avg(metrics_bl['snr_db']),7,2)} │ {fmt(avg(metrics_im['pcc']),7)} "
        f"{fmt(avg(metrics_im['rmse_uv']),8,1)} {fmt(avg(metrics_im['snr_db']),7,2)} │ {d_str:>7}"
    )
    print()


def save_csv(
    metrics_bl: dict[str, list[float]],
    metrics_im: dict[str, list[float]],
    output_dir: str,
    record_name: str,
) -> None:
    path = os.path.join(output_dir, f"{record_name}_eval.csv")
    with open(path, "w") as f:
        f.write("lead,bl_pcc,bl_rmse_uv,bl_snr_db,im_pcc,im_rmse_uv,im_snr_db,delta_pcc\n")
        for i in range(len(metrics_bl["pcc"])):
            name = LEAD_NAMES[i] if i < len(LEAD_NAMES) else f"L{i}"
            bl_pcc = metrics_bl["pcc"][i]
            im_pcc = metrics_im["pcc"][i]
            d = im_pcc - bl_pcc if not (np.isnan(im_pcc) or np.isnan(bl_pcc)) else float("nan")
            f.write(
                f"{name},{bl_pcc:.6f},{metrics_bl['rmse_uv'][i]:.2f},{metrics_bl['snr_db'][i]:.2f},"
                f"{im_pcc:.6f},{metrics_im['rmse_uv'][i]:.2f},{metrics_im['snr_db'][i]:.2f},{d:.6f}\n"
            )
        avg = lambda lst: float(np.nanmean(lst))  # noqa: E731
        d_avg = avg(metrics_im["pcc"]) - avg(metrics_bl["pcc"])
        f.write(
            f"AVERAGE,{avg(metrics_bl['pcc']):.6f},{avg(metrics_bl['rmse_uv']):.2f},{avg(metrics_bl['snr_db']):.2f},"
            f"{avg(metrics_im['pcc']):.6f},{avg(metrics_im['rmse_uv']):.2f},{avg(metrics_im['snr_db']):.2f},"
            f"{d_avg:.6f}\n"
        )
    print(f"  CSV saved: {path}")


# ── ECG paper-style plotting (like ecg-images-generator) ─────────────


def _draw_ecg_paper_style(
    ax: plt.Axes,
    signals: npt.NDArray,
    sample_rate: int = 500,
    title: str = "Ground Truth",
) -> None:
    """Draw 12-lead ECG on an axes with standard ECG paper grid background.

    Mimics ecg-images-generator 12x1 layout: 1 column, 12 rows, 10 seconds per lead.
    Order top-to-bottom: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.
    Pink/red grid, black traces, lead names.
    """
    n_leads = min(signals.shape[0], 12)
    secs_per_lead = 10.0
    samples_per_lead = min(int(secs_per_lead * sample_rate), signals.shape[1])

    # ECG paper grid colours (classic pink)
    color_major = (1.0, 0.796, 0.866)
    color_minor = (0.996, 0.929, 0.973)
    color_line = (0.0, 0.0, 0.0)

    # Vertical spacing per row in µV — enough room for typical ECG amplitude
    row_height_uv = 1500.0

    # Time axis in seconds
    t = np.arange(samples_per_lead) / sample_rate

    # 12x1 layout: lead order top → bottom = I, II, III, aVR, aVL, aVF, V1...V6
    lead_order = list(range(n_leads))  # [0, 1, 2, ..., 11]

    x_max = secs_per_lead
    y_min = -(n_leads + 0.5) * row_height_uv
    y_max = row_height_uv * 0.5

    ax.set_xlim(0, x_max)
    ax.set_ylim(y_min, y_max)

    # Grid: major = 0.2s × 500µV, minor = 0.04s × 100µV
    ax.set_facecolor("white")
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(500))
    ax.grid(which="major", color=color_major, linewidth=0.5, alpha=0.8)
    ax.grid(which="minor", color=color_minor, linewidth=0.3, alpha=0.6)
    ax.tick_params(axis="both", which="both", length=0, labelsize=0)

    # Draw traces
    for row_idx, lead_idx in enumerate(lead_order):
        sig = signals[lead_idx, :samples_per_lead]
        if np.all(np.isnan(sig)):
            continue
        y_offset = -(row_idx + 0.5) * row_height_uv

        ax.plot(t, sig + y_offset, color=color_line, linewidth=0.6)

        # Lead name label
        name = LEAD_NAMES[lead_idx] if lead_idx < len(LEAD_NAMES) else f"L{lead_idx}"
        ax.text(
            0.05,
            y_offset + row_height_uv * 0.35,
            name,
            fontsize=7,
            fontweight="bold",
            color="black",
            verticalalignment="top",
        )

    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.set_xlabel("")
    ax.set_ylabel("")


# ── 2×2 composite plot ───────────────────────────────────────────────


def plot_composite(
    gt: npt.NDArray,
    bl: npt.NDArray,
    im: npt.NDArray,
    metrics_bl: dict[str, list[float]],
    metrics_im: dict[str, list[float]],
    output_dir: str,
    record_name: str,
    ecg_image_path: Optional[str] = None,
    segmentation_npy_path: Optional[str] = None,
) -> None:
    """Generate 2×2 composite evaluation figure.

    Top-left:     Original ECG scan image
    Top-right:    U-Net segmentation probability map
    Bottom-left:  Digitized (improved) waveform in ECG paper style
    Bottom-right: GT / Baseline / Improved waveform overlay
    """
    fig, axs = plt.subplots(2, 2, figsize=(22, 16))

    # ─── Top-left: Original ECG image ────────────────────────────────
    ax_tl = axs[0, 0]
    if ecg_image_path and os.path.exists(ecg_image_path):
        img = Image.open(ecg_image_path)
        ax_tl.imshow(np.array(img))
        ax_tl.set_title("(a) Original ECG Scan", fontsize=11, fontweight="bold", pad=4)
    else:
        ax_tl.text(0.5, 0.5, "ECG image\nnot found", ha="center", va="center", fontsize=14, color="gray")
        ax_tl.set_title("(a) Original ECG Scan", fontsize=11, fontweight="bold", pad=4)
    ax_tl.axis("off")

    # ─── Top-right: U-Net segmentation ───────────────────────────────
    ax_tr = axs[0, 1]
    if segmentation_npy_path and os.path.exists(segmentation_npy_path):
        seg = np.load(segmentation_npy_path)
        ax_tr.imshow(seg, cmap="hot", interpolation="none", vmin=0, vmax=1)
        ax_tr.set_title("(b) U-Net Signal Segmentation", fontsize=11, fontweight="bold", pad=4)
    else:
        ax_tr.text(0.5, 0.5, "Segmentation\nnot available", ha="center", va="center", fontsize=14, color="gray")
        ax_tr.set_title("(b) U-Net Signal Segmentation", fontsize=11, fontweight="bold", pad=4)
    ax_tr.axis("off")

    # ─── Bottom-left: Digitized (improved) in ECG paper style ─────────
    ax_bl = axs[1, 0]
    _draw_ecg_paper_style(ax_bl, im, sample_rate=500, title="(c) Digitized Output (ECG Paper Style)")

    # ─── Bottom-right: Waveform comparison ───────────────────────────
    ax_br = axs[1, 1]
    n_leads = min(gt.shape[0], bl.shape[0], im.shape[0], 12)
    n_pts = min(gt.shape[1], bl.shape[1], im.shape[1])

    # Stack leads with vertical offset
    offset_step = 2000  # µV per lead
    for i in range(n_leads):
        offset = -i * offset_step
        name = LEAD_NAMES[i] if i < len(LEAD_NAMES) else f"L{i}"
        x = np.arange(n_pts)

        ax_br.plot(x, gt[i, :n_pts] + offset, color="black", linewidth=0.7, alpha=0.85, label="GT" if i == 0 else "")
        ax_br.plot(
            x, bl[i, :n_pts] + offset, color="dodgerblue", linewidth=0.5, alpha=0.55, label="Baseline" if i == 0 else ""
        )
        ax_br.plot(
            x, im[i, :n_pts] + offset, color="red", linewidth=0.5, alpha=0.55, label="Improved" if i == 0 else ""
        )

        # Lead name + PCC
        bl_pcc = metrics_bl["pcc"][i]
        im_pcc = metrics_im["pcc"][i]
        bl_str = f"{bl_pcc:.2f}" if not np.isnan(bl_pcc) else "N/A"
        im_str = f"{im_pcc:.2f}" if not np.isnan(im_pcc) else "N/A"
        ax_br.text(
            -n_pts * 0.01,
            offset,
            f"{name}",
            fontsize=7,
            fontweight="bold",
            ha="right",
            va="center",
        )
        ax_br.text(
            n_pts * 1.01,
            offset,
            f"BL:{bl_str} IM:{im_str}",
            fontsize=5.5,
            ha="left",
            va="center",
            color="gray",
        )

    ax_br.legend(fontsize=8, loc="upper right", ncol=3)
    ax_br.set_title("(d) Waveform Comparison: GT vs Baseline vs Improved", fontsize=11, fontweight="bold", pad=4)
    ax_br.set_xlabel("Sample", fontsize=9)
    ax_br.tick_params(labelsize=7)
    ax_br.spines["top"].set_visible(False)
    ax_br.spines["right"].set_visible(False)

    # ─── Global ──────────────────────────────────────────────────────
    avg_bl_pcc = float(np.nanmean(metrics_bl["pcc"]))
    avg_im_pcc = float(np.nanmean(metrics_im["pcc"]))
    fig.suptitle(
        f"{record_name}    Avg PCC: Baseline={avg_bl_pcc:.4f}  Improved={avg_im_pcc:.4f}  "
        f"(\u0394={avg_im_pcc - avg_bl_pcc:+.4f})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, f"{record_name}_eval.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Plot saved: {path}")


# ── File finding helpers ─────────────────────────────────────────────


def find_gt_records(gt_dir: str) -> dict[str, str]:
    """Return {record_id: record_dir} for all WFDB records under gt_dir."""
    records: dict[str, str] = {}
    for entry in sorted(os.listdir(gt_dir)):
        full = os.path.join(gt_dir, entry)
        if os.path.isdir(full):
            hea = os.path.join(full, entry + ".hea")
            if os.path.exists(hea):
                records[entry] = full
    return records


def find_pred_csv(pred_dir: str, record_id: str) -> Optional[str]:
    """Find the canonical CSV for a record id (tries {id}-0, {id}-1, {id})."""
    for suffix in ["-0", "-1", ""]:
        candidate = os.path.join(pred_dir, f"{record_id}{suffix}_timeseries_canonical.csv")
        if os.path.exists(candidate):
            return candidate
    return None


def find_file_with_suffixes(directory: str, record_id: str, extension: str) -> Optional[str]:
    """Find a file like {record_id}-0{extension} or {record_id}{extension}."""
    for suffix in ["-0", "-1", ""]:
        candidate = os.path.join(directory, f"{record_id}{suffix}{extension}")
        if os.path.exists(candidate):
            return candidate
    return None


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline & improved vs ground truth (WFDB).")
    parser.add_argument("--gt", required=True, help="Directory containing WFDB ground truth records")
    parser.add_argument("--baseline", required=True, help="Baseline inference output directory")
    parser.add_argument("--improved", required=True, help="Improved inference output directory")
    parser.add_argument("--images", default="ecg_images", help="Directory containing original ECG scan images")
    parser.add_argument("--output", default="sandbox/evaluation_results", help="Output directory for results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    gt_records = find_gt_records(args.gt)
    if not gt_records:
        print(f"No WFDB records found in {args.gt}")
        sys.exit(1)

    print(f"Found {len(gt_records)} ground-truth record(s): {list(gt_records.keys())}\n")

    all_bl: dict[str, dict[str, list[float]]] = {}
    all_im: dict[str, dict[str, list[float]]] = {}

    for rec_id, rec_dir in sorted(gt_records.items()):
        bl_csv = find_pred_csv(args.baseline, rec_id)
        im_csv = find_pred_csv(args.improved, rec_id)

        if bl_csv is None and im_csv is None:
            print(f"[SKIP] {rec_id}: no prediction CSV found in either directory")
            continue

        gt_data = load_wfdb(rec_dir, rec_id)
        if gt_data is None:
            continue

        bl_data = load_csv(bl_csv) if bl_csv else None
        im_data = load_csv(im_csv) if im_csv else None

        nan_placeholder = np.full_like(gt_data, np.nan)
        if bl_data is None:
            print(f"  [WARN] {rec_id}: no baseline CSV, using NaN")
            bl_data = nan_placeholder
        if im_data is None:
            print(f"  [WARN] {rec_id}: no improved CSV, using NaN")
            im_data = nan_placeholder

        metrics_bl = compute_metrics(gt_data, bl_data)
        metrics_im = compute_metrics(gt_data, im_data)
        all_bl[rec_id] = metrics_bl
        all_im[rec_id] = metrics_im

        print_side_by_side(metrics_bl, metrics_im, title=f"Record: {rec_id}")
        save_csv(metrics_bl, metrics_im, args.output, rec_id)

        # Find associated image and segmentation files
        ecg_image_path = find_file_with_suffixes(args.images, rec_id, ".png")
        if ecg_image_path is None:
            ecg_image_path = find_file_with_suffixes(args.images, rec_id, ".jpg")
        seg_npy_path = find_file_with_suffixes(args.improved, rec_id, "_segmentation.npy")

        plot_composite(
            gt_data,
            bl_data,
            im_data,
            metrics_bl,
            metrics_im,
            args.output,
            rec_id,
            ecg_image_path=ecg_image_path,
            segmentation_npy_path=seg_npy_path,
        )

    # ── Global summary ────────────────────────────────────────────────
    if len(all_bl) > 1:
        print(f"\n{'=' * 80}")
        print(f"  GLOBAL SUMMARY ({len(all_bl)} records)")
        print(f"{'=' * 80}")
        header = f"{'Lead':<6} │ {'BL PCC':>7} {'IM PCC':>7} {'ΔPCC':>7}"
        print(header)
        print("─" * len(header))
        for i, name in enumerate(LEAD_NAMES):
            bl_vals = [all_bl[r]["pcc"][i] for r in all_bl if i < len(all_bl[r]["pcc"])]
            im_vals = [all_im[r]["pcc"][i] for r in all_im if i < len(all_im[r]["pcc"])]
            bl_avg = float(np.nanmean(bl_vals))
            im_avg = float(np.nanmean(im_vals))
            d = im_avg - bl_avg
            sign = "+" if d > 0 else ""
            print(f"{name:<6} │ {fmt(bl_avg,7)} {fmt(im_avg,7)} {sign}{d:.4f}")
        print()


if __name__ == "__main__":
    main()
