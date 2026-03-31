"""
对比两个版本的 ECG 数字化结果，生成量化指标和可视化。

用法:
    python -m src.metrics.comparison \
        --baseline sandbox/inference_output_baseline \
        --improved sandbox/inference_output_pmcardio \
        --output sandbox/comparison_results
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def load_timeseries(csv_path: str) -> npt.NDArray[np.float64] | None:
    """加载 canonical timeseries CSV，返回 (n_leads, n_points) 数组。"""
    if not os.path.exists(csv_path):
        return None
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    # CSV 是 (n_points, n_leads)，转置为 (n_leads, n_points)
    return data.T


def pearson_correlation(a: npt.NDArray, b: npt.NDArray) -> float:
    """计算两个信号的 Pearson 相关系数。"""
    # 去除 NaN
    mask = ~(np.isnan(a) | np.isnan(b))
    a_clean, b_clean = a[mask], b[mask]
    if len(a_clean) < 10:
        return float("nan")
    a_centered = a_clean - a_clean.mean()
    b_centered = b_clean - b_clean.mean()
    denom = np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2))
    if denom < 1e-12:
        return float("nan")
    return float(np.sum(a_centered * b_centered) / denom)


def rmse(a: npt.NDArray, b: npt.NDArray) -> float:
    """计算 RMSE (µV)。"""
    mask = ~(np.isnan(a) | np.isnan(b))
    a_clean, b_clean = a[mask], b[mask]
    if len(a_clean) < 10:
        return float("nan")
    return float(np.sqrt(np.mean((a_clean - b_clean) ** 2)))


def snr(signal: npt.NDArray, noise: npt.NDArray) -> float:
    """计算 SNR (dB)。signal 为参考信号，noise = signal - reconstructed。"""
    mask = ~(np.isnan(signal) | np.isnan(noise))
    sig_clean, noise_clean = signal[mask], noise[mask]
    if len(sig_clean) < 10:
        return float("nan")
    signal_power = np.mean(sig_clean**2)
    noise_power = np.mean(noise_clean**2)
    if noise_power < 1e-12:
        return float("inf")
    if signal_power < 1e-12:
        return float("-inf")
    return float(10 * np.log10(signal_power / noise_power))


def compute_metrics(
    baseline: npt.NDArray, improved: npt.NDArray
) -> dict[str, list[float]]:
    """
    计算逐导联的 PCC / RMSE / SNR。
    baseline 和 improved 均为 (n_leads, n_points)。
    """
    n_leads = min(baseline.shape[0], improved.shape[0])
    # 对齐长度
    n_points = min(baseline.shape[1], improved.shape[1])
    baseline = baseline[:n_leads, :n_points]
    improved = improved[:n_leads, :n_points]

    results: dict[str, list[float]] = {"pcc": [], "rmse_uv": [], "snr_db": []}
    for i in range(n_leads):
        pcc_val = pearson_correlation(baseline[i], improved[i])
        rmse_val = rmse(baseline[i], improved[i])
        diff = baseline[i] - improved[i]
        snr_val = snr(baseline[i], diff)
        results["pcc"].append(pcc_val)
        results["rmse_uv"].append(rmse_val)
        results["snr_db"].append(snr_val)
    return results


def print_metrics_table(metrics: dict[str, list[float]], title: str = "") -> None:
    """终端打印量化指标表格。"""
    n_leads = len(metrics["pcc"])
    lead_names = LEAD_NAMES[:n_leads]

    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    header = f"{'Lead':<6} {'PCC':>8} {'RMSE(µV)':>10} {'SNR(dB)':>10}"
    print(header)
    print("-" * len(header))

    for i, name in enumerate(lead_names):
        pcc = metrics["pcc"][i]
        rmse_val = metrics["rmse_uv"][i]
        snr_val = metrics["snr_db"][i]
        pcc_str = f"{pcc:.4f}" if not np.isnan(pcc) else "NaN"
        rmse_str = f"{rmse_val:.2f}" if not np.isnan(rmse_val) else "NaN"
        snr_str = f"{snr_val:.2f}" if not np.isnan(snr_val) else "NaN"
        print(f"{name:<6} {pcc_str:>8} {rmse_str:>10} {snr_str:>10}")

    # 平均值
    avg_pcc = np.nanmean(metrics["pcc"])
    avg_rmse = np.nanmean(metrics["rmse_uv"])
    avg_snr = np.nanmean(metrics["snr_db"])
    print("-" * len(header))
    print(f"{'AVG':<6} {avg_pcc:>8.4f} {avg_rmse:>10.2f} {avg_snr:>10.2f}")
    print()


def save_metrics_csv(
    metrics: dict[str, list[float]], output_path: str, filename: str
) -> None:
    """保存指标到 CSV。"""
    n_leads = len(metrics["pcc"])
    lead_names = LEAD_NAMES[:n_leads]
    csv_path = os.path.join(output_path, filename + "_metrics.csv")
    with open(csv_path, "w") as f:
        f.write("lead,pcc,rmse_uv,snr_db\n")
        for i, name in enumerate(lead_names):
            f.write(f"{name},{metrics['pcc'][i]:.6f},{metrics['rmse_uv'][i]:.4f},{metrics['snr_db'][i]:.4f}\n")
        avg_pcc = np.nanmean(metrics["pcc"])
        avg_rmse = np.nanmean(metrics["rmse_uv"])
        avg_snr = np.nanmean(metrics["snr_db"])
        f.write(f"AVERAGE,{avg_pcc:.6f},{avg_rmse:.4f},{avg_snr:.4f}\n")
    print(f"Metrics saved to {csv_path}")


def plot_comparison(
    baseline: npt.NDArray,
    improved: npt.NDArray,
    metrics: dict[str, list[float]],
    output_path: str,
    filename: str,
) -> None:
    """生成并排波形对比 PNG。"""
    n_leads = min(baseline.shape[0], improved.shape[0])
    n_points = min(baseline.shape[1], improved.shape[1])
    baseline = baseline[:n_leads, :n_points]
    improved = improved[:n_leads, :n_points]
    lead_names = LEAD_NAMES[:n_leads]

    fig, axes = plt.subplots(n_leads, 1, figsize=(16, 2.2 * n_leads), sharex=True)
    if n_leads == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, lead_names)):
        x = np.arange(n_points)
        ax.plot(x, baseline[i], color="blue", linewidth=0.6, alpha=0.7, label="Baseline")
        ax.plot(x, improved[i], color="red", linewidth=0.6, alpha=0.7, label="PMcardio")

        pcc_val = metrics["pcc"][i]
        rmse_val = metrics["rmse_uv"][i]
        pcc_str = f"{pcc_val:.4f}" if not np.isnan(pcc_val) else "NaN"
        rmse_str = f"{rmse_val:.1f}" if not np.isnan(rmse_val) else "NaN"
        ax.set_ylabel(name, fontsize=10, fontweight="bold")
        ax.set_title(f"PCC={pcc_str}  RMSE={rmse_str}µV", fontsize=8, loc="right")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8, loc="upper left")

    axes[-1].set_xlabel("Sample", fontsize=10)
    plt.suptitle("Baseline (blue) vs PMcardio-improved (red)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    png_path = os.path.join(output_path, filename + "_comparison.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to {png_path}")


def find_timeseries_files(directory: str) -> dict[str, str]:
    """在目录中查找所有 timeseries CSV 文件，返回 {basename: full_path}。"""
    result: dict[str, str] = {}
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith("_timeseries_canonical.csv"):
                basename = f.replace("_timeseries_canonical.csv", "")
                result[basename] = os.path.join(root, f)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs improved ECG digitization results.")
    parser.add_argument("--baseline", required=True, help="Path to baseline output directory")
    parser.add_argument("--improved", required=True, help="Path to improved output directory")
    parser.add_argument("--output", default="sandbox/comparison_results", help="Path to save comparison results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    baseline_files = find_timeseries_files(args.baseline)
    improved_files = find_timeseries_files(args.improved)

    common = set(baseline_files.keys()) & set(improved_files.keys())
    if not common:
        print("No matching timeseries files found between baseline and improved directories.")
        print(f"  Baseline files: {list(baseline_files.keys())}")
        print(f"  Improved files: {list(improved_files.keys())}")
        sys.exit(1)

    print(f"Found {len(common)} matching file(s) to compare.\n")

    for name in sorted(common):
        print(f"--- Comparing: {name} ---")
        baseline_data = load_timeseries(baseline_files[name])
        improved_data = load_timeseries(improved_files[name])

        if baseline_data is None or improved_data is None:
            print(f"  Skipping {name}: could not load data.")
            continue

        metrics = compute_metrics(baseline_data, improved_data)
        print_metrics_table(metrics, title=f"File: {name}")
        save_metrics_csv(metrics, args.output, name)
        plot_comparison(baseline_data, improved_data, metrics, args.output, name)


if __name__ == "__main__":
    main()
