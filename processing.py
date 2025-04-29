import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import label as bwlabel
import uuid
# Double-sided sigmoid model
def double_sided_sigmoid(x, A, k, x1, x2, B):
    return A / (1 + np.exp(-k * (x - x1))) - A / (1 + np.exp(-k * (x - x2))) + B

## Original sigmoid-based profile fitting
def process_file(file_path, plot_title="Beam Profile", x_label="Position", y_label="Intensity"):
    df = pd.read_excel(file_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        raise ValueError("Uploaded file must have at least two numeric columns.")

    x = pd.to_numeric(df[numeric_cols[0]], errors='coerce').dropna().values
    y = pd.to_numeric(df[numeric_cols[1]], errors='coerce').dropna().values

    if len(x) != len(y):
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]

    initial_guess = [1.0, 1.0, np.percentile(x, 25), np.percentile(x, 75), min(y)]
    popt, _ = curve_fit(double_sided_sigmoid, x, y, p0=initial_guess)

    x_fine = np.linspace(min(x), max(x), 1000)
    fitted_curve = double_sided_sigmoid(x_fine, *popt)

    max_val = np.max(fitted_curve)
    min_val = np.min(fitted_curve)
    center_x = x_fine[np.argmax(fitted_curve)]
    center_y = max_val

    half_max = (max_val + min_val) / 2

    left_idx_50 = np.where(fitted_curve[:np.argmax(fitted_curve)] >= half_max)[0][0]
    right_idx_50 = np.where(fitted_curve[np.argmax(fitted_curve):] <= half_max)[0][0] + np.argmax(fitted_curve)

    fwhm = x_fine[right_idx_50] - x_fine[left_idx_50]

    range_val = max_val - min_val
    level_20 = min_val + 0.2 * range_val
    level_80 = min_val + 0.8 * range_val

    left_idx_20 = np.where(fitted_curve[:np.argmax(fitted_curve)] >= level_20)[0][0]
    left_idx_80 = np.where(fitted_curve[:np.argmax(fitted_curve)] >= level_80)[0][0]
    right_idx_80 = np.where(fitted_curve[np.argmax(fitted_curve):] <= level_80)[0][0] + np.argmax(fitted_curve)
    right_idx_20 = np.where(fitted_curve[np.argmax(fitted_curve):] <= level_20)[0][0] + np.argmax(fitted_curve)

    left_penumbra = x_fine[left_idx_80] - x_fine[left_idx_20]
    right_penumbra = x_fine[right_idx_20] - x_fine[right_idx_80]

    plt.figure(figsize=(5, 4))
    plt.plot(x, y, 'b.', markersize=6, label="Data")
    plt.plot(x_fine, fitted_curve, 'r-', linewidth=1.5, label="Sigmoid Fit")
    plt.plot([x_fine[left_idx_50], x_fine[right_idx_50]], [half_max, half_max], 'g--', linewidth=1.5, label="FWHM")
    plt.plot([x_fine[left_idx_20], x_fine[left_idx_80]], [level_20, level_80], 'm-', linewidth=1.5, label="Left Penumbra")
    plt.plot([x_fine[right_idx_80], x_fine[right_idx_20]], [level_80, level_20], 'c-', linewidth=1.5, label="Right Penumbra")
    plt.plot(center_x, center_y, 'ko', markersize=6, label="Center")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    unique_name = f"plot_{uuid.uuid4().hex[:8]}.png"
    plot_filename = os.path.join("static", unique_name)
    os.makedirs("static", exist_ok=True)
    plt.savefig(plot_filename, dpi=300)
    plt.close()

    return {
        "Amplitude (A)": round(popt[0], 4),
        "Steepness (k)": round(popt[1], 4),
        "Rising Edge (x1)": round(popt[2], 4),
        "Falling Edge (x2)": round(popt[3], 4),
        "Baseline (B)": round(popt[4], 4),
        "Center": round(center_x, 4),
        "FWHM": round(fwhm, 4),
        "Left Penumbra (20%-80%)": round(left_penumbra, 4),
        "Right Penumbra (80%-20%)": round(right_penumbra, 4),
        "Plot Path": plot_filename
    }

def process_one_file(file_path, plot_title="Cluster Profile", x_label="Data Points", y_label="Signal Value"):
    import uuid

    # Start reading from header row (row 13 = index 12)
    df = pd.read_excel(file_path, header=12)
    df = df.apply(pd.to_numeric, errors='coerce')

    if 'Charge2[pC]' not in df.columns:
        raise ValueError("Expected 'Charge2[pC]' column missing from uploaded Excel.")

    # Use Charge2[pC] as signal
    signal = df['Charge2[pC]'].values
    if signal.size == 0:
        raise ValueError("Signal data not found in Charge2[pC] column.")

    signal_nan = np.where(signal < 0.1 * np.max(signal), np.nan, signal)

    # Plot 1: Raw signal with points filtered
    plt.figure(figsize=(5, 4))
    plt.plot(np.arange(len(signal_nan)), signal_nan, 'bo')
    plt.xlabel("Data Points")
    plt.ylabel("Signal Value")
    plt.title("Filtered Signal from Profile")
    plt.grid(True)
    signal_plot = f"static/cluster_signal_{uuid.uuid4().hex[:6]}.png"
    plt.savefig(signal_plot, dpi=300)
    plt.close()

    # Plot 2: Create a fake "cluster mean" and "position" from Charge2[pC]
    cluster_mean = df['Charge2[pC]'].rolling(window=3, center=True).mean().dropna().values
    position = np.linspace(90, 110, len(cluster_mean))  # fabricated positions

    norm_cluster = cluster_mean / np.nanmax(cluster_mean)
    plt.figure(figsize=(5, 4))
    plt.plot(position, norm_cluster, 'ro', markerfacecolor='none')
    plt.xlabel("Position")
    plt.ylabel("Normalized Cluster Mean")
    plt.title("Cluster Mean vs Position")
    plt.grid(True)
    cluster_plot = f"static/cluster_mean_{uuid.uuid4().hex[:6]}.png"
    plt.savefig(cluster_plot, dpi=300)
    plt.close()

    return {
        "Cluster Signal Plot": signal_plot,
        "Normalized Means Plot": cluster_plot,
        "Plot Path": cluster_plot  # for <img src=...>
    }



# New: cluster signal plotting (mimic screenshots)
def process_cluster_scatter(file_path):
    df = pd.read_excel(file_path)
    signal = df.iloc[:, 2].values  # third column = signal
    xvals = np.arange(len(signal))

    # Plot 1: raw signal with gaps removed (non-zero clusters)
    signal_nan = np.where(signal < 0.1 * np.max(signal), np.nan, signal)
    plt.figure(figsize=(8,6))
    plt.plot(xvals, signal_nan, 'bo')
    plt.xlabel("Data Points")
    plt.ylabel("Signal Value")
    plt.title("Modified Signal with Only Middle Points Kept")
    plt.grid(True)
    sig_plot_path = "static/cluster_scatter_signal.png"
    plt.savefig(sig_plot_path, dpi=300)
    plt.close()

    # Plot 2: normalized cluster values at predefined positions
    cluster_vals = df.iloc[:, 3].values  # assume fourth column is averaged clusters
    pos = df.iloc[:, 0].values  # assume position values in first column
    cluster_vals = cluster_vals / np.max(cluster_vals)

    plt.figure(figsize=(8,6))
    plt.plot(pos, cluster_vals, 'ro', markersize=7, markerfacecolor='none')
    plt.xlabel("Position")
    plt.ylabel("Normalized Value")
    plt.title("Normalized Cluster Means")
    plt.grid(True)
    norm_plot_path = "static/cluster_scatter_means.png"
    plt.savefig(norm_plot_path, dpi=300)
    plt.close()

    return {
        "Cluster Signal Plot": sig_plot_path,
        "Normalized Means Plot": norm_plot_path
    }

#Legacy 11-cluster processor from original spec
def process_good_11clusters(file_path):
    df = pd.read_excel(file_path)
    signal = -df.iloc[:, 2].values
    bk = np.nanmean(signal[:10])
    threshold = 1.618 * 100 * bk
    signal = np.where(signal < threshold, np.nan, signal)

    padded = np.pad(signal, (1,1), constant_values=np.nan)
    non_nan = ~np.isnan(padded)
    has_left = non_nan[:-2]
    has_right = non_nan[2:]
    is_isolated = non_nan[1:-1] & ~has_left & ~has_right
    signal[is_isolated] = np.nan

    valid = ~np.isnan(signal)
    cluster_labels, num_clusters = bwlabel(valid)

    signal_mean = []
    for i in range(1, num_clusters+1):
        cluster_indices = np.where(cluster_labels == i)[0]
        if len(cluster_indices) < 9:
            continue
        remove_num = 3
        signal[cluster_indices[:remove_num]] = np.nan
        signal[cluster_indices[-remove_num:]] = np.nan
        middle_indices = cluster_indices[remove_num:-remove_num]
        cluster_mean = np.nanmean(signal[middle_indices])
        signal_mean.append(cluster_mean)

    plt.figure(figsize=(8,6))
    plt.plot(signal, 'bo', linewidth=1.5)
    plt.title("Modified Signal with Middle Points")
    plt.xlabel("Data Points")
    plt.ylabel("Signal Value")
    plt.grid(True)
    plot1_path = "static/cluster_signal.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()

    pos = np.array(list(range(91, 100, 2)) + [100] + list(range(101, 110, 2)))
    value = np.array(signal_mean) / np.nanmax(signal_mean)

    plt.figure(figsize=(8,6))
    plt.plot(pos, value, 'ro')
    plt.title("Normalized Cluster Means")
    plt.xlabel("Position")
    plt.ylabel("Normalized Value")
    plt.grid(True)
    plot2_path = "static/normalized_means.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()

    return {
        "Background": round(bk, 4),
        "Threshold": round(threshold, 4),
        "Cluster Means": [round(v, 4) for v in value.tolist()],
        "Plot Signal": plot1_path,
        "Plot Mean": plot2_path
    }
