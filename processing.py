# processing.py

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

# Define double-sided sigmoid model
def double_sided_sigmoid(x, A, k, x1, x2, B):
    return A / (1 + np.exp(-k * (x - x1))) - A / (1 + np.exp(-k * (x - x2))) + B

def process_file(file_path, plot_title="Beam Profile with FWHM and Penumbra", 
                 x_label="Position", y_label="Intensity"):
    # Load the Excel data
    df = pd.read_excel(file_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        raise ValueError("Uploaded file must have at least two numeric columns.")

    x = df[numeric_cols[0]].values
    y = df[numeric_cols[1]].values

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

    left_x_50 = x_fine[left_idx_50]
    right_x_50 = x_fine[right_idx_50]
    fwhm = right_x_50 - left_x_50

    range_val = max_val - min_val
    level_20 = min_val + 0.2 * range_val
    level_80 = min_val + 0.8 * range_val

    left_idx_20 = np.where(fitted_curve[:np.argmax(fitted_curve)] >= level_20)[0][0]
    left_idx_80 = np.where(fitted_curve[:np.argmax(fitted_curve)] >= level_80)[0][0]
    right_idx_80 = np.where(fitted_curve[np.argmax(fitted_curve):] <= level_80)[0][0] + np.argmax(fitted_curve)
    right_idx_20 = np.where(fitted_curve[np.argmax(fitted_curve):] <= level_20)[0][0] + np.argmax(fitted_curve)

    left_x_20 = x_fine[left_idx_20]
    left_x_80 = x_fine[left_idx_80]
    right_x_80 = x_fine[right_idx_80]
    right_x_20 = x_fine[right_idx_20]

    left_penumbra = left_x_80 - left_x_20
    right_penumbra = right_x_20 - right_x_80

    # === Now plot ===
    plt.figure(figsize=(8,6))
    plt.plot(x, y, 'b.', markersize=10, label="Data")
    plt.plot(x_fine, fitted_curve, 'r-', linewidth=2, label="Sigmoid Fit")

    plt.plot([left_x_50, right_x_50], [half_max, half_max], 'g--', linewidth=2, label="FWHM")
    plt.plot([left_x_20, left_x_80], [level_20, level_80], 'm-', linewidth=2, label="Left Penumbra (20%-80%)")
    plt.plot([right_x_80, right_x_20], [level_80, level_20], 'c-', linewidth=2, label="Right Penumbra (80%-20%)")
    plt.plot(center_x, center_y, 'ko', markersize=10, label="Center")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_filename = "static/plot.png"
    os.makedirs("static", exist_ok=True)
    plt.savefig(plot_filename)
    plt.close()

    result = {
        "Amplitude (A)": popt[0],
        "Steepness (k)": popt[1],
        "Rising Edge (x1)": popt[2],
        "Falling Edge (x2)": popt[3],
        "Baseline (B)": popt[4],
        "Center": center_x,
        "FWHM": fwhm,
        "Left Penumbra (20%-80%)": left_penumbra,
        "Right Penumbra (80%-20%)": right_penumbra,
    }
    return result
