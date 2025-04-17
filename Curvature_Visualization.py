import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog


def load_excel_files():
    """Loads several Excel files via a pop-up window."""
    Tk().withdraw()  # Prevents the appearance of a Tkinter main window
    file_paths = filedialog.askopenfilenames(title="Select Excel files",
                                             filetypes=[("Excel files", "*.xlsx")])
    return list(file_paths)


def find_global_min_max(file_paths, column_index):
    """Finds the global minimum and maximum across all files for a given column."""
    global_min = float('inf')
    global_max = float('-inf')

    for file_path in file_paths:
        df = pd.read_excel(file_path)
        data = df.iloc[:, column_index].dropna().values
        global_min = min(global_min, data.min())
        global_max = max(global_max, data.max())

    return global_min, global_max


def generate_bins(global_min, global_max, bin_width):
    """Creates automatic bins based on minimum, maximum and bin width."""
    return np.arange(global_min, global_max + bin_width, bin_width)


def count_values_in_bins(data, bins):
    """Counts how many values fall into each range (bin)."""
    counts, _ = np.histogram(data, bins=bins)
    return counts


def normalize_counts(counts):
    """Normalizes the frequencies to a sum of 1."""
    return counts / counts.sum()


def process_replicates(file_paths, bins, column_index):
    """Processes the Excel replicates and calculates the average and standard deviation for each range."""
    all_normalized_counts = []
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Centers of the bins

    for file_path in file_paths:
        df = pd.read_excel(file_path)
        data = df.iloc[:, column_index].dropna().values
        counts = count_values_in_bins(data, bins)
        normalized_counts = normalize_counts(counts)
        all_normalized_counts.append(normalized_counts)

    all_normalized_counts = np.array(all_normalized_counts)
    mean_counts = np.mean(all_normalized_counts, axis=0)
    std_counts = np.std(all_normalized_counts, axis=0)

    return bin_centers, mean_counts, std_counts


def plot_histogram(bin_centers1, mean_counts1, std_counts1, bin_centers2, mean_counts2, std_counts2):
    """Creates a line chart with average and standard deviations for both columns."""
    plt.figure(figsize=(12, 8))

    # Average and standard deviation lines for column 1
    plt.plot(bin_centers1, mean_counts1, color='deepskyblue', label='Mean Curvature- Signal', linewidth=2)
    upper_bound1 = mean_counts1 + std_counts1
    lower_bound1 = mean_counts1 - std_counts1
    plt.fill_between(bin_centers1, lower_bound1, upper_bound1, color='skyblue', alpha=0.5, label='StdDev Signal')

    # Average and standard deviation lines for column 2
    plt.plot(bin_centers2, mean_counts2, color='darkgreen', label='Mean Curvature- midlane', linewidth=2)
    upper_bound2 = mean_counts2 + std_counts2
    lower_bound2 = mean_counts2 - std_counts2
    plt.fill_between(bin_centers2, lower_bound2, upper_bound2, color='lightgreen', alpha=0.5, label='StdDev midlane')

    # Axis title and legend
    plt.title("relative frequency of curvatures", fontsize=14)
    plt.xlabel("curvature", fontsize=12)
    plt.ylabel("relative frequency", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    file_paths = load_excel_files()
    if not file_paths:
        print("No files selected. The program is terminated.")
        return

    # Processing for column 1
    global_min1, global_max1 = find_global_min_max(file_paths, column_index=0)
    print(f"Column 1 - Global minimum: {global_min1}, Global minimum: {global_max1}")

    # Processing for column 2
    global_min2, global_max2 = find_global_min_max(file_paths, column_index=1)
    print(f"Column 2 - Global minimum: {global_min2}, Global maximum: {global_max2}")

    # Enter user-defined bin width
    bin_width = float(input("Enter the width of the number ranges (bin width) (e.g. 0.5):"))

    bins1 = generate_bins(global_min1, global_max1, bin_width)
    bins2 = generate_bins(global_min2, global_max2, bin_width)

    # Calculation of mean values and standard deviations for both columns
    bin_centers1, mean_counts1, std_counts1 = process_replicates(file_paths, bins1, column_index=0)
    bin_centers2, mean_counts2, std_counts2 = process_replicates(file_paths, bins2, column_index=1)

    # Plot of the results
    plot_histogram(bin_centers1, mean_counts1, std_counts1, bin_centers2, mean_counts2, std_counts2)


if __name__ == "__main__":
    main()
