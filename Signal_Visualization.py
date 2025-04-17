import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import hdbscan
from sklearn.cluster import DBSCAN, KMeans
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon, Point
from shapely.affinity import scale, translate
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from svgpathtools import svg2paths
from sklearn.preprocessing import StandardScaler
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon as ShapelyPolygon
from scipy.spatial import ConvexHull, QhullError
from matplotlib import cm  # Für Farbskala


def load_excel_file():
    """Load an Excel file using a file dialog"""
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    return pd.read_excel(filename)


def save_excel_file(df, default_filename):
    """Save a DataFrame to an Excel file using a file dialog"""
    Tk().withdraw()
    filename = asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")],
                                 initialfile=default_filename)
    if filename:
        df.to_excel(filename, index=False)


def load_svg_contour(svg_path, num_points=100):
    """Load an SVG file and extract the contour points"""
    paths, _ = svg2paths(svg_path)
    all_points = []
    for path in paths:
        for segment in path:
            points = [segment.point(t) for t in np.linspace(0, 1, num_points)]
            all_points.extend(points)
    all_points = np.array([(p.real, p.imag) for p in all_points])
    return all_points

def get_svg_path(svg_filename):
    # Aktuellen Skriptpfad abrufen
    script_dir = os.path.dirname(os.path.abspath(__file__))
    svg_path = os.path.join(script_dir, svg_filename)
    return svg_path


def create_scaled_polygon(contour):
    # Ensure that the contour is available in a suitable format
    try:
        polygon = ShapelyPolygon(contour)
        if not polygon.is_valid:
            print("Invalid contour. Check for duplicates or invalid points.")
            return None
        if polygon.is_empty:
            print("The polygon is empty.")
            return None

        # Polygon skalieren
        expanded_polygon = scale(polygon, xfact=1.2, yfact=1.2)
        return expanded_polygon
    except Exception as e:
        print(f"Error when creating or scaling the polygon:{e}")
        return None

def normalize_contour(contour):
    """Normalize the contour to set its center to (0,0)"""
    center_x = (contour[:, 0].min() + contour[:, 0].max()) / 2
    center_y = (contour[:, 1].min() + contour[:, 1].max()) / 2
    contour[:, 0] -= center_x
    contour[:, 1] -= center_y

    max_x_extent = max(abs(contour[:, 0].min()), abs(contour[:, 0].max()))
    max_y_extent = max(abs(contour[:, 1].min()), abs(contour[:, 1].max()))

    contour[:, 0] = contour[:, 0] / max_x_extent * max(max_x_extent, max_y_extent)
    contour[:, 1] = contour[:, 1] / max_y_extent * max(max_x_extent, max_y_extent)

    return contour


def adjust_contour_to_extremes(contour, signal_maxima):
    """Adjust the contour to ensure it fits within the extrema of the signal maxima"""
    max_x_point, min_x_point, max_y_point, min_y_point = find_extreme_points(signal_maxima)

    # Define new bounds for the contour based on the extrema points
    #min_x = min_x_point['Projected_X']
    #max_x = max_x_point['Projected_X']
    #min_y = signal_maxima['Projected_Y']
    #max_y = signal_maxima['Projected_Y']

    min_x = -51
    max_x = +51
    min_y = -4.6
    max_y = +4.6

    # Scale the contour to fit within these bounds
    contour_polygon = ShapelyPolygon(contour)  # Kontur als Shapely-Polygon
    minx, miny, maxx, maxy = contour_polygon.bounds

    scale_x = (max_x - min_x) / (maxx - minx)
    scale_y = (max_y - min_y) / (maxy - miny)

    adjusted_contour = scale(contour_polygon, xfact=scale_x, yfact=scale_y, origin='center')
    adjusted_contour = translate(adjusted_contour, xoff=min_x - minx * scale_x, yoff=min_y - miny * scale_y)

    return np.array(adjusted_contour.exterior.coords)


def find_extreme_points(signal_maxima):
    """Find the points that are furthest from the X and Y axes"""
    max_x_point = signal_maxima.loc[signal_maxima['Projected_X'].idxmax()]
    min_x_point = signal_maxima.loc[signal_maxima['Projected_X'].idxmin()]
    max_y_point = signal_maxima.loc[signal_maxima['Projected_Y'].idxmax()]
    min_y_point = signal_maxima.loc[signal_maxima['Projected_Y'].idxmin()]
    return max_x_point, min_x_point, max_y_point, min_y_point

def optimal_bins(data):
    iqr = np.percentile(data, 75) - np.percentile(data, 25)  # Interquartilabstand
    bin_width = 2 * iqr / len(data)**(1/3)  # Freedman-Diaconis-Regel
    return int((data.max() - data.min()) / bin_width)

def create_heatmap(signal_maxima, contour, ax):
    # Berechne Bins für X und Y
    bins_x = optimal_bins(signal_maxima['Projected_X'])
    bins_y = optimal_bins(signal_maxima['Projected_Y'])

    # Finales bins-Tupel
    bins = (bins_x, bins_y)
    heatmap_data, xedges, yedges = np.histogram2d(signal_maxima['Projected_X'], signal_maxima['Projected_Y'],
                                                  bins=bins)
    heatmap_data = gaussian_filter(heatmap_data, sigma = max(1, min(bins) // 100))

    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    points = np.vstack((X.ravel(), Y.ravel())).T
    expanded_contour = scale(Polygon(contour), xfact=1.2, yfact=1.2)
    mask = np.array([expanded_contour.contains(Point(p)) for p in points])
    #mask = np.array([Polygon(contour).contains(Point(p)) for p in points])
    mask = mask.reshape(heatmap_data.shape)
    heatmap_data = np.ma.masked_where(~mask, heatmap_data)

    pcm = ax.pcolormesh(xedges, yedges, heatmap_data.T, cmap='viridis', shading='auto')
    ax.plot(contour[:, 0], contour[:, 1], color='white', linewidth=2)
    ax.set_facecolor("black")
    x_min, x_max = signal_maxima['Projected_X'].min(), signal_maxima['Projected_X'].max()
    y_min, y_max = signal_maxima['Projected_Y'].min(), signal_maxima['Projected_Y'].max()
    ax.set_xlim(x_min - (x_max - x_min) * 0.1, x_max + (x_max - x_min) * 0.1)
    ax.set_ylim(y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1)
    ax.axis('off')
    return pcm

def plot_with_histograms(signals, contour, bins):
    # Main scatter plot of signals with contour
    fig, ax_main = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(top=0.8, bottom=0.3, left=0.205, right=0.8)
    ax_main.scatter(signals['Projected_X'], signals['Projected_Y'], color='blue', s=5, edgecolor="black", linewidth=0.5)
    ax_main.plot(contour[:, 0], contour[:, 1], color='black', linewidth=2)
    ax_main.axis('off')

    # X-axis histogram (top)
    ax_xhist = ax_main.inset_axes([0, 1.05, 1, 0.2], transform=ax_main.transAxes)
    ax_xhist.hist(signals['Projected_X'], bins=bins, color='skyblue', edgecolor="black")  # Corrected for X-axis histogram
    ax_xhist.set_ylabel("Frequency (X-axis)")
    ax_xhist.yaxis.set_label_position("right")  # Move label to the right side
    ax_xhist.yaxis.tick_right()  # Move ticks to the right side
    ax_xhist.spines['top'].set_visible(False)
    ax_xhist.spines['right'].set_visible(False)
    ax_xhist.get_xaxis().set_visible(False)  # Hide x-axis labels



    # Y-axis histogram as bar chart (left)
    ax_yhist = ax_main.inset_axes([-0.25, 0, 0.2, 1], transform=ax_main.transAxes)
    ax_yhist.barh(range(bins), np.histogram(signals['Projected_Y'], bins=bins)[0], color='skyblue',edgecolor="black")  # Corrected for Y-axis histogram
    ax_yhist.set_xlabel("Frequency (Y-axis)")
    ax_yhist.spines['top'].set_visible(False)
    ax_yhist.spines['right'].set_visible(False)
    ax_yhist.get_yaxis().set_visible(False)  # Hide y-axis labels



def plot_dbscan_clusters(signals, contour):
    fig, ax = plt.subplots(figsize=(8, 6))
    data = StandardScaler().fit_transform(signals[['Projected_X', 'Projected_Y']])
    dbscan = DBSCAN(eps=0.075, min_samples=5).fit(data)
    db_labels = dbscan.labels_

    ax.scatter(signals['Projected_X'], signals['Projected_Y'], c=db_labels, cmap='tab10', s=5)
    ax.plot(contour[:, 0], contour[:, 1], color='black')
    ax.axis('off')
    ax.set_title("DBSCAN Clustering")
    plt.show()



def plot_hdbscan_clusters(signals, contour, scale_factor=10):
    # Standardization of data
    data = StandardScaler().fit_transform(signals[['Projected_X', 'Projected_Y']])

    # Initialize and adapt HDBSCAN model
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=10, cluster_selection_epsilon=0.075)
    clusterer.fit(data)

    # Retrieve labels and probabilities
    hdb_labels = clusterer.labels_
    probabilities = clusterer.probabilities_

    # Unique cluster labels
    unique_labels = set(hdb_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("black")  # Hintergrund schwarz setzen

    # Define color scale
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Noise points
            noise_indices = np.where(hdb_labels == -1)[0]
            ax.scatter(
                signals.iloc[noise_indices]['Projected_X'],
                signals.iloc[noise_indices]['Projected_Y'],
                c="black", marker="x", s=10, label="Noise"
            )
        else:
            # Cluster points
            cluster_indices = np.where(hdb_labels == k)[0]
            cluster_points = data[cluster_indices]

            # Draw dots
            ax.scatter(
                signals.iloc[cluster_indices]['Projected_X'],
                signals.iloc[cluster_indices]['Projected_Y'],
                color=tuple(col), edgecolor="black", linewidth=0.5,
                s=20,
                label=f"Cluster {k + 1}"
            )

            # Calculate and draw convex hull
            if len(cluster_points) >= 3:
                try:
                    hull = ConvexHull(cluster_points)

                    # Scale convex hull (X-coordinates only)
                    hull_points = signals.iloc[cluster_indices][['Projected_X', 'Projected_Y']].to_numpy()
                    hull_points[:, 0] *= scale_factor  # Nur x-Werte skalieren

                    # Extract points for polygon
                    polygon_points = hull_points[hull.vertices] / [scale_factor, 1]

                    # Draw the border separately (for safety)
                    for simplex in hull.simplices:
                        ax.plot(
                            hull_points[simplex, 0] / scale_factor,
                            hull_points[simplex, 1],
                            color=tuple(np.array(col) * 1),
                            linewidth=1.5
                        )
                except Exception as e:

                    continue

                except Exception as e:

                    continue

    # Kontur zeichnen
    ax.plot(contour[:, 0], contour[:, 1], color='black', linewidth=2)

    # Achsen ausblenden
    ax.axis('off')

    # Title with cluster information
    ax.set_title(f"Estimated number of clusters: {n_clusters}")

    # Show plot
    plt.show()

def plot_kmeans_clusters(signals, contour):
    fig, ax = plt.subplots(figsize=(8, 6))
    data = StandardScaler().fit_transform(signals[['Projected_X', 'Projected_Y']])
    kmeans = KMeans(n_clusters=20, random_state=0).fit(data)
    kmeans_labels = kmeans.labels_

    ax.scatter(signals['Projected_X'], signals['Projected_Y'], c=kmeans_labels, cmap='tab10', s=5, edgecolor="black", linewidth=0.5)
    ax.plot(contour[:, 0], contour[:, 1], color='white')
    ax.axis('off')
    ax.set_title("K-Means Clustering")
    plt.show()

print("Select the Excel file for Signal Maxima")
signal_maxima_df = load_excel_file()
contour_svg_path = get_svg_path("Cell_contour.svg")
contour = load_svg_contour(contour_svg_path)
normalized_contour = normalize_contour(contour)
adjusted_contour = adjust_contour_to_extremes(normalized_contour, signal_maxima_df)

# Display each plot individually
fig, ax = plt.subplots(figsize=(20,14))
plot_with_histograms(signal_maxima_df, adjusted_contour, bins=50)
create_heatmap(signal_maxima_df, adjusted_contour, ax)
plot_hdbscan_clusters(signal_maxima_df, adjusted_contour)

