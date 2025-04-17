import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import geopandas as gpd
import pygeoops
import pandas as pd
import traceback
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import nearest_points, split
from shapely.affinity import rotate
from shapely.affinity import scale
from shapely.affinity import translate
from scipy.spatial import distance
from scipy.ndimage import uniform_filter1d
from shapely.ops import split, linemerge
from scipy.interpolate import splprep, splev
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from skimage.morphology import closing, remove_small_objects, square
from skimage import img_as_bool
from skimage import io, img_as_ubyte
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from openpyxl import Workbook
from tkinter import filedialog
from tkinter import Tk
from skimage.morphology import medial_axis
from skimage import io
from collections import Counter
from math import atan2, degrees



# Pixelgröße in µm
pixel_size = 0.08


# Klasse zur Darstellung von Konturen mit verschiedenen Attributen
class ContourWithAttributes:
    def __init__(self, contour, contour_id, centerline=None):
        self.contour_id = contour_id
        self.contour = contour
        self.length = cv2.arcLength(contour, True) * pixel_size
        self.area = cv2.contourArea(contour) * (pixel_size ** 2)
        self.width = self.area / self.length if self.length != 0 else 0
        self.center = self.compute_center()
        self.centerline= centerline
        self.scale_factor = max(self.length, self.width)
        self.signals = []


    def compute_center(self):
        M = cv2.moments(self.contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        return (cX, cY)

    def normalize_point(self, point):
        norm_x = (point[0] - self.center[0]) / self.scale_factor
        norm_y = (point[1] - self.center[1]) / self.scale_factor
        return (norm_x, norm_y)

    def add_signal(self, signal):
        self.signals.append(signal)


class SignalMaxima:
    def __init__(self, maxima_id, x, y, contour_id, norm_x=None, norm_y=None):
        self.maxima_id = maxima_id
        self.x = x
        self.y = y
        self.contour_id = contour_id
        self.norm_x = norm_x
        self.norm_y = norm_y


def select_files_and_directory():
    root = Tk()
    root.withdraw()

    # Selection of binary images
    input_files = filedialog.askopenfilenames(title="Select the binary images",
                                              filetypes=[("PNG files", "*.png"), ("All files", "*.*")])

    # Selection of fluorescence images
    fluorescence_files = filedialog.askopenfilenames(title="Select the fluorescence images",
                                                     filetypes=[("PNG files", "*.png"), ("All files", "*.*")])

    # Selection of the output directory
    output_directory = filedialog.askdirectory(title="Select the output directory")

    return input_files, fluorescence_files, output_directory

def extract_and_number_contours(input_files, output_folder):
    contour_folder = os.path.join(output_folder, 'contours')
    os.makedirs(contour_folder, exist_ok=True)
    json_folder = os.path.join(output_folder, 'json')
    os.makedirs(json_folder, exist_ok=True)

    contours_with_attributes = []
    contour_id = 1

    for image_path in input_files:
        filename = os.path.basename(image_path)
        original_image = cv2.imread(image_path)

        # Create an empty image with a white background
        image_with_contours = np.ones_like(original_image) * 255

        # Load the binary image and invert it
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        inverted_image = cv2.bitwise_not(image)

        # Apply thresholding to obtain black bacterial cells on a white background
        _, thresh = cv2.threshold(inverted_image, 128, 255, cv2.THRESH_BINARY)

        # Find the cell contour
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Create a user-defined legend element for contours
        contour_legend = mlines.Line2D([], [], color='black', label='Konturen')

        # Create a legend element for the center line
        centerline_legend = mlines.Line2D([], [], color='red', label='Mittellinie')

        plt.legend(handles=[contour_legend, centerline_legend])

        # Create objects for the contours with different attributes
        for contour in contours:

            contour = np.squeeze(contour)

            contour_line = LineString(contour)

            num_interpolated_points=20
            length = contour_line.length
            distances = np.linspace(0, length,len(contour) + (len(contour) - 1) * num_interpolated_points)

            # Interpolate new points along the contour line
            interpolated_points = [contour_line.interpolate(distance) for distance in distances]
            interpolated_contour = np.array([(point.x, point.y) for point in interpolated_points], dtype=np.float32)

            smoothed_contour = smooth_and_interpolate_contour(interpolated_contour)

            contour_attr = ContourWithAttributes(smoothed_contour, contour_id)
            contours_with_attributes.append(contour_attr)

            cv2.drawContours(image_with_contours, [contour], -1, (0, 255, 0), 2)

            # Number the contours in the picture
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            cv2.putText(image_with_contours, str(contour_id), (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Save the contours as a JSON file
            output_filename = f'contour_attributes_{filename}.json'
            output_path = os.path.join(json_folder, output_filename)
            contour_data = {
                "Contour_ID": contour_id,
                "Length": contour_attr.length,
                "Width": contour_attr.width,
                "Area": contour_attr.area,
                "Polygon": contour_attr.contour.tolist()
            }
            with open(output_path, 'w') as json_file:
                json.dump(contour_data, json_file, indent=4)

            contour_id += 1

        # Save the resulting image with the contours and numbering in the output folder
        contour_image_filename = f'contour_{filename}'
        contour_image_path = os.path.join(contour_folder, contour_image_filename)
        cv2.imwrite(contour_image_path, image_with_contours)

    return contours_with_attributes, inverted_image

def smooth_and_interpolate_contour(contour, num_points=200):

    contour_coords = contour.squeeze()
    x, y = contour_coords[:, 0], contour_coords[:, 1]

    # Spline-Vorbereitung und -Glättung
    tck, _ = splprep([x, y], s=4)  # Der Parameter 's' steuert den Glättungsgrad
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)

    # Kombiniere die interpolierten Punkte in eine neue Kontur
    smoothed_contour = np.vstack((x_new, y_new)).T
    return smoothed_contour.astype(np.float32)

def preprocess_image(binary_image, sigma=1, min_size=200):
    """
    This method smoothes the binary image and removes noise.

    - sigma: the standard deviation value for the Gaussian filter.
    - min_size: minimum size of objects that are retained after noise reduction.
    """
    # Smoothing the image with a Gaussian filter
    smoothed_image = gaussian(binary_image, sigma=sigma)

    # Threshold value for binary image (set all pixels > 0.5 to True)
    binary_smoothed = smoothed_image > 0.5

    # Remove small objects below a certain size (min_size)
    cleaned_image = remove_small_objects(binary_smoothed, min_size=min_size)

    # Convert to a Boolean image (True for foreground, False for background)
    return img_as_bool(cleaned_image)



def find_centerline_from_extremes(contour, centerline_points=75, min_opposite_distance=5):
    """
        Calculates a centerline based on the farthest points and adjusts it to the middle course of the contour.

        Args:
            contour (ndarray): The contour as (x, y) coordinates.
            centerline_points (int): The number of points on the centerline.

        Returns:
            centerline (LineString): The calculated centerline as a shapely LineString.
            centerline_coords (list): The coordinates of the centerline.
    """
    # 1. Calculate the furthest points on the contour
    max_distance = 0
    start_point, end_point = None, None

    # Run through all possible pairs of dots and find the pair with the maximum distance
    for i, point1 in enumerate(contour):
        for point2 in contour[i + 1:]:
            dist = distance.euclidean(point1, point2)
            if dist > max_distance:
                max_distance = dist
                start_point, end_point = point1, point2

    # 2. Divide the contour into two halves based on the start and end points
    half1, half2 = [], []
    add_to_first_half = True

    for pt in contour:
        half1.append(pt) if add_to_first_half else half2.append(pt)
        if np.array_equal(pt, start_point) or np.array_equal(pt, end_point):
            add_to_first_half = not add_to_first_half  # Wechsel bei Erreichen eines der Trennpunkte


    # 3. Create a rough center line between the maximum distant points
    line_x = np.linspace(start_point[0], end_point[0], centerline_points)
    line_y = np.linspace(start_point[1], end_point[1], centerline_points)
    rough_centerline_coords = list(zip(line_x, line_y))

    # 4. Calculate the center point for each point on the rough center line
    adjusted_centerline_coords = []
    for pt in rough_centerline_coords:
        # Bestimme die gegenüberliegenden Punkte aus den zwei Hälften
        distances_half1 = distance.cdist([pt], half1)
        distances_half2 = distance.cdist([pt], half2)

        closest_half1 = half1[np.argmin(distances_half1[0])]
        closest_half2 = half2[np.argmin(distances_half2[0])]

        # Berechne den Mittelpunkt der zwei gegenüberliegenden Punkte
        mid_x = (closest_half1[0] + closest_half2[0]) / 2
        mid_y = (closest_half1[1] + closest_half2[1]) / 2
        adjusted_centerline_coords.append((mid_x, mid_y))

    # Check whether there are enough points for the center line
    if len(adjusted_centerline_coords) < 2:
        raise ValueError("The calculated center line contains too few points.")

    # Create the final center line as a LineString
    centerline = LineString(adjusted_centerline_coords)
    return centerline, adjusted_centerline_coords


def straighten_centerline_and_relocate_signals(centerline, signal_data):
    """
    Transforms the curved centerline into a straight line of the same length and adjusts the signal positions, to maintain their relative position to the centerline.
        Args:
            centerline (LineString): The original (curved) centerline.
            signal_data (list): List of signal data as (distance_along_centerline, orthogonal_distance, x, y) tuples.

        Returns:
            straight_centerline (LineString): The straight centerline with the same length as the curved line.
            relocated_signals (list): List of the newly calculated signal coordinates as (x, y)-tuples.
    """
    # Calculate the length of the curved center line
    total_length = centerline.length

    # Define the straight centerline with the same length along the x-axis
    start_point = centerline.coords[0]
    end_point = (start_point[0] + total_length, start_point[1])  # Horizontally stretched
    straight_centerline = LineString([start_point, end_point])

    # Calculate new positions for the signals based on the relative positions
    relocated_signals = []
    for i, (distance_along_centerline, orthogonal_distance,  proj_x, proj_y, orig_signal_point) in enumerate(signal_data):

        # Find the point on the new, straight center line
        point_on_straight = straight_centerline.interpolate(distance_along_centerline)

        # Compare the y-coordinate of the signal with the y-coordinate of nearest_on_centerline
        if (orig_signal_point.y - proj_y)  < 0:
            orthogonal_distance = -orthogonal_distance  # Set the sign of orthogonal_distance correctly

        # Move the signal along the right angle to the straight center line
        new_x = point_on_straight.x + orthogonal_distance * (end_point[1] - start_point[1]) / total_length
        new_y = point_on_straight.y + orthogonal_distance

        relocated_signals.append((new_x, new_y, orig_signal_point.x, orig_signal_point.y))

    return straight_centerline, relocated_signals

def shift_to_origin(centerline, signals):
    """
    Moves the center line and the signal positions to the origin.
    """
    # Determine the center of the center line
    centerline_midpoint = centerline.interpolate(0.5, normalized=True)
    dx, dy = -centerline_midpoint.x, -centerline_midpoint.y

    # Move the center line
    shifted_centerline = translate(centerline, dx, dy)

    # Shift the signals relative to the center line and retain additional data
    shifted_signals = [
        (signal[0] + dx, signal[1] + dy, *signal[2:])  # Hier werden alle weiteren Werte beibehalten
        for signal in signals
    ]

    return shifted_centerline, shifted_signals

def normalize_centerline_length(centerline, signals):
    """
    Normalizes the length of the center line and adjusts the signal positions accordingly, whereby the original signal coordinates are retained.

        Args:
            centerline (LineString): The centerline to be normalized.
            signals (list): List of signal data as (distance_along_centerline, orthogonal_distance, x, y, original_x, original_y).

        Returns:
            normalized_centerline (LineString): The normalized centerline.
            normalized_signals (list): List of scaled signal coordinates as ((proj_x, proj_y), (orig_x, orig_y)) tuples.
    """
    # Calculate the scaling factors for the normalization of the center line length
    target_length = 100  # Example: Target distance (normalized length)
    scale_factor = target_length / centerline.length

    # Scale the center line
    normalized_centerline = scale(centerline, xfact=scale_factor, yfact=scale_factor, origin=centerline.centroid)

    # Scale the signals and save both the scaled and the original coordinates
    normalized_signals = [
        (signal[0] * scale_factor, signal[1], signal[2], signal[3])
        # Skaliertes proj_x, unverändertes proj_y, orig_x, orig_y
        for signal in signals
    ]

    return normalized_centerline, normalized_signals

def project_signals_onto_centerline(centerline, signal_positions):

    projected_signals = []
    for i, signal in enumerate(signal_positions) :

        if isinstance(signal, (tuple, list)) and len(signal) == 2:
            orig_signal_point = Point(signal)
        else:
            print(f"Ungültiges Signal an Position {i + 1}: {signal}")
            continue  # Skip invalid signals

        # Find the next point on the centerline to the signal
        nearest_on_centerline = nearest_points(centerline, orig_signal_point)[0]

        # Calculate the distance along the centerline from the starting point to the projection point
        distance_along_centerline = centerline.project(nearest_on_centerline)

        # Calculate the orthogonal distance between signal and projection
        orthogonal_distance = orig_signal_point.distance(nearest_on_centerline)

        # Add the projection information
        projected_signals.append((nearest_on_centerline.x, nearest_on_centerline.y, distance_along_centerline, orthogonal_distance, orig_signal_point))

    return projected_signals

def calculate_gaussian_curvature_with_radius(midline, radius):
    # Convert the center line into a numpy array
    midline_coords = np.array(midline)

    # Calculate the first and second derivatives with sliding window
    dx = np.gradient(midline_coords[:, 0])
    dy = np.gradient(midline_coords[:, 1])

    # Smooth the derivatives based on the radius
    dx_smoothed = np.convolve(dx, np.ones(2 * radius + 1) / (2 * radius + 1), mode='same')
    dy_smoothed = np.convolve(dy, np.ones(2 * radius + 1) / (2 * radius + 1), mode='same')

    ddx = np.gradient(dx_smoothed)
    ddy = np.gradient(dy_smoothed)

    # Calculate the curvature (Gaussian curvature)
    curvature = (dx_smoothed * ddy - dy_smoothed * ddx) / (dx_smoothed ** 2 + dy_smoothed ** 2) ** (3 / 2)

    # Scaling: pixel-to-micrometer conversion
    pixel_to_micrometer = 0.1  # Scaling factor in µm
    curvature_in_um2 = curvature * (1 / pixel_to_micrometer) ** 2

    return curvature_in_um2

def analyze_curvature_and_assign_signals(midline, signals):
    radius =5
    radius_window=7

    if isinstance(midline, LineString):
        midline_coords = np.array(midline.coords)
    else:
        midline_coords = np.array(midline)

    # Calculate the curvature along the center line
    curvature = calculate_gaussian_curvature_with_radius(midline_coords, radius)
    # Smoothing the curvature to reduce noise
    smoothed_curvature = gaussian_filter1d(curvature, sigma=1)

    # Calculate total curvature (separately for positive and negative curvature)
    positive_curvature_total = np.sum(smoothed_curvature[smoothed_curvature > 0])
    negative_curvature_total = np.sum(smoothed_curvature[smoothed_curvature < 0])

    # Determine regular positions for meter reading
    num_points = len(midline_coords)
    interval = max(1, num_points // 20)  # Choose 20 even distances (or adjust)
    regular_positions = np.arange(0, num_points, interval)
    curvature_values_midline = []

    for regular_position in regular_positions:
        # Read the curvature values at the regular positions
        curvature_values_mid = smoothed_curvature[regular_position]
        curvature_values_midline.append(curvature_values_mid)

    # Determine and assign the next curvature for each point
    signal_assignments = []
    curvature_values_at_signals = []
    for signal in signals:
        projected_x, projected_y = signal[0], signal[1]

        signal_coords = signal[4]  # <POINT (x y)>
        signal_point = Point(signal_coords.x, signal_coords.y)

        # Find the nearest point on the center line for the signal
        distances = np.sqrt((midline_coords[:, 0] - signal[0]) ** 2 + (midline_coords[:, 1] - signal[1]) ** 2)
        nearest_idx = np.argmin(distances)

        # Determine a window around the next point
        start_idx = max(nearest_idx - radius_window, 0)
        end_idx = min(nearest_idx + radius_window + 1, len(midline_coords))
        window_coords = midline_coords[start_idx:end_idx]

        signal_to_midline_vector = np.array([projected_x, projected_y]) - np.array([signal_point.x, signal_point.y])

        signal_to_midline_vector /= np.linalg.norm(signal_to_midline_vector) # Normalize

        # Angle calculation for each point in the window
        angles = []
        for midline_point in window_coords:
            signal_to_point_vector = np.array([midline_point[0],midline_point[1]]) - np.array([projected_x,projected_y])

            signal_to_point_vector /= np.linalg.norm(signal_to_point_vector)  # Normalize

            # Calculate the angle between the two vectors
            dot_product = np.dot(signal_to_midline_vector, signal_to_point_vector)
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Winkel (in Radiant)
            if not np.isnan(angle):
                angles.append(np.degrees(angle))

        # Calculate the mean value of the angles
        mean_angle = 180-(np.mean(angles))

        # Determine the curvature based on the angle
        if mean_angle < 90:  # Negative Krümmung

            curvature_value = -abs(smoothed_curvature[nearest_idx])
        else:
            curvature_value = abs(smoothed_curvature[nearest_idx])

        # Determine curvature value and sign
        curvature_type = "positive" if curvature_value > 0 else "negative"

        # Save assignment and curvature value on the signal
        signal_assignments.append((signal, curvature_value, curvature_type))
        curvature_values_at_signals.append(curvature_value)

    # Calculate the relative frequency of each curvature value
    signal_counts = Counter(curvature_values_at_signals)

    # Determine the highest frequency to perform the normalization
    if signal_counts:
        max_frequency = max(signal_counts.values())
        relative_frequencies = {k: v / max_frequency for k, v in signal_counts.items()}
    else:
        # Falls signal_counts leer ist, relative_frequencies als leeres Dictionary definieren
        relative_frequencies = {}

    return {
        "curvature_value": curvature_values_at_signals,
        "curvature_value_midline": curvature_values_midline,
        "positive_curvature_total": positive_curvature_total,
        "negative_curvature_total": negative_curvature_total,
        "signal_assignments": signal_assignments,
        "relative_frequencies": relative_frequencies
    }

def write_curvature_frequencies_to_excel(all_curvature_data, output_folder, filename="curvature_frequencies.xlsx"):

    curvature_values = []
    curvature_midline_values = []

    # Iteriere über die Daten
    for data in all_curvature_data:
        if isinstance(data, dict):
            # Extrahiere Werte von "curvature_value"
            if "curvature_value" in data:
                values = data["curvature_value"]
                if isinstance(values, list):
                    curvature_values.extend([float(v) for v in values])
                else:
                    curvature_values.append(float(values))

            if "curvature_value_midline" in data:
                midline_values = data["curvature_value_midline"]
                if isinstance(midline_values, list):
                    curvature_midline_values.extend([float(v) for v in midline_values])
                else:
                    curvature_midline_values.append(float(midline_values))

    curvature_values = sorted(curvature_values)
    curvature_midline_values = sorted(curvature_midline_values)

    # Ensure that both lists are the same length
    max_length = max(len(curvature_values), len(curvature_midline_values))
    curvature_values += [None] * (max_length - len(curvature_values))
    curvature_midline_values += [None] * (max_length - len(curvature_midline_values))

    df = pd.DataFrame({
        "curvature_value": curvature_values,
        "curvature_value_midline": curvature_midline_values,
    })

    file_path = os.path.join(output_folder, filename)

    df = df.sort_index()
    df.to_excel(file_path, index=False)

    print(f"The data was successfully saved in '{filename}'.")

def save_signals_to_excel_new(signals, output_folder, file_name="normalized_signals.xlsx"):

    file_path = os.path.join(output_folder, file_name)

    # Erstelle die Excel-Tabelle
    data = [
        {
            "Projected_X": proj_x,
            "Projected_Y": proj_y,
            "Original_X": orig_x,
            "Original_Y": orig_y,
        }
        for proj_x, proj_y, orig_x, orig_y in signals
    ]

    # Save the data in an Excel file
    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)

    print(f"File saved successfully: {file_name}")

def detect_maxima(image, sigma=2, threshold=0.01, min_distance=5):
    blurred_image = gaussian(image, sigma=sigma)
    maxima = peak_local_max(blurred_image, min_distance=min_distance, threshold_abs=threshold)
    return maxima


def mark_maxima(image, maxima_positions):
    marked_image = cv2.cvtColor(img_as_ubyte(image), cv2.COLOR_GRAY2BGR)
    for (y, x) in maxima_positions:
        cv2.drawMarker(marked_image, (int(x), int(y)), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10,
                       thickness=2)
    return marked_image


def overlay_markers_on_contours(contour_image, maxima_positions):
    for (y, x) in maxima_positions:
        contour_image = cv2.drawMarker(contour_image, (int(x), int(y)), (255, 0, 0), markerType=cv2.MARKER_CROSS,
                                       markerSize=10, thickness=2)
    return contour_image


def save_contours_to_excel(contours, output_path):

    # Create an Excel Writer object with pandas
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Iterate over the contours
        for idx, contour in enumerate(contours):
            # Extract the X and Y coordinates of the contour
            x_coords = contour.contour[:, 0]
            y_coords = contour.contour[:, 1]

            # Create a DataFrame with the coordinates
            df = pd.DataFrame({
                'X': x_coords,
                'Y': y_coords
            })

            sheet_name = f'Kontur_{idx + 1}'

            df.to_excel(writer, sheet_name=sheet_name, index=False)

def display_image_with_maxima(image, maxima_positions, sigma, threshold, min_distance):
    marked_image = mark_maxima(image, maxima_positions)
    plt.imshow(marked_image, cmap='gray')
    plt.title(f'Sigma: {sigma}, Threshold: {threshold}, Min Distance: {min_distance}')
    plt.show()

def get_user_input(sigma, threshold, min_distance):
    sigma = float(input(f"Enter new sigma (current: {sigma}): ") or sigma)
    threshold = float(input(f"Enter new threshold (current: {threshold}): ") or threshold)
    min_distance = int(input(f"Enter new minimum distance (current: {min_distance}): ") or min_distance)
    return sigma, threshold, min_distance

def process_images():
    input_files, fluorescence_files, output_folder = select_files_and_directory()

    all_signals = []
    all_centerlines = []
    all_adjusted_signals = []

    contours, inverted_image = extract_and_number_contours(input_files, output_folder)
    # Set path to the Excel file
    output_path_con = os.path.join(output_folder, "kontur_koordinaten.xlsx")

    # Save the contours in an Excel spreadsheet
    save_contours_to_excel(contours, output_path_con)

    cleaned_image=preprocess_image(inverted_image)

    for fluorescence_image_path in fluorescence_files:
        filename = os.path.basename(fluorescence_image_path)
        image = io.imread(fluorescence_image_path, as_gray=True)

        # Initial values for sigma, threshold, and min_distance
        sigma = 1
        threshold = 0.02
        min_distance = 1

        while True:
            maxima_positions = detect_maxima(image, sigma=sigma, threshold=threshold, min_distance=min_distance)
            display_image_with_maxima(image, maxima_positions, sigma, threshold, min_distance)

            user_response = input("Keep current parameters? (y/n): ")
            if user_response.lower() == 'y':
                break

            sigma, threshold, min_distance = get_user_input(sigma, threshold, min_distance)

        maxima_id = 1
        for (y, x) in maxima_positions:
            for contour in contours:
                if cv2.pointPolygonTest(contour.contour, (float(x), float(y)), False) >= 0:
                    norm_x, norm_y = contour.normalize_point((x, y))
                    signal = SignalMaxima(maxima_id, x, y, contour.contour_id, norm_x, norm_y)
                    contour.add_signal(signal)
                    all_signals.append(signal)

                    maxima_id += 1

        all_curvature_data = []
        for contour in contours:
            enhanced_signals = []

            centerline, centerline_coords = find_centerline_from_extremes(contour.contour)
            contour.centerline = centerline

            signals_in_contour = {(signal.x, signal.y): signal for signal in all_signals if signal.contour_id == contour.contour_id}
            signal_positions = list(signals_in_contour.keys())  # Nur die Positionen ohne Duplikate extrahieren

            # Project the signals onto the center line of the contour
            projected_signals = project_signals_onto_centerline(centerline, signal_positions)

            results = analyze_curvature_and_assign_signals(centerline, projected_signals)
            all_curvature_data.append(results)

            for proj_x, proj_y, distance_along_centerline, orthogonal_distance,orig_signal_point in projected_signals:
                enhanced_signals.append((distance_along_centerline, orthogonal_distance, proj_x, proj_y, orig_signal_point))

            # Smooth centerline and adjust signals
            smoothed_centerline, relocated_signals = straighten_centerline_and_relocate_signals(centerline, enhanced_signals)

            unique_relocated_signals = []
            seen_coords = set()

            for proj_x, proj_y, orig_x, orig_y in relocated_signals:
                if (proj_x, proj_y) not in seen_coords:
                    seen_coords.add((proj_x, proj_y))
                    unique_relocated_signals.append((proj_x, proj_y, orig_x, orig_y))

            #Move centerline and signals to the origin
            shifted_centerline, shifted_signals = shift_to_origin(smoothed_centerline, unique_relocated_signals)

            all_centerlines.append(shifted_centerline)
            all_adjusted_signals.append(shifted_signals)

    normalized_centerlines = []
    normalized_signals = []

    for i, (centerline, signals) in enumerate(zip(all_centerlines, all_adjusted_signals)):
        norm_centerline, norm_signals = normalize_centerline_length(centerline, signals)
        normalized_centerlines.append(norm_centerline)
        normalized_signals.extend(norm_signals)  # Optional: für alle Signale auf einmal

    unique_signals = list(set(normalized_signals))

    save_signals_to_excel_new(unique_signals, output_folder)

    marked_image = mark_maxima(image, maxima_positions)
    marked_contour_image = overlay_markers_on_contours(marked_image, maxima_positions)

    # Speichern Sie das Bild mit maxima
    result_filename = f'marked_{filename}'
    result_path = os.path.join(output_folder, result_filename)
    io.imsave(result_path, img_as_ubyte(marked_contour_image))

    write_curvature_frequencies_to_excel(all_curvature_data, output_folder)

if __name__ == "__main__":
    process_images()
