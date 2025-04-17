import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable, get_cmap
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from collections import defaultdict
import re
from svgpathtools import svg2paths
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Polygon, Point
from shapely.affinity import scale, translate

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    svg_path = os.path.join(script_dir, svg_filename)
    return svg_path

def create_scaled_polygon(contour):
    # Ensure that the contour is available in a suitable format
    try:
        polygon = ShapelyPolygon(contour)
        if not polygon.is_valid:
            print("Ungültige Kontur. Prüfe auf Duplikate oder ungültige Punkte.")
            return None
        if polygon.is_empty:
            print("Das Polygon ist leer.")
            return None

        # Polygon skalieren
        expanded_polygon = scale(polygon, xfact=1.2, yfact=1.2)
        return expanded_polygon
    except Exception as e:
        print(f"Fehler beim Erstellen oder Skalieren des Polygons: {e}")
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


def adjust_contour_to_extremes(contour):
    """Adjust the contour to ensure it fits within the extrema of the signal maxima"""

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

# Function for parsing the Normalized_Coordinates column
def parse_coordinates(coord_str):
    try:
        # Remove outer brackets and spaces
        coord_str = coord_str.strip("[]")
        # Extracting the coordinate pairs
        coord_pairs = re.findall(r"\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)", coord_str)
        # Convert the string pair into tuples of floats
        coordinates = [tuple(map(float, pair)) for pair in coord_pairs]
        print(f"Parsed coordinates: {coordinates}")  # Debug-Ausgabe der Koordinaten
        return coordinates
    except (SyntaxError, ValueError) as e:
        print(f"Fehler beim Parsen der Koordinaten: {coord_str} - {e}")
        return []


# Function for plotting the tracks
def plot_tracks(tracks, contour, title, displacement_range, show_arrows=True):
    if not tracks:
        print(f"Keine gültigen Tracks zum Plotten für {title}")
        return

    # Determine the value range
    displacement_min = min(track['displacement'] for track in tracks)
    displacement_max = max(track['displacement'] for track in tracks)

    # Color scale based on the global displacement range
    norm = Normalize(vmin=displacement_min, vmax=displacement_max)
    cmap = plt.get_cmap('coolwarm')  # Farbskala z. B. von Blau (kalt) nach Rot (warm)

    # Define sub-areas
    static_range = (displacement_min, displacement_min + (displacement_max - displacement_min) * 0.33)
    fast_mobile_range = (static_range[1], static_range[1] + (displacement_max - displacement_min) * 0.33)
    mobile_range = (fast_mobile_range[1], displacement_max)

    # Sub-color scales for categories
    static_norm = Normalize(vmin=static_range[0], vmax=static_range[1])
    fast_mobile_norm = Normalize(vmin=fast_mobile_range[0], vmax=fast_mobile_range[1])
    mobile_norm = Normalize(vmin=mobile_range[0], vmax=mobile_range[1])

    # Creation of the color bar separately
    fig_colorbar, ax_colorbar = plt.subplots(figsize=(6, 1))
    cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_colorbar, orientation='horizontal')
    cb1.set_label("Track Displacement (µm)")
    fig_colorbar.tight_layout()
    plt.show()


    # Create the subplots
    fig, ax_main = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(top=0.8, bottom=0.3, left=0.7, right=0.8)

    all_coords = []
    y_locations = []

    for track in tracks:
        coordinates = track['coordinates']

        displacement = track['displacement']
        if displacement <= static_range[1]:
            color = ScalarMappable(norm=static_norm, cmap=cmap).to_rgba(displacement)
        elif displacement <= fast_mobile_range[1]:
            color = ScalarMappable(norm=fast_mobile_norm, cmap=cmap).to_rgba(displacement)
        else:
            color = ScalarMappable(norm=mobile_norm, cmap=cmap).to_rgba(displacement)


        #displacement = track['displacement']
        if not coordinates:
            continue  # Überspringt leere oder fehlerhafte Koordinaten

        # Extract X and Y coordinates
        x, y = zip(*coordinates)
        all_coords.extend(x)
        y_locations.extend(y)

        # Farbe basierend auf TRACK_DISPLACEMENT
        #displacement = track['displacement']
        #color = cmap(norm(displacement))
        ax_main.plot(x, y, color=color, linewidth=2)  # Add connecting lines

        # Optional: Pfeile anzeigen
        if show_arrows and len(coordinates) > 1:
            for j in range(1, len(coordinates)):
                ax_main.arrow(x[j - 1], y[j - 1], x[j] - x[j - 1], y[j] - y[j - 1],
                            head_width=0.05, head_length=0.1, fc=color, ec=color)

    # Histogram of the X-coordinates
    dummy_values_x = [-51, 51]  # Dummy-Werte für x-Achse
    all_coords = np.concatenate([all_coords, dummy_values_x])
    #bins = np.linspace(-60, 60, 50)

    ax_xhist = ax_main.inset_axes([0, 1.05, 1, 0.2], transform=ax_main.transAxes)
    ax_xhist.hist(all_coords, bins=70, color='skyblue', edgecolor='black')
    ax_xhist.set_ylabel("Frequency (X-axis)")
    ax_xhist.yaxis.set_label_position("right")  # Move label to the right side
    ax_xhist.yaxis.tick_right()  # Move ticks to the right side
    ax_xhist.spines['top'].set_visible(False)
    ax_xhist.spines['right'].set_visible(False)
    ax_xhist.get_xaxis().set_visible(False)

    ax_xhist.set_aspect(ax_main.get_aspect())  # Adjust the aspect ratio

    dummy_values_y = [-4.5, 4.5]
    y_locations = np.concatenate([y_locations, dummy_values_y])
    #print(f"y-locations: {y_locations}")
    ax_yhist = ax_main.inset_axes([-0.25, 0, 0.2, 1], transform=ax_main.transAxes)
    ax_yhist.barh(range(25), np.histogram(y_locations, bins=25)[0], color='skyblue', edgecolor="black")
    ax_yhist.set_xlabel("Frequency (Y-axis)")
    ax_yhist.spines['top'].set_visible(False)
    ax_yhist.spines['right'].set_visible(False)
    ax_yhist.get_yaxis().set_visible(False)  # Hide y-axis labels



    # Achsen und Layout
    ax_main.plot(contour[:, 0], contour[:, 1], color='black', linewidth=2)
    ax_main.set_xlabel("EDGE_X_LOCATION")
    ax_main.set_ylabel("EDGE_Y_LOCATION")
    ax_main.grid(False)  # Optional: Grid lines for better readability
    ax_main.set_aspect(4)
    ax_main.axis('off')  # Removes axles and frames

    # Titel
    track_count = len(tracks)
    displacement_text = f"{displacement_range[0]:.3f} - {displacement_range[1]:.3f} µm"
    plt.suptitle(f"{title}\nTracks: {track_count}, Track Displacement Range: {displacement_text}")

    plt.tight_layout()
    plt.show()


# Excel-Datei auswählen
def select_excel_file():
    root = Tk()
    root.withdraw()  # Hides the main window
    file_path = askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
    root.destroy()  # Closes the main window after selecting the file
    return file_path


# Hauptfunktion
def main():
    file_path = select_excel_file()
    if not file_path:
        print("Keine Datei ausgewählt.")
        return

    contour_svg_path = get_svg_path("Cell_contour.svg")
    contour = load_svg_contour(contour_svg_path)
    normalized_contour = normalize_contour(contour)
    adjusted_contour = adjust_contour_to_extremes(normalized_contour)

    xls = pd.ExcelFile(file_path)

    color_map = {'Static': 'blue', 'Mobile': 'green', 'Fast Mobile': 'red'}
    all_tracks = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)

        tracks = []
        track_colors = []

        # Dictionary to collect coordinates by Track_ID
        track_dict = defaultdict(list)

        for _, row in df.iterrows():
            track_id = row['Track_ID']
            x = row['EDGE_X_LOCATION']
            y = row['EDGE_Y_LOCATION']
            displacement = row['TRACK_DISPLACEMENT']

            # Save coordinates and displacement for the respective Track_ID
            track_dict[track_id].append((x, y, displacement))

        # Create all tracks after completing the iteration
        for track_id, points in track_dict.items():
            coordinates = [(x, y) for x, y, _ in points]
            displacements = [disp for _, _, disp in points]

            # Calculate track displacement for the entire Track_ID (e.g. as a maximum)
            track_displacement = max(displacements)

            # Save track data
            track = {
                'coordinates': coordinates,
                'displacement': track_displacement
            }
            tracks.append(track)
            track_colors.append(color_map.get(sheet_name, 'black'))  # Farbe zuweisen
            all_tracks.append({
                'coordinates': coordinates,
                'displacement': track_displacement,
                'sheet': sheet_name
            })

        show_arrows = sheet_name in ['Mobile', 'Fast Mobile']
        # Calculation of the displacement area
        global_displacement_min = min(track['displacement'] for track in tracks)
        global_displacement_max = max(track['displacement'] for track in tracks)
        displacement_range = (global_displacement_min, global_displacement_max)

        plot_tracks(tracks,adjusted_contour, title=f"Tracks from {sheet_name}", displacement_range=displacement_range)


    if all_tracks:
        combined_tracks = []

        # Collect tracks from all spreadsheets
        for track in all_tracks:
            combined_tracks.append(track)

        # Calculate displacement area
        displacement_min = min(track['displacement'] for track in combined_tracks)
        displacement_max = max(track['displacement'] for track in combined_tracks)
        displacement_range = (displacement_min, displacement_max)

        # Plotting the combined tracks
        plot_tracks(combined_tracks, adjusted_contour, title="Combined Tracks from All Sheets",
                    displacement_range=displacement_range)

    #


if __name__ == "__main__":
    main()
