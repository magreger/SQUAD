# SQUAD
(Signal QUantification And Display) is a Python toolkit for quantifying fluorescence signals and tracking data in filamentous bacterial cells. It summaries signal intensities and trajectories from multiple cells, calculates their cell-specific relative positions, and provides clear visualizations.

Description
The python scripts “Signal_quantification” and “Track_quantififcation” are used to quantify signal or track localizations by collecting their relative position in the respective cell and projecting and normalizing them to a standard cell using data from all selected bacterial cells. The script “Signal_visualization” is used to visualize the signal distribution via heatmap and HDBScan whereas “Track_visualization” display the localization of all tracks according to defined categories by “track_displacement”. The distribution of the curvature can be displayed using the script “Curvature_Visualization”. In addition, the script “Signal_pattern_profiling” provides a summarized overview of manually marked signal paths by normalizing their length and signal intensity and, if desired, sorting them according to the global intensity minimum or maximum.

Installation

All python scripts are design to run with python 3.12.8 which can be downloaded here: https://www.python.org/downloads/ and install it according to instructions. To execute a script run the following steps:
1.	Open CMD command terminal and navigate with “cd” to the python script- file-
2.	Install and create a virtual python environment with pip install virtualenv and virtualenv env. You only have to do this once
3.	Activate virtual environment with env\Scripts\activate
4.	Install all required libraries: pip install opencv-python numpy matplotlib geopandas pygeoops pandas shapely scipy scikit-image openpyxl pandas hdbscan re os
5.	Run the scripts. Example: python Signal_quantification.py

Requirements

Signal_quantification

- Cell contours are based on brightfield images with a clear highlighting of the cell. We recommend the creation brightfield images in a binary look in imageJ by transferring the cell body to be considered onto a white background. The pictures should be saved as png files with the naming structure: Image 1_bf_1
- Images with fluorescent signal should be processed so that only the actual signal and no background signal is visible. These images should also be saved as png files: Image 1_mNG_1 

Signal_Visualization

- This Script requires the excel file normalized_signals provided by Signal_quantification. For proper visualization via heatmap it is recommended to exclude signal outliers based on their Y coordinate in Excel
- The Cell_contour.svg file is required for displaying cell contour in figures and should be saved in the same files as the scripts.

Curvature_Visualization

- Requires three Excel files of curvature_frequencies representing individual biological replicates.
- It is recommended to exclude outliers by sorting all values of column in excel. The excel structure should not be altered.

Track_quantification

- Requires cell contours which are based on brightfield images with a clear highlighting of the cell. We recommend the creation brightfield images in a binary look in imageJ by transferring the cell body to be considered onto a white background. The pictures should be saved as png files with the naming structure: Image_1.
- Files Links in tracks statistics_1 and Track statistics_1 that contain tracking data come from ImageJ´s plugin TrackMate. All files must contain a column Image with the number of the respective file.

Track_Visualization

- Requires excel file tracks_categorized provided by Track_quantification and the Cell_contour.svg file for displaying cell contour which should be saved in the same file as the scripts.

Signal_pattern_profiling

- Requires csv file containing table of signal intensity values of all pixels. Table of respective images were generated in ImageJ (Image -> Transform -> Image to Results). It is possible to skip path selection process by loading already prepared and saved excel file.
