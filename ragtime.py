"""
Ragtime-v5.0: A tool for Size-Exclusion Chromatography Small-Angle X-ray Scattering (SEC-SAXS) data analysis.

This script provides a graphical user interface (GUI) built with PySide6 and Matplotlib
to perform Guinier analysis on a series of SAXS profiles, typically from a SEC-SAXS experiment.

---
SUMMARY OF FUNCTIONALITY:
---
1.  **Ragtime Frame Selection**: The user can now specify a start and end frame to limit the
    range of the Ragtime analysis, allowing for focused analysis on specific regions of the SEC profile.
    If left blank, the analysis runs on all frames.
2.  **MODIFICATION**: 4-panel plot layout updated to match the Sexier application for
    visual consistency. The new layout is:
    - Top-left: Form Factor
    - Top-right: Guinier Fit (top) & Residuals (bottom)
    - Bottom-left: Kratky Plot
    - Bottom-right: Volume of Correlation (VC) Plot

---
PREVIOUS FUNCTIONALITY :
---
-   **Automated Unit Detection**: Automatically detects if q-values are in nm⁻¹ and converts to Å⁻¹.
-   **Direct Frame Navigation**: An input box allows jumping directly to a specific frame for analysis.
-   **Streamlined Peak Processing**: A single button extracts frames and calculates the weighted average,
    creating an `average-XXX-YYY.dat` file.
-   **Data Loading**: Loads all '.dat' files from a directory in a background thread.
-   **Automated Guinier Analysis**: Identifies the best profile and finds the optimal Guinier region.
-   **Manual Guinier Analysis**: Allows users to manually select the q-range for the fit.
-   **Batch Processing (Ragtime Plot)**: Applies the fit range to all files to generate Rg/I(0) plots.
-   **Molecular Weight (MW) Estimation**: Estimates MW using the Volume of Correlation (Vc) method.
-   **Organized Output**: Saves all results in a dedicated "Ragtime-Results" folder.

---
ASSOCIATED MATHEMATICS:
---
-   **Guinier Approximation**: For small q values, the scattering intensity I(q) of a particle
    can be approximated by the Guinier equation:
        I(q) = I(0) * exp(-((q*Rg)^2)/3)

    By taking the natural logarithm, we get a linear equation:
        ln(I(q)) = ln(I(0)) - (Rg^2 / 3) * q^2

-   **Molecular Weight (MW) Estimation**: The molecular weight (MW) is estimated using the Porod invariant (Q_R)
    and the volume of correlation (Vc), where:
    -   Vc = I(0) / integral(q*I(q) dq) from 0 to infinity. (Here, we integrate to q=0.3 Å⁻¹)
    -   Q_R = Vc^2 / Rg
    -   MW ≈ Q_R / 0.1231 (for proteins, in Da, assuming q is in Å⁻¹)

"""
import matplotlib
# This line MUST be before the pyplot import to ensure the correct backend is loaded.
# 'QtAgg' tells Matplotlib to use the Qt backend, making it compatible with PySide6.
matplotlib.use('QtAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from scipy.stats import linregress
from scipy import integrate
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QLineEdit, QFileDialog, QMessageBox,
                             QGridLayout, QFrame)
# Qt.QtCore contains core non-GUI functionality, including threading and signals
from PySide6.QtCore import Qt, QThread, Signal, Slot

import os
import sys
import re  # Regular expressions for parsing filenames
import shutil  # For copying files (peak extraction)


###############################################
#
# Helper Functions & Worker Threads
#
###############################################

def _read_dat_file(file_path):
    """
    Reads a SAXS .dat file (typically 3-column: q, I, Err).
    Skips any lines starting with '#' or empty lines.
    Performs basic validation to ensure data is numeric and not NaN.
    """
    valid_data_rows = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Skip header or empty lines
                if not line.strip() or line.strip().startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        # Try to convert the first 3 columns to floats
                        q, i, err = map(float, parts[:3])
                        # Skip rows with NaN (Not a Number) values
                        if np.isnan(q) or np.isnan(i) or np.isnan(err):
                            continue
                        valid_data_rows.append([q, i, err])
                    except ValueError:
                        # Skip lines that cannot be converted to float
                        continue
    except Exception as e:
        print(f"❌ Error reading file {os.path.basename(file_path)}: {e}")
        return None
    if not valid_data_rows:
        print(f"⚠️ No valid data found in {os.path.basename(file_path)}")
        return None
    # Return the data as a NumPy array for efficient numerical operations
    return np.array(valid_data_rows, dtype=np.float64)


class FileLoadingThread(QThread):
    """
    A QThread worker for loading all .dat files from a directory.
    This runs in the background to prevent the GUI from freezing ("Not Responding")
    when loading hundreds or thousands of files.
    """
    # Signal(list) defines a signal that will emit a Python list when finished.
    loading_finished = Signal(list)

    def __init__(self, directory):
        super().__init__()
        self.directory = directory

    def run(self):
        """This is the entry point for the thread."""
        all_files_data = []
        try:
            # Get all files ending in .dat and sort them (usually by frame number)
            filenames = sorted([f for f in os.listdir(self.directory) if f.endswith(".dat")])
        except FileNotFoundError:
            print(f"❌ Directory not found: {self.directory}")
            self.loading_finished.emit([])  # Emit empty list on failure
            return

        for filename in filenames:
            file_path = os.path.join(self.directory, filename)
            data = _read_dat_file(file_path)
            if data is not None and data.size > 0:
                # Append a tuple: (filename, numpy_array_of_data)
                all_files_data.append((filename, data))

        # Emit the signal with the final list of all loaded data
        self.loading_finished.emit(all_files_data)


class PeakAveragingThread(QThread):
    """
    A QThread worker for averaging SAXS data files.
    This is also run in the background as interpolation and averaging can be slow.
    """
    # Signal(str) will emit a status message (string) upon completion.
    averaging_finished = Signal(str)

    def __init__(self, peak_directory, i0_values_map=None, start_frame=None, end_frame=None):
        super().__init__()
        self.peak_directory = peak_directory
        # i0_values_map contains I(0) for each file, used for normalization
        self.i0_values_map = i0_values_map if i0_values_map is not None else {}
        self.start_frame = start_frame
        self.end_frame = end_frame

    def run(self):
        all_q_values = []
        files_to_process = []

        filenames_in_peak_folder = sorted([f for f in os.listdir(self.peak_directory) if f.endswith(".dat")])
        if not filenames_in_peak_folder:
            self.averaging_finished.emit(f"No .dat files found in {self.peak_directory}")
            return

        # First pass: read all files and determine a common q-range
        for filename in filenames_in_peak_folder:
            file_path = os.path.join(self.peak_directory, filename)
            data = _read_dat_file(file_path)
            if data is not None:
                files_to_process.append((filename, data))
                all_q_values.extend(data[:, 0])

        if not files_to_process:
            self.averaging_finished.emit("No valid data found in selected peak files for averaging.")
            return

        # Create a common, high-resolution q-grid to interpolate all data onto
        q_min_common = np.min(all_q_values)
        q_max_common = np.max(all_q_values)
        common_q_grid = np.linspace(q_min_common, q_max_common, 1000)

        sum_I_over_err_sq = np.zeros_like(common_q_grid)
        sum_one_over_err_sq = np.zeros_like(common_q_grid)

        # Second pass: normalize, interpolate, and sum for weighted average
        for filename, data in files_to_process:
            q, I, Err = data[:, 0], data[:, 1], data[:, 2]

            # Get the I(0) for this file from the Ragtime analysis
            i0_for_file = self.i0_values_map.get(filename)

            # Normalize data by I(0) if available
            if i0_for_file is not None and i0_for_file > 0:
                I_normalized = I / i0_for_file
                Err_normalized = Err / i0_for_file
            else:
                # Fallback if I(0) wasn't calculated (e.g., bad fit)
                print(f"Warning: I(0) not found for {filename}. Using unnormalized data.")
                I_normalized = I
                Err_normalized = Err

            # Interpolate I and Err onto the common q-grid
            interp_I = np.interp(common_q_grid, q, I_normalized, left=np.nan, right=np.nan)
            interp_Err = np.interp(common_q_grid, q, Err_normalized, left=np.nan, right=np.nan)

            # --- Weighted Average Calculation ---
            # This loop performs the summation part of the weighted average formula:
            #   Avg_I = SUM( I_i / Err_i^2 ) / SUM( 1 / Err_i^2 )
            # We are calculating the numerator and denominator sums separately.
            for i, (iq, ierr) in enumerate(zip(interp_I, interp_Err)):
                if not np.isnan(iq) and not np.isnan(ierr) and ierr > 0:
                    weight = 1 / (ierr**2)
                    sum_I_over_err_sq[i] += iq * weight
                    sum_one_over_err_sq[i] += weight

        # Initialize arrays with NaN
        avg_I = np.full_like(common_q_grid, np.nan)
        err_on_mean = np.full_like(common_q_grid, np.nan)

        # Final division step of the weighted average
        valid_indices = sum_one_over_err_sq > 0
        avg_I[valid_indices] = sum_I_over_err_sq[valid_indices] / sum_one_over_err_sq[valid_indices]
        
        # --- Error Propagation ---
        # The error on the weighted mean is:
        #   Err_avg = 1 / sqrt( SUM( 1 / Err_i^2 ) )
        err_on_mean[valid_indices] = 1 / np.sqrt(sum_one_over_err_sq[valid_indices])

        # Filter out any remaining NaN values
        final_mask = ~np.isnan(avg_I)
        final_q = common_q_grid[final_mask]
        final_avg_I = avg_I[final_mask]
        final_err = err_on_mean[final_mask]

        if not final_q.size:
            self.averaging_finished.emit("Failed to average data. No common valid q-points found.")
            return

        # Save the final averaged data
        output_filename = os.path.join(self.peak_directory, f"average-{self.start_frame}-{self.end_frame}.dat")
        try:
            np.savetxt(output_filename, np.c_[final_q, final_avg_I, final_err],
                       fmt="%.6e", header="q\tI(q) (Normalized)\tError (Normalized)", comments="# ")
            self.averaging_finished.emit(f"Successfully averaged and saved normalized data to {output_filename}")
        except Exception as e:
            self.averaging_finished.emit(f"Error saving averaged file: {e}")


###############################################
#
# Main Application Class
#
###############################################

class SAXSAnalysisApp(QMainWindow):
    """
    The main application window. QMainWindow provides a standard window
    with menus, toolbars, etc. (though we only use a central widget).
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ragtime Sec-SAXS Analysis")
        self.setGeometry(100, 100, 1600, 800)  # x, y, width, height
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # --- Initialize class variables ---
        self.directory_path = None  # Path to the folder with .dat files
        self.raw_files_data = []    # List of (filename, data) tuples, as read
        self.files_data = []        # List of (filename, data) tuples, after q-conversion
        self.best_file = None       # Filename of the "best" profile
        self.best_file_index = None # Index of the "best" profile
        self.current_file_index = None # Index of the currently viewed profile
        self.data = None            # NumPy array of the currently viewed profile
        self.MW = None              # Molecular Weight of the current profile
        self.last_peak_folder = None # Path to the most recently created Peak-X folder
        self.file_i0_values = {}    # Dictionary to store {filename: I(0)} for averaging
        
        # Reference values from the "best" file's auto-analysis
        self.reference_rg = None
        self.reference_mw = None
        self.reference_i0 = None
        
        self.ax_integral = None # Placeholder for the VC plot's twin axis
        self.v_lines_2panel = [] # Vertical marker to visualise the current frame
        
        

        # This creates a 2x2 grid of axes (subplots)
        self.fig4panel, axes_flat = plt.subplots(2, 2, figsize=(8, 8))
        
        self.axA = axes_flat[0, 0] # Top-left: Form Factor
        self.axC = axes_flat[1, 0] # Bottom-left: Kratky
        self.axD = axes_flat[1, 1] # Bottom-right: VC
        
        # --- Create nested gridspec for Guinier + Residuals (top-right) ---
        # This is a more advanced Matplotlib trick.
        # 1. Get the gridspec (layout manager) of the main 2x2 grid.
        gs = axes_flat[0, 1].get_gridspec()
        # 2. Remove the original top-right axes.
        axes_flat[0, 1].remove()
        # 3. Create a *new* 2x1 sub-grid *inside* the top-right cell.
        gs_nested = gs[0, 1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        # 4. Add the two new subplots to this nested grid.
        self.axB = self.fig4panel.add_subplot(gs_nested[0]) # Top: Guinier
        self.axB_res = self.fig4panel.add_subplot(gs_nested[1], sharex=self.axB) # Bottom: Residual
        
        # Store all axes for easy clearing later
        self.all_4panel_axes = [self.axA, self.axB, self.axB_res, self.axC, self.axD]
        
        # FigureCanvasQTAgg is the Qt widget that displays the Matplotlib figure
        self.canvas4panel = FigureCanvasQTAgg(self.fig4panel)
        # --- END MODIFICATION ---

        # The 2-panel plot for Ragtime results (Rg/I0 vs frame)
        self.fig2panel = Figure(figsize=(8, 8), constrained_layout=True)
        self.canvas2panel = FigureCanvasQTAgg(self.fig2panel)

        # Create and arrange all the GUI widgets
        self.setup_ui()

    @property
    def output_path(self):
        """
        A @property makes this method behave like an attribute (e.g., self.output_path).
        It defines the path for the 'Ragtime-Results' folder, creating it if needed.
        The folder is located one level *above* the selected data directory.
        """
        if not self.directory_path:
            return None
        # os.path.dirname(self.directory_path) gets the parent folder
        path = os.path.join(os.path.dirname(self.directory_path), "Ragtime-Results")
        os.makedirs(path, exist_ok=True) # Doesn't error if the folder already exists
        return path

    def setup_ui(self):
        """
        Creates and arranges all the widgets in the main window.
        """
        # The main layout is horizontal (HBox): [4-panel plot] [Controls] [2-panel plot]
        main_layout = QHBoxLayout(self.central_widget)

        # --- Center Control Panel ---
        # QFrame adds a visual border
        control_widget = QFrame()
        # QVBoxLayout arranges widgets vertically
        control_layout = QVBoxLayout(control_widget)
        control_widget.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        control_widget.setMaximumWidth(260) # Fix the width of the control panel

        # --- Widgets for file browsing ---
        self.browse_button = QPushButton("Select Data Folder")
        self.browse_button.clicked.connect(self.browse_directory)
        self.folder_label = QLabel("Selected Folder: None")
        self.folder_label.setWordWrap(True) # Allow text to wrap to new lines

        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)

        control_layout.addWidget(self.browse_button)
        control_layout.addWidget(self.folder_label)
        control_layout.addWidget(self.file_label)
        
        # --- Widgets for file navigation ---
        nav_layout = QHBoxLayout() # Horizontal layout for buttons
        self.prev_file_button = QPushButton("<-")
        self.best_file_button = QPushButton("Best")
        self.next_file_button = QPushButton("->")
        
        self.frame_num_entry = QLineEdit() # Text input box
        self.frame_num_entry.setPlaceholderText("Frame #")
        self.frame_num_entry.setMaximumWidth(60)
        self.goto_frame_button = QPushButton("Go")
        
        # .clicked.connect(...) links a button click to a class method
        self.prev_file_button.clicked.connect(self.show_prev_file)
        self.best_file_button.clicked.connect(self.show_best_file)
        self.next_file_button.clicked.connect(self.show_next_file)
        self.goto_frame_button.clicked.connect(self.show_specific_frame)
        
        nav_layout.addWidget(self.prev_file_button)
        nav_layout.addWidget(self.best_file_button)
        nav_layout.addWidget(self.next_file_button)
        nav_layout.addStretch() # Pushes subsequent widgets to the right
        nav_layout.addWidget(self.frame_num_entry)
        nav_layout.addWidget(self.goto_frame_button)
        
        control_layout.addLayout(nav_layout)
        control_layout.addSpacing(10) # Add 10px vertical space

        # --- Widgets for Guinier analysis inputs ---
        # QGridLayout arranges widgets in a grid (row, col)
        input_grid = QGridLayout()
        self.qmin_label = QLabel("qmin index:")
        self.qmin_entry = QLineEdit()
        self.qmax_label = QLabel("qmax index:")
        self.qmax_entry = QLineEdit()

        input_grid.addWidget(self.qmin_label, 0, 0)
        input_grid.addWidget(self.qmin_entry, 0, 1)
        input_grid.addWidget(self.qmax_label, 1, 0)
        input_grid.addWidget(self.qmax_entry, 1, 1)
        control_layout.addLayout(input_grid)

        self.manual_guinier_button = QPushButton("Update Ragtime")
        self.manual_guinier_button.clicked.connect(self.manual_guinier_analysis)
        control_layout.addWidget(self.manual_guinier_button)
        control_layout.addSpacing(10)
        
        # --- Widgets for Ragtime analysis range ---
        control_layout.addWidget(QLabel("Ragtime Analysis Range (optional)"))
        ragtime_range_layout = QGridLayout()
        self.ragtime_start_frame_entry = QLineEdit()
        self.ragtime_end_frame_entry = QLineEdit()
        ragtime_range_layout.addWidget(QLabel("Start frame:"), 0, 0)
        ragtime_range_layout.addWidget(self.ragtime_start_frame_entry, 0, 1)
        ragtime_range_layout.addWidget(QLabel("End frame:"), 1, 0)
        ragtime_range_layout.addWidget(self.ragtime_end_frame_entry, 1, 1)
        control_layout.addLayout(ragtime_range_layout)

        self.apply_to_all_button = QPushButton("Run Ragtime")
        self.apply_to_all_button.clicked.connect(self.apply_to_all)
        control_layout.addWidget(self.apply_to_all_button)
        control_layout.addSpacing(10)

        # --- Labels to display results ---
        self.rg_label = QLabel("Rg: ")
        self.i0_label = QLabel("I(0): ")
        self.qmin_rg_label = QLabel("qmin*Rg: ")
        self.qmax_rg_label = QLabel("qmax*Rg: ")
        self.mw_label = QLabel("MW: ")

        control_layout.addWidget(self.rg_label)
        control_layout.addWidget(self.i0_label)
        control_layout.addWidget(self.qmin_rg_label)
        control_layout.addWidget(self.qmax_rg_label)
        control_layout.addWidget(self.mw_label)
        control_layout.addSpacing(10)

        # --- Widgets for peak extraction ---
        control_layout.addWidget(QLabel("Peak Processing"))
        layout_peak = QGridLayout()
        self.first_frame_entry = QLineEdit()
        self.last_frame_entry = QLineEdit()
        layout_peak.addWidget(QLabel("First frame:"),0,0)
        layout_peak.addWidget(self.first_frame_entry,0,1)
        layout_peak.addWidget(QLabel("Last frame:"),1,0)
        layout_peak.addWidget(self.last_frame_entry,1,1)
        control_layout.addLayout(layout_peak)

        self.extract_and_average_button = QPushButton("Extract & Average Peak")
        self.extract_and_average_button.clicked.connect(self.extract_and_average_peak)
        control_layout.addWidget(self.extract_and_average_button)

        control_layout.addStretch() # Pushes everything below it to the bottom

        # --- Reset and Quit buttons ---
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset)
        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.quit)

        control_layout.addWidget(self.reset_button)
        control_layout.addWidget(self.quit_button)

        # --- Version Label ---
        self.version_label = QLabel("JMB-Scripts Ragtime-Sec-SAXS-v5.0")
        self.version_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.version_label)

        # --- Label for cursor coordinates ---
        self.coord_label = QLabel("Cursor position: (x=N/A, y=N/A)")
        self.coord_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        control_layout.addWidget(self.coord_label)
        
        # --- Assemble the 3-panel layout ---
        # addWidget(widget, stretch_factor)
        # The 'stretch' factor controls how much space each widget takes up.
        # [Plot 1 (3)] [Controls (1)] [Plot 2 (3)]
        main_layout.addWidget(self.canvas4panel, stretch=3)
        main_layout.addWidget(control_widget, stretch=1)
        main_layout.addWidget(self.canvas2panel, stretch=3)

        # --- Connect mouse motion events ---
        # This links mouse movement over the canvas to our _on_motion_update_coords method
        self.canvas4panel.mpl_connect('motion_notify_event', self._on_motion_update_coords)
        self.canvas2panel.mpl_connect('motion_notify_event', self._on_motion_update_coords)

    #########################################################################
    #
    # Core Application Logic and Slots
    #
    #########################################################################

    def clear_4_panel_plot(self):
        """Helper to clear all axes on the 4-panel plot."""
        if hasattr(self, 'all_4panel_axes'):
            for ax in self.all_4panel_axes:
                ax.clear()
        # The twin axis (ax_integral) must be removed separately
        if hasattr(self, 'ax_integral') and self.ax_integral:
            self.ax_integral.remove()
            self.ax_integral = None
        self.canvas4panel.draw() # Redraw the empty canvas

    def browse_directory(self):
        """
        Opens a dialog for the user to select a directory and starts the file loading thread.
        """
        folder_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not folder_path: return # User cancelled

        self.reset() # Clear previous data
        self.directory_path = folder_path
        
        self.folder_label.setText(f"Folder: {os.path.basename(folder_path)}")
        self.file_label.setText("Loading files...")

        # Disable buttons while loading to prevent errors
        self.browse_button.setEnabled(False)
        self.manual_guinier_button.setEnabled(False)
        self.apply_to_all_button.setEnabled(False)
        self.extract_and_average_button.setEnabled(False)
        self.prev_file_button.setEnabled(False)
        self.best_file_button.setEnabled(False)
        self.next_file_button.setEnabled(False)
        self.goto_frame_button.setEnabled(False)

        # Create and start the file loading thread
        self.worker = FileLoadingThread(folder_path)
        self.worker.loading_finished.connect(self.finish_file_loading)
        self.worker.start()

    @Slot(list) # This decorator identifies this method as a "slot"
    def finish_file_loading(self, raw_files_data):
        """
        Slot called when FileLoadingThread emits its 'loading_finished' signal.
        The 'raw_files_data' argument is the list emitted by the thread.
        """
        self.browse_button.setEnabled(True) # Re-enable browse button
        
        if not raw_files_data:
            self.file_label.setText("No valid files found.")
            QMessageBox.warning(self, "No Data", "No valid .dat files could be read.")
            return
        
        self.raw_files_data = raw_files_data
        print(f"✅ Successfully loaded {len(self.raw_files_data)} raw files.")
        
        # --- Automatic q-unit detection and conversion ---
        # Look at the q-values of the first file
        first_file_q = self.raw_files_data[0][1][:, 0]
        # Heuristic: If the max q value is > 2.0, it's likely in nm⁻¹ (since 0.5-1.0 Å⁻¹ is a common max)
        needs_conversion = np.max(first_file_q) > 2.0
        
        if needs_conversion:
             QMessageBox.information(self, "Unit Detection", "q-values appear to be in nm⁻¹. They will be converted to Å⁻¹.")
             print("ℹ️ Detected q-units as nm⁻¹. Converting to Å⁻¹.")
        
        self.files_data = [] # This will hold the processed data
        for filename, raw_data in self.raw_files_data:
            processed_data = raw_data.copy()
            if needs_conversion:
                # 1 nm⁻¹ = 0.1 Å⁻¹
                processed_data[:, 0] *= 0.1 # Convert q-column from nm⁻¹ to Å⁻¹
            self.files_data.append((filename, processed_data))

        print("Data processed. Running analysis...")
        
        # Enable all analysis buttons now that data is loaded
        self.manual_guinier_button.setEnabled(True)
        self.apply_to_all_button.setEnabled(True)
        self.extract_and_average_button.setEnabled(True)
        self.prev_file_button.setEnabled(True)
        self.best_file_button.setEnabled(True)
        self.next_file_button.setEnabled(True)
        self.goto_frame_button.setEnabled(True)
        
        # --- Start the automatic analysis pipeline ---
        self.find_best_file()

        if self.best_file:
            self.auto_guinier_analysis() # 1. Find best Guinier range
            self.apply_to_all()          # 2. Apply it to all files
            
    def find_best_file(self):
        """
        Identifies the best scattering profile from the loaded data
        by finding the file with the highest total integral (v5.0 method).
        """
        best_integral = -np.inf  # Changed from best_intensity
        best_file_info = None
        best_idx = -1

        for i, (filename, data) in enumerate(self.files_data):
            q, I = data[:, 0], data[:, 1]
            
            try:
                # Use Simpson's rule for integration, just like in SubMe
                file_integral = integrate.simpson(I, x=q)
            except Exception as e:
                # This might fail if q is not sorted, but it should be.
                print(f"Warning: Could not calculate integral for {filename}: {e}")
                continue


            if file_integral > best_integral: # Compare integral instead of intensity
                best_integral = file_integral
                best_file_info = (filename, data)
                best_idx = i

        if best_file_info:
            self.best_file, self.data = best_file_info
            self.best_file_index = best_idx
            self.current_file_index = best_idx
            # Updated label to reflect the new method
            self.file_label.setText(f"Best file (max integral): {self.best_file}")
            print(f"🏆 Best file selected (max integral): {self.best_file} at index {self.best_file_index}")
            self.prev_file_button.setEnabled(self.current_file_index > 0)
            self.next_file_button.setEnabled(self.current_file_index < len(self.files_data) - 1)
        else:
            self.file_label.setText("Could not find best file.")

    def auto_guinier_analysis(self):
        """
        Performs an automated Guinier analysis on the 'best_file'.
        It searches for the optimal q-range by testing many different
        start points and window sizes.
        """
        if self.data is None: return
        q, I, E = self.data[:, 0], self.data[:, 1], self.data[:, 2] 
        
        # Guinier fit requires I > 0 for ln(I)
        positive_mask = I > 0
        if not np.any(positive_mask): return
        q_safe, I_safe = q[positive_mask], I[positive_mask]
        
        # --- MATH: Guinier Linearization ---
        # I(q) = I(0) * exp(-((q*Rg)^2)/3)
        # ln(I(q)) = ln(I(0)) - (Rg^2 / 3) * q^2
        # This is a linear equation y = c + m*x
        # y = ln(I(q))
        # c = ln(I(0))  (intercept)
        # m = -Rg^2 / 3  (slope)
        # x = q^2
        q_squared_safe, ln_I_safe = q_safe**2, np.log(I_safe)

        best_score, best_qmin_idx, best_qmax_idx, best_rg, best_i0 = -np.inf, 0, 0, 0, 0

        # --- Search parameters ---
        max_start_idx = min(40, len(q_squared_safe) - 2) # Only start fit in the first 40 points
        min_window_size = 5  # Need at least 5 points for a reliable fit
        max_window_size = min(160, len(q_squared_safe))

        # Brute-force search loop
        for window_size in range(min_window_size, max_window_size + 1):
            for start_idx in range(0, max_start_idx + 1):
                end_idx = start_idx + window_size
                if end_idx > len(q_squared_safe): continue

                q_sq_fit, ln_I_fit = q_squared_safe[start_idx:end_idx], ln_I_safe[start_idx:end_idx]
                if len(q_sq_fit) < 2: continue

                # Perform the linear regression
                slope, intercept, r_value, _, _ = linregress(q_sq_fit, ln_I_fit)

                # --- Fit Validation ---
                if slope >= 0: continue # Slope must be negative
                
                # --- MATH: Calculate Rg and I(0) from slope and intercept ---
                # m = -Rg^2 / 3  => Rg^2 = -3*m => Rg = sqrt(-3 * slope)
                # c = ln(I(0))   => I(0) = exp(intercept)
                rg, i0 = np.sqrt(-3 * slope), np.exp(intercept)
                
                if not (0 < rg < 1000 and i0 > 0): continue # Sanity check

                # --- MATH: Check Guinier Condition (q_max * Rg) ---
                # The Guinier approximation is only valid for q*Rg < ~1.3-1.5
                qmin_rg, qmax_rg = q_safe[start_idx] * rg, q_safe[end_idx - 1] * rg
                if not (0.8 <= qmax_rg <= 1.5 and qmin_rg <= 0.8): continue

                # --- Scoring Function ---
                # This score prioritizes high linearity (r_value^8)
                # and a qmax*Rg value close to the ideal 1.3.
                score = (r_value**8) + (1 - abs(qmax_rg - 1.3))
                if score > best_score:
                    best_score, best_qmin_idx, best_qmax_idx, best_rg, best_i0 = score, start_idx, end_idx - 1, rg, i0

        if best_score > -np.inf:
            # Convert indices from the 'safe' (I>0) array back to the original data array
            original_q_min_idx = np.where(q == q_safe[best_qmin_idx])[0][0]
            original_q_max_idx = np.where(q == q_safe[best_qmax_idx])[0][0]

            # Set the GUI text boxes to the best-fit range
            self.qmin_entry.setText(str(original_q_min_idx))
            self.qmax_entry.setText(str(original_q_max_idx))

            # Calculate MW for this best fit
            self.MW = self.estimate_molecular_weight(q, I, best_i0, best_rg)
            
            # Store these values as the reference for the 2-panel plot
            self.reference_rg = best_rg
            self.reference_mw = self.MW
            self.reference_i0 = best_i0
            
            # Generate the plots and update labels
            self.generate_4_panel_plot(q, I, E, original_q_min_idx, original_q_max_idx, best_rg, best_i0, self.MW)
            self.update_results(best_rg, best_i0, self.MW, q[original_q_min_idx] * best_rg, q[original_q_max_idx] * best_rg)
            self._update_2panel_marker()
        else:
            QMessageBox.warning(self, "Estimation Failed", "Could not determine a valid Guinier region.")

    def manual_guinier_analysis(self):
        """
        Performs Guinier analysis on the 'best file' using the user-specified q-range.
        This updates the 'reference' values and then re-runs the Ragtime analysis.
        """
        if self.best_file_index is None or not self.files_data:
            QMessageBox.critical(self, "Error", "No 'best file' has been determined. Load data first.")
            return
        
        try:
            # Get q-range indices from the GUI text boxes
            qmin_idx, qmax_idx = int(self.qmin_entry.text()), int(self.qmax_entry.text())
            
            # Get data for the "best" file
            _ , best_file_data = self.files_data[self.best_file_index]
            q, I, E = best_file_data[:, 0], best_file_data[:, 1], best_file_data[:, 2]

            if not (0 <= qmin_idx < qmax_idx < len(q)):
                QMessageBox.critical(self, "Error", "qmin/qmax range is out of bounds for the best file.")
                return

            # --- Perform Guinier fit on the specified range ---
            I_slice = I[qmin_idx:qmax_idx+1]
            valid_mask = I_slice > 0
            if np.count_nonzero(valid_mask) < 2:
                QMessageBox.warning(self, "Analysis Error", "Not enough valid data points (I > 0) in selected range for the best file.")
                return

            # --- MATH: Same linearization as auto_guinier_analysis ---
            q_squared_fit = q[qmin_idx:qmax_idx+1][valid_mask]**2
            ln_I_fit = np.log(I_slice[valid_mask])
            slope, intercept, _, _, _ = linregress(q_squared_fit, ln_I_fit)

            if slope >= 0:
                QMessageBox.warning(self, "Analysis Failed", "Guinier fit failed: slope is not negative.")
                return

            # --- MATH: Calculate Rg and I(0) ---
            best_i0 = np.exp(intercept)
            best_rg = np.sqrt(-3 * slope)

            self.MW = self.estimate_molecular_weight(q, I, best_i0, best_rg)
            
            # --- Update Reference Values ---
            # The manual fit on the best file now becomes the new reference
            self.reference_rg = best_rg
            self.reference_mw = self.MW
            self.reference_i0 = best_i0
            
            # Reset view to the best file
            self.current_file_index = self.best_file_index
            self.file_label.setText(f"Best file: {self.best_file}")
            self.prev_file_button.setEnabled(self.current_file_index > 0)
            self.next_file_button.setEnabled(self.current_file_index < len(self.files_data) - 1)
            
            # Update plots and labels
            self.generate_4_panel_plot(q, I, E, qmin_idx, qmax_idx, best_rg, best_i0, self.MW)
            self.update_results(best_rg, best_i0, self.MW, q[qmin_idx] * best_rg, q[qmax_idx] * best_rg)
            
            # --- Re-run Ragtime ---
            # This applies the *new* manual q-range to all files
            self.apply_to_all(q_range_manual=(qmin_idx, qmax_idx))
            self._update_2panel_marker()
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid qmin/qmax indices. Please enter integers.")
            
    def show_prev_file(self):
        """Navigates to the previous file in the list."""
        if self.current_file_index is not None and self.current_file_index > 0:
            # Calls the helper function to do the analysis and plotting
            self._display_and_analyze_file_at_index(self.current_file_index - 1)

    def show_next_file(self):
        """Navigates to the next file in the list."""
        if self.current_file_index is not None and self.current_file_index < len(self.files_data) - 1:
            self._display_and_analyze_file_at_index(self.current_file_index + 1)

    def show_best_file(self):
        """Resets the view and analysis to the original best file."""
        if self.best_file_index is not None:
            # Re-run auto_guinier_analysis to restore the original auto-fit q-range
            _ , self.data = self.files_data[self.best_file_index]
            self.auto_guinier_analysis()

    def show_specific_frame(self):
        """Displays and analyzes a specific frame number from the input box."""
        if not self.files_data:
            QMessageBox.warning(self, "No Data", "No files are loaded.")
            return
        try:
            # User enters 1-based frame number, list is 0-indexed
            frame_num = int(self.frame_num_entry.text())
            index = frame_num - 1 # Convert to 0-based index
            
            if not (0 <= index < len(self.files_data)):
                QMessageBox.warning(self, "Invalid Frame", f"Please enter a frame number between 1 and {len(self.files_data)}.")
                return
            
            self._display_and_analyze_file_at_index(index)

        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid frame number. Please enter an integer.")

    def _display_and_analyze_file_at_index(self, index):
        """
        Helper function: Analyzes a file at a specific index using the
        q-range *currently in the GUI* and updates the 4-panel plot.
        """
        if not self.files_data or not (0 <= index < len(self.files_data)):
            return

        self.current_file_index = index
        current_filename, current_data = self.files_data[index]
        self.file_label.setText(f"Displaying: {current_filename}")

        try:
            # Get the q-range from the GUI
            qmin_idx = int(self.qmin_entry.text())
            qmax_idx = int(self.qmax_entry.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid qmin/qmax indices. Cannot navigate files.")
            return

        q, I, E = current_data[:, 0], current_data[:, 1], current_data[:, 2]

        # Check if the q-range is valid *for this specific file*
        if not (0 <= qmin_idx < qmax_idx < len(q)):
            self.clear_4_panel_plot() 
            self.mw_label.setText(f"MW: q-range out of bounds for {current_filename}")
            return

        # --- Perform Guinier fit (same logic as manual_guinier_analysis) ---
        I_slice = I[qmin_idx:qmax_idx+1]
        valid_mask = I_slice > 0
        if np.count_nonzero(valid_mask) < 2:
            self.clear_4_panel_plot() 
            self.update_results(np.nan, np.nan, np.nan, np.nan, np.nan)
            self.mw_label.setText(f"MW: Not enough valid points for {current_filename}")
            return

        q_squared_fit = q[qmin_idx:qmax_idx+1][valid_mask]**2
        ln_I_fit = np.log(I_slice[valid_mask])
        slope, intercept, _, _, _ = linregress(q_squared_fit, ln_I_fit)

        if slope >= 0:
            self.clear_4_panel_plot() 
            self.update_results(np.nan, np.nan, np.nan, np.nan, np.nan)
            self.mw_label.setText(f"MW: Guinier fit failed for {current_filename}")
            return

        # Calculate results for *this* frame
        i0 = np.exp(intercept)
        rg = np.sqrt(-3 * slope)
        self.MW = self.estimate_molecular_weight(q, I, i0, rg)

        # Update the 4-panel plot and labels
        self.generate_4_panel_plot(q, I, E, qmin_idx, qmax_idx, rg, i0, self.MW)
        self.update_results(rg, i0, self.MW, q[qmin_idx] * rg, q[qmax_idx] * rg)

        # Update navigation button states
        self.prev_file_button.setEnabled(self.current_file_index > 0)
        self.next_file_button.setEnabled(self.current_file_index < len(self.files_data) - 1)
        self._update_2panel_marker()

    def _update_2panel_marker(self):
        """
        Updates the vertical line on the 2-panel (Ragtime) plot
        to show the currently selected frame.
        """
        if self.v_lines_2panel:
            for line in self.v_lines_2panel:
                try:
                    line.remove()
                except (ValueError, AttributeError):
                    pass # Line was already gone
        self.v_lines_2panel = [] # Reset the list

        if self.current_file_index is None:
            return 

        current_frame_num = self.current_file_index + 1
        
        try:
            # --- MODIFICATION: Corrected axis indices ---
            ax1 = self.fig2panel.axes[0]  # Top main axis
            ax2 = self.fig2panel.axes[1]  # Bottom main axis
            # --- END MODIFICATION ---
        except IndexError:
            return # Axes don't exist yet

        # Get the x-limits from one of the axes
        xmin, xmax = ax2.get_xlim()
        
        if not (xmin <= current_frame_num <= xmax):
            self.canvas2panel.draw()
            return

        # --- MODIFICATION: Added zorder=10 to draw on top ---
        line1 = ax1.axvline(current_frame_num, color="#D8B0D4D2", linestyle=':', linewidth=2, zorder=10) 
        line2 = ax2.axvline(current_frame_num, color="#D8B0D4D2", linestyle=':', linewidth=2, zorder=10)
        # --- END MODIFICATION ---
        
        self.v_lines_2panel.append(line1)
        self.v_lines_2panel.append(line2)
        
        self.canvas2panel.draw()

    def estimate_molecular_weight(self, q, I, best_i0, best_rg):
        """
        Estimates the molecular weight (MW) in Daltons (Da) using the
        Volume of Correlation (Vc) / Porod Invariant (Q_R) method.
        """
        # Ensure data is sorted by q for integration
        sort_indices = np.argsort(q)
        q_sorted, I_sorted = q[sort_indices], I[sort_indices]
        
        # --- MATH: Limit integration range ---
        # The integral is technically to infinity, but for MW estimation
        # using this method, a cutoff of 0.3 Å⁻¹ is common.
        q_filtered = q_sorted[q_sorted <= 0.3]
        I_filtered = I_sorted[q_sorted <= 0.3]

        if len(q_filtered) < 2: return 0 # Not enough points to integrate

        # --- MATH: Calculate the integral ---
        # Use Simpson's rule for numerical integration of integral(q*I(q) dq)
        Intgr = integrate.simpson(I_filtered * q_filtered, x=q_filtered)

        if Intgr <= 0 or best_rg <= 0 or best_i0 <= 0: return 0 # Invalid input

        # --- MATH: Volume of Correlation (Vc) ---
        # Vc = I(0) / integral(q*I(q) dq)
        VC = best_i0 / Intgr
        
        # --- MATH: Porod Invariant (Q_R) ---
        # Q_R = Vc^2 / Rg
        QR = VC**2 / best_rg
        
        # --- MATH: Molecular Weight (MW) ---
        # MW (Da) ≈ Q_R / 0.1231
        # The 0.1231 is an empirical constant for proteins, assuming
        # a typical protein density and contrast, with q in Å⁻¹.
        return QR / 0.1231

    def generate_4_panel_plot(self, q, I, E, qmin_idx, qmax_idx, rg, i0, MW):
        """
        Generates and displays the 4-panel diagnostic plot,
        styled to match the Sexier application.
        """
        # --- Clear all axes ---
        for ax in self.all_4panel_axes: 
            ax.clear()
        if hasattr(self, 'ax_integral') and self.ax_integral: 
            self.ax_integral.remove()
            self.ax_integral = None
        
        axA, axB, axB_res, axC, axD = self.axA, self.axB, self.axB_res, self.axC, self.axD
        c1, c2, c3, c4, c5 = '#0D92F4','#77CDFF','#F95454','#C62E2E','#F3F3E0'

        # --- Plot 1: Form Factor (Top-Left) ---
        # Standard log(I) vs. q plot
        axA.plot(q, I, c=c1)
        axA.set_yscale('log')
        axA.set_xscale('linear')
        axA.grid(True)
        plot_title = os.path.basename(self.directory_path) if self.directory_path else "SAXS"
        axA.set_title(f'{plot_title} Form Factor', fontsize='small')
        axA.set_xlabel('q (Å⁻¹)')
        axA.set_ylabel('log(I(q))')
        axA.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        # --- Data prep for Guinier & Residuals ---
        q_range_full, I_range_full, E_range_full = q[qmin_idx:qmax_idx+1], I[qmin_idx:qmax_idx+1], E[qmin_idx:qmax_idx+1]
        valid_mask = I_range_full > 0
        q_range_safe, I_range_safe, E_range_safe = q_range_full[valid_mask], I_range_full[valid_mask], E_range_full[valid_mask]
        
        residuals = np.array([])
        slope, intercept = 0, 0
        if q_range_safe.size > 1:
            with np.errstate(invalid='ignore'): # Ignore log(0) warnings if any
                # Re-calculate slope/intercept for plotting the fit line
                slope, intercept = linregress(q_range_safe**2, np.log(I_range_safe))[:2]
                
                # --- MATH: Normalized Residuals ---
                # Residual = (Data - Fit) / Error
                # Data = ln(I)
                # Fit = slope*q^2 + intercept
                # Error on ln(I) ≈ Error_I / I
                # We use Error_I / sqrt(I) for Poisson statistics scaling
                residuals = (np.log(I_range_safe) - (slope*q_range_safe**2 + intercept)) / (E_range_safe / np.sqrt(I_range_safe))

        # --- Plot 2: Guinier Plot (Top-Right-Top) ---
        # This is the ln(I) vs q^2 plot
        valid_plot_mask = I_range_full > 0
        with np.errstate(invalid='ignore'):
            axB.errorbar(q_range_full[valid_plot_mask]**2, np.log(I_range_full[valid_plot_mask]), 
                         yerr=E_range_full[valid_plot_mask]/I_range_full[valid_plot_mask], # Error on ln(I) ≈ Err(I)/I
                         fmt='o', ms=4, c=c1, label='exp.')
        if q_range_safe.size > 1:
            axB.plot(q_range_safe**2, slope*q_range_safe**2 + intercept, c=c4, label='Fit')
        
        axB.set(ylabel='ln(I(q))')
        axB.grid(False) 
        axB.set_title('Guinier Plot', fontsize='small')
        axB.legend(loc='upper center')
        
        qmin_Rg, qmax_Rg = q[qmin_idx] * rg, q[qmax_idx] * rg
        
        # Add text box with fit results
        axB.text(0.05, 0.05, f'Rg = {rg:.2f} Å\nI0 = {i0:.2e}\nq·Rg: {qmin_Rg:.2f}–{qmax_Rg:.2f}', 
                 transform=axB.transAxes, ha='left', va='bottom', 
                 bbox=dict(boxstyle='round', fc=c5, alpha=0.7))
        
        plt.setp(axB.get_xticklabels(), visible=False) # Hide x-axis labels
        axB.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        # --- Plot 3: Residuals Plot (Top-Right-Bottom) ---
        # Shows (data-fit)/error. A good fit has residuals randomly scattered around 0.
        if residuals.size > 0:
            axB_res.plot(q_range_safe**2, residuals, 'o', ms=2, c=c1)
        axB_res.axhline(0, c=c4, lw=1) # Add a line at y=0
        axB_res.set_xlabel('q² (Å⁻²)', fontsize='small') 
        axB_res.set_ylabel('Res/σ', fontsize='small')
        axB_res.tick_params(axis='both', which='major', labelsize='x-small')
        axB_res.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        axB_res.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        axB.xaxis.get_offset_text().set_visible(False)
        if axB_res.xaxis.get_offset_text():
            axB_res.xaxis.get_offset_text().set_fontsize('x-small')

        # --- Plot 4: Kratky Plot (Bottom-Left) ---
        # --- MATH: Normalized Kratky ---
        # y-axis = (q*Rg)^2 * (I(q) / I(0))
        # x-axis = q*Rg
        kratky_y = (q * rg)**2 * I / i0 if i0 > 0 else np.zeros_like(q)
        axC.plot(q*rg, kratky_y, c=c3)
        axC.grid(True)
        axC.set(xlabel='q·Rg', ylabel='(q·Rg)²·I(q)/I(0)')
        axC.set_title('Kratky', fontsize='small') 
        
        # For a globular protein, this plot should peak at x=sqrt(3) (≈1.73)
        # with a peak height of 3/e (≈1.1)
        y_max_kratky = np.max(kratky_y[np.isfinite(kratky_y)]) if np.any(np.isfinite(kratky_y)) else 2.0
        axC.set_ylim(0, max(2.0, min(y_max_kratky * 1.1, 5.0))) # Auto-zoom
        axC.axvline(np.sqrt(3), c='grey', ls='--') # Mark x=sqrt(3)
        axC.axhline(1.1, c='grey', ls='--')        # Mark y=1.1
        axC.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

        # --- Plot 5: VC Plot (Bottom-Right) ---
        # This plot visualizes the components of the Vc calculation
        self.ax_integral = axD.twinx() # Create a second y-axis
        
        # --- MATH: Vc components ---
        # Plot 1 (blue): q*I(q)
        # Plot 2 (red): cumulative integral(q*I(q) dq)
        line1, = axD.plot(q, I*q, c=c1, label='q·I(q)')
        
        # Calculate the cumulative integral using Simpson's rule at each q-point
        cumulative_integral = [integrate.simpson(I[:i]*q[:i], x=q[:i]) if i > 1 else 0 for i in range(1, len(q)+1)]
        line2, = self.ax_integral.plot(q, cumulative_integral, c=c3, label='Integral')
        
        axD.set(xlabel='q (Å⁻¹)', ylabel='q·I(q)')
        axD.grid(True)
        axD.set_title('Volume of Corr.', fontsize='small') 
        axD.tick_params(axis='y', labelcolor=c1)
        self.ax_integral.tick_params(axis='y', labelcolor=c3)
        
        # Mark the q=0.3 cutoff used for the MW calculation
        axD.axvline(0.3, c=c1, ls='--', alpha=0.8)
        
        self.ax_integral.legend(handles=[line1, line2], loc='upper left')
        
        # Add text box with the final MW result
        axD.text(0.05, 0.05, f'MW(q<0.3): {MW:,.0f} Da', 
                 transform=axD.transAxes, ha='left', va='bottom', 
                 bbox=dict(boxstyle='round', fc=c5, alpha=0.7), zorder=10)
        
        axD.set_ylim(bottom=0)
        self.ax_integral.set_ylim(bottom=0)
        axD.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        self.ax_integral.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        # --- Finalize plot ---
        self.fig4panel.tight_layout() # Adjusts plot to prevent labels overlapping
        self.canvas4panel.draw()      # Redraw the canvas
        
        # Save a high-resolution copy of the plot
        if self.output_path:
            self.fig4panel.savefig(os.path.join(self.output_path, "Guinier_Analysis.png"))

    def update_results(self, best_rg, best_i0, MW, qmin_rg, qmax_rg):
        """
        Updates the GUI labels in the control panel with the latest analysis results.
        """
        self.rg_label.setText(f"Rg: {best_rg:.2f} Å")
        self.i0_label.setText(f"I(0): {best_i0:.2e}")
        self.qmin_rg_label.setText(f"qmin*Rg: {qmin_rg:.2f}")
        self.qmax_rg_label.setText(f"qmax*Rg: {qmax_rg:.2f}")
        self.mw_label.setText(f"MW: {MW:.0f} Da")

    def apply_to_all(self, q_range_manual=None):
        """
        Applies the *current* Guinier q-range to all loaded files.
        This is the "Ragtime" batch analysis.
        """
        if not self.files_data: return

        # --- Get optional frame range from GUI  ---
        try:
            start_frame_text = self.ragtime_start_frame_entry.text()
            end_frame_text = self.ragtime_end_frame_entry.text()
            
            # Default to all frames if boxes are empty
            start_frame = int(start_frame_text) if start_frame_text else 1
            end_frame = int(end_frame_text) if end_frame_text else len(self.files_data)

            if start_frame > end_frame:
                QMessageBox.warning(self, "Invalid Range", "Start frame must be <= End frame for Ragtime analysis.")
                return
            if start_frame < 1 or end_frame > len(self.files_data):
                QMessageBox.warning(self, "Invalid Range", f"Frame range must be between 1 and {len(self.files_data)}.")
                return

        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid frame number. Please enter integers for the Ragtime range.")
            return

        # --- Filter files to process based on range ---
        # Frame numbers are 1-based, list indices are 0-based
        # (start_frame - 1) is the 0-based index for the start
        # (end_frame) is the 0-based index for the *slice end*
        files_to_process = list(enumerate(self.files_data))[start_frame - 1 : end_frame]
        
        # Clear the I(0) dictionary before refilling it
        self.file_i0_values.clear()

        # --- Determine which q-range to use ---
        qmin_idx, qmax_idx = -1, -1
        use_manual_range = False
        if q_range_manual:
            # Range was passed directly (e.g., from manual_guinier_analysis)
            qmin_idx, qmax_idx = q_range_manual
            use_manual_range = True
        else:
            try:
                # Get range from the GUI text boxes
                qmin_idx = int(self.qmin_entry.text())
                qmax_idx = int(self.qmax_entry.text())
                use_manual_range = True
            except ValueError:
                # This should not happen if auto-analysis ran, but as a fallback.
                use_manual_range = False

        i0_values, rg_values, mw_values, file_numbers = [], [], [], []
        
        # --- Loop over all files in the specified range ---
        for file_index, (file_name, data) in files_to_process:
            q, I = data[:, 0], data[:, 1]
            i0, rg, mw = 0.0, 0.0, 0.0 # Default to 0 if fit fails

            # Check if q-range is valid for *this* file
            if use_manual_range and 0 <= qmin_idx < qmax_idx < len(q):
                I_slice = I[qmin_idx:qmax_idx+1]
                valid_mask = I_slice > 0
                if np.count_nonzero(valid_mask) >= 2:
                    q_squared_fit = q[qmin_idx:qmax_idx+1][valid_mask]**2
                    ln_I_fit = np.log(I_slice[valid_mask])
                    slope, intercept, _, _, _ = linregress(q_squared_fit, ln_I_fit)
                    
                    if slope < 0: # Check for valid fit
                        # --- MATH: Calculate Rg and I(0) ---
                        i0 = np.exp(intercept)
                        rg = np.sqrt(-3 * slope)
                        # --- MATH: Calculate MW ---
                        mw = self.estimate_molecular_weight(q, I, i0, rg)
            
            # Store results
            i0_values.append(i0)
            rg_values.append(rg)
            mw_values.append(mw)
            file_numbers.append(file_index + 1) # Use 1-based frame number
            
            # Store I(0) for the averaging thread
            self.file_i0_values[file_name] = i0

        if np.any(np.array(i0_values)): # Check if any fits were successful
            self.generate_2_panel_plot(i0_values, rg_values, mw_values, file_numbers)
            self._update_2panel_marker()
        else:
            QMessageBox.warning(self, "Ragtime Error", "Could not perform Guinier fit for any files in the selected range.")

    def generate_2_panel_plot(self, i0_values, rg_values, mw_values, file_numbers):
        """
        Generates the 2-panel Ragtime plot (Rg/I0 vs frame, MW/I0 vs frame)
        and saves the results to a text file.
        """

        self.fig2panel.clear()
        self.v_lines_2panel = []
        (ax1, ax2) = self.fig2panel.subplots(2, 1, sharex=True)
        i0, rg, mw, fn = np.array(i0_values), np.array(rg_values), np.array(mw_values), np.array(file_numbers)
        valid_mask_for_saving = ~np.isnan(i0) & ~np.isnan(rg) & ~np.isnan(mw)
        colb, colr = '#0D92F4', '#C62E2E'

        # --- Top Plot: I(0) and Rg vs Frame ---
        ax1.plot(fn, i0, color=colr, label="I(0)")
        ax1.set(ylabel="I(0)", title="Ragtime: I(0) and Rg vs Frame")
        ax1.tick_params(axis="y", labelcolor=colr)
        ax1b = ax1.twinx() # Create a second y-axis
        ax1b.plot(fn, rg, color=colb, label="Rg", marker='.', linestyle='', markersize=4)
        ax1b.set_ylabel("Rg (Å)", color=colb)
        ax1b.tick_params(axis="y", labelcolor=colb)

        # --- Bottom Plot: I(0) and Mw vs Frame ---
        ax2.plot(fn, i0, color=colr, label="I(0)")
        ax2.set(xlabel="Frame", ylabel="I(0)", title="Ragtime: I(0) and Mw vs Frame")
        ax2.tick_params(axis="y", labelcolor=colr)
        ax2b = ax2.twinx() # Create a second y-axis
        ax2b.plot(fn, mw, color=colb, label="Mw", marker='.', linestyle='', markersize=4)
        ax2b.set_ylabel("Mw (Da)", color=colb)
        ax2b.tick_params(axis="y", labelcolor=colb)

        # --- Add reference lines and zoom ---
        # Draw a dashed line at the "best" file's Rg value
        if self.reference_rg is not None and self.reference_rg > 0:
            ax1b.axhline(y=self.reference_rg, color='grey', linestyle='--')
            # Zoom the y-axis to be centered on the reference value
            y_center = self.reference_rg
            y_range = y_center * 0.4 # Zoom to +/- 20% of the reference value
            ax1b.set_ylim(y_center - y_range / 2, y_center + y_range / 2)

        # Draw a dashed line at the "best" file's MW value
        if self.reference_mw is not None and self.reference_mw > 0:
            ax2b.axhline(y=self.reference_mw, color='grey', linestyle='--')
            # Zoom the y-axis
            y_center = self.reference_mw
            y_range = y_center * 0.4
            ax2b.set_ylim(y_center - y_range / 2, y_center + y_range / 2)

        # This locks the x-axis to the plotted range, fixing the "zoom"
        # issue when a range is applied.
        if fn.size > 0:
            # Add a small padding of 1 frame to each side
            ax2.set_xlim(np.min(fn) - 1, np.max(fn) + 1)
        
        self.fig2panel.tight_layout()
        
        # Save plot images
        if self.output_path:
            self.fig2panel.savefig(os.path.join(self.output_path, "Ragtime_Analysis.png"))
            self.fig2panel.savefig(os.path.join(self.output_path, "Ragtime_Analysis.svg"))
        self.canvas2panel.draw()

        # --- Save data to text file ---
        if self.output_path:
            path_rg_mw = os.path.join(self.output_path, "Ragtime_Results.txt")
            header = "Frame\tI(0)\tRg\tMW(Da)"
            # Save the data, filtering out any rows with NaN values
            np.savetxt(path_rg_mw, np.c_[fn[valid_mask_for_saving], i0[valid_mask_for_saving], rg[valid_mask_for_saving], mw[valid_mask_for_saving]], header=header, fmt="%.4e")

    def extract_and_average_peak(self):
        """
        1. Copies files from a user-specified frame range into a new 'Peak-#' folder.
        2. Starts a background thread to average the copied files.
        """
        if not self.directory_path:
            QMessageBox.warning(self, "No Data", "Please select a data folder first.")
            return
        try:
            start_frame = int(self.first_frame_entry.text())
            end_frame = int(self.last_frame_entry.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Start and End frames must be integers.")
            return

        if start_frame > end_frame:
            QMessageBox.warning(self, "Invalid Range", "Start frame must be <= End frame.")
            return

        # --- Part 1: Find and match files ---
        # This regex logic tries to find the frame number in the filename.
        # It prioritizes numbers in {braces}, e.g., file_{123}.dat
        pattern_priority = re.compile(r'\{(\d+)\}')
        # Fallback: find the *last* number in the filename, e.g., file_123.dat
        pattern_fallback = re.compile(r'(\d+)')
        
        matched_files = []
        for filename, _ in self.files_data:
            frame_num = None
            match = pattern_priority.search(filename)
            if match:
                frame_num = int(match.group(1))
            else:
                fallback_matches = pattern_fallback.findall(filename)
                if fallback_matches:
                    frame_num = int(fallback_matches[-1]) # Get the last number
            
            # Check if the found frame number is in the user's range
            if frame_num is not None and start_frame <= frame_num <= end_frame:
                matched_files.append(filename)

        if not matched_files:
            QMessageBox.information(self, "No Matches", f"No files found in the frame range {start_frame}-{end_frame}.")
            return
            
        # --- Part 2: Create folder and copy files ---
        parent_dir = os.path.dirname(self.directory_path) # Folder above data
        peak_index = 1
        while True:
            # Find the next available "Peak-X" folder name
            target_folder = os.path.join(parent_dir, f"Peak-{peak_index}")
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
                break
            peak_index += 1
        self.last_peak_folder = target_folder

        # Copy all matched files to the new folder
        for f in matched_files:
            src = os.path.join(self.directory_path, f)
            dst = os.path.join(target_folder, f)
            shutil.copy2(src, dst) # copy2 preserves metadata
        
        print(f"Copied {len(matched_files)} files to {os.path.basename(target_folder)}")

        # --- Part 3: Start averaging thread ---
        self.extract_and_average_button.setEnabled(False) # Disable button
        QMessageBox.information(self, "Processing", f"Copied {len(matched_files)} files.\nStarting averaging...")

        # Create and start the averaging thread
        # Pass the I(0) values so the thread can normalize the data
        self.averaging_worker = PeakAveragingThread(self.last_peak_folder, self.file_i0_values, start_frame, end_frame)
        self.averaging_worker.averaging_finished.connect(self.finish_peak_averaging)
        self.averaging_worker.start()

    @Slot(str)
    def finish_peak_averaging(self, message):
        """
        Slot called when the PeakAveragingThread finishes.
        Displays the completion message.
        """
        QMessageBox.information(self, "Processing Complete", message)
        self.extract_and_average_button.setEnabled(True) # Re-enable button

    def _on_motion_update_coords(self, event):
        """
        Handles the mouse motion event ('motion_notify_event') on the plots
        to display the cursor's data coordinates.
        """
        if event.inaxes: # If the cursor is inside a plot's axes
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                # Update the coordinate label
                self.coord_label.setText(f"Cursor: (x={x:.3e}, y={y:.3e})")
        else:
            self.coord_label.setText("Cursor: (x=N/A, y=N/A)")

    def reset(self):
        """
        Resets the application to its initial state.
        """
        # Clear all data variables
        self.directory_path = None
        self.raw_files_data = []
        self.files_data, self.best_file, self.data, self.MW = [], None, None, None
        self.best_file_index = None
        self.current_file_index = None
        self.last_peak_folder = None
        self.file_i0_values.clear()
        self.reference_rg = None
        self.reference_mw = None
        self.reference_i0 = None
        self.v_lines_2panel = []

        # Clear all GUI input boxes
        self.qmin_entry.clear(); self.qmax_entry.clear()
        self.first_frame_entry.clear(); self.last_frame_entry.clear()
        self.frame_num_entry.clear()
        self.ragtime_start_frame_entry.clear()
        self.ragtime_end_frame_entry.clear()

        # Reset all labels
        self.folder_label.setText("Selected Folder: None")
        self.file_label.setText("No file selected")
        self.rg_label.setText("Rg: "); self.i0_label.setText("I(0): ")
        self.qmin_rg_label.setText("qmin*Rg: "); self.qmax_rg_label.setText("qmax*Rg: ")
        self.mw_label.setText("MW: ")
        self.coord_label.setText("Cursor position: (x=N/A, y=N/A)")

        # Clear plots
        self.clear_4_panel_plot() 
        self.fig2panel.clear(); self.canvas2panel.draw()
        
        # Disable buttons that require data
        self.manual_guinier_button.setEnabled(False)
        self.apply_to_all_button.setEnabled(False)
        self.extract_and_average_button.setEnabled(False)
        self.prev_file_button.setEnabled(False)
        self.best_file_button.setEnabled(False)
        self.next_file_button.setEnabled(False)
        self.goto_frame_button.setEnabled(False)

    def quit(self):
        """
        Closes the application.
        """
        self.close()