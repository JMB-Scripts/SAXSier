"""
SubMe v3.6: A tool for SAXS buffer subtraction.

This script provides a GUI to subtract background scattering from a series of SAXS profiles.

---
SUMMARY OF CHANGES (v3.6):
---
1. Unified 'Auto' and 'Custom' baseline methods into a single 'Baseline' tab.
2. Homogenized output folder naming conventions:
   - Average: SUB-Man-Avg-XX-YY
   - Auto Baseline: SUB-Auto-BL
   - Manual Baseline: SUB-Man-BL-XX-YY
3. Added << and >> buttons for stepping frame numbers by 10.
"""

import matplotlib
matplotlib.use('QtAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.widgets import SpanSelector
from scipy import integrate
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QLineEdit, QFileDialog, QMessageBox,
                             QGridLayout, QFrame, QTabWidget, QGroupBox)
from PySide6.QtCore import Qt, QThread, Signal, Slot

import os
import sys
import re
import shutil

###############################################
#
# Helper Functions
#
###############################################

def _read_dat_file(file_path):
    valid_data_rows = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip() or line.strip().startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        q, i, err = map(float, parts[:3])
                        if np.isnan(q) or np.isnan(i) or np.isnan(err):
                            continue
                        valid_data_rows.append([q, i, err])
                    except ValueError:
                        continue
    except Exception as e:
        print(f"❌ Error reading file {os.path.basename(file_path)}: {e}")
        return None
    if not valid_data_rows:
        print(f"⚠️ No valid data found in {os.path.basename(file_path)}")
        return None
    return np.array(valid_data_rows, dtype=np.float64)


def _parse_frame_number(filename):
    pattern_priority = re.compile(r'\{(\d+)\}')
    pattern_fallback = re.compile(r'(\d+)')
    frame_num = None
    match = pattern_priority.search(filename)
    if match:
        frame_num = int(match.group(1))
    else:
        fallback_matches = pattern_fallback.findall(filename)
        if fallback_matches:
            frame_num = int(fallback_matches[-1])
    return frame_num


def _average_files(file_list, common_q_grid):
    if not file_list:
        return np.zeros_like(common_q_grid), np.zeros_like(common_q_grid)
    sum_I = np.zeros_like(common_q_grid)
    sum_Err_sq = np.zeros_like(common_q_grid)
    count = 0
    for _, data in file_list:
        q, I, Err = data[:, 0], data[:, 1], data[:, 2]
        interp_I = np.interp(common_q_grid, q, I, left=np.nan, right=np.nan)
        interp_Err = np.interp(common_q_grid, q, Err, left=np.nan, right=np.nan)
        valid_mask = ~np.isnan(interp_I) & ~np.isnan(interp_Err)
        if np.any(valid_mask):
            sum_I[valid_mask] += interp_I[valid_mask]
            sum_Err_sq[valid_mask] += interp_Err[valid_mask]**2
            count += 1
    if count == 0:
        return np.zeros_like(common_q_grid), np.zeros_like(common_q_grid)
    avg_I = sum_I / count
    avg_Err = np.sqrt(sum_Err_sq) / count
    avg_I[np.isnan(avg_I)] = 0
    avg_Err[np.isnan(avg_Err)] = 0
    return avg_I, avg_Err


###############################################
#
# Worker Threads
#
###############################################

class FileLoadingThread(QThread):
    loading_finished = Signal(list)
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
    def run(self):
        all_files_data = []
        try:
            filenames = sorted([f for f in os.listdir(self.directory) if f.endswith(".dat")])
        except FileNotFoundError:
            self.loading_finished.emit([])
            return
        for filename in filenames:
            file_path = os.path.join(self.directory, filename)
            data = _read_dat_file(file_path)
            if data is not None and data.size > 0:
                all_files_data.append((filename, data))
        self.loading_finished.emit(all_files_data)


class AverageSubtractThread(QThread):
    subtraction_finished = Signal(str, list, list)
    def __init__(self, directory_path, all_files_data, files_to_average, start_frame, end_frame):
        super().__init__()
        self.directory_path = directory_path
        self.all_files_data = all_files_data
        self.files_to_average = files_to_average
        self.start_frame = start_frame
        self.end_frame = end_frame

    def run(self):
        try:
            parent_dir = os.path.dirname(self.directory_path)
            buffer_dir = os.path.join(parent_dir, "Buffer")
            # CHANGE: Naming convention SUB-Man-Avg-XX-YY
            sub_dir = os.path.join(parent_dir, f"SUB-Man-Avg-{self.start_frame}-{self.end_frame}")
            
            os.makedirs(buffer_dir, exist_ok=True)
            os.makedirs(sub_dir, exist_ok=True)
            if not self.files_to_average:
                self.subtraction_finished.emit("No files found in the specified buffer range.", [], [])
                return
            master_q_grid = self.all_files_data[0][1][:, 0]
            avg_I, avg_Err = _average_files(self.files_to_average, master_q_grid)
            avg_filename = os.path.join(buffer_dir, f"buffer-av-{self.start_frame}-{self.end_frame}.dat")
            np.savetxt(avg_filename, np.c_[master_q_grid, avg_I, avg_Err],
                       fmt="%.6e", header="q\tI(q) (Average)\tError", comments="# ")
            subtracted_integrals = []
            for filename, data in self.all_files_data:
                q, I, Err = data[:, 0], data[:, 1], data[:, 2]
                interp_avg_I = np.interp(q, master_q_grid, avg_I)
                interp_avg_Err = np.interp(q, master_q_grid, avg_Err)
                sub_I = I - interp_avg_I
                sub_Err = np.sqrt(Err**2 + interp_avg_Err**2)
                sub_filename = os.path.join(sub_dir, f"SUB_{filename}")
                np.savetxt(sub_filename, np.c_[q, sub_I, sub_Err],
                           fmt="%.6e", header="q\tI(q) (Subtracted)\tError", comments="# ")
                sub_integral = integrate.simpson(sub_I, x=q)
                subtracted_integrals.append(sub_integral)
            self.subtraction_finished.emit(f"Average subtraction complete.\nSaved to: {os.path.basename(sub_dir)}", subtracted_integrals, [])
        except Exception as e:
            self.subtraction_finished.emit(f"An error occurred: {e}", [], [])


class AutoBaselineSubtractThread(QThread):
    subtraction_finished = Signal(str, list, list) 
    def __init__(self, directory_path, all_files_data):
        super().__init__()
        self.directory_path = directory_path
        self.all_files_data = all_files_data
    def run(self):
        try:
            parent_dir = os.path.dirname(self.directory_path)
            # CHANGE: Naming convention SUB-Auto-BL
            sub_dir = os.path.join(parent_dir, "SUB-Auto-BL")
            os.makedirs(sub_dir, exist_ok=True)
            if len(self.all_files_data) < 20:
                self.subtraction_finished.emit("Not enough files for auto baseline (need at least 20).", [], [])
                return
            master_q_grid = self.all_files_data[0][1][:, 0]
            files_start = self.all_files_data[:10]
            files_end = self.all_files_data[-10:]
            I_start_avg, Err_start_avg = _average_files(files_start, master_q_grid)
            I_end_avg, Err_end_avg = _average_files(files_end, master_q_grid)
            total_frames = len(self.all_files_data)
            subtracted_integrals = []
            baseline_integrals = [] 
            for i in range(total_frames):
                filename, data = self.all_files_data[i]
                q, I, Err = data[:, 0], data[:, 1], data[:, 2]
                factor = i / (total_frames - 1) if total_frames > 1 else 0
                baseline_I = I_start_avg + (I_end_avg - I_start_avg) * factor
                baseline_Err_sq = (1-factor)**2 * Err_start_avg**2 + factor**2 * Err_end_avg**2
                baseline_Err = np.sqrt(np.maximum(0, baseline_Err_sq))
                interp_baseline_I = np.interp(q, master_q_grid, baseline_I)
                interp_baseline_Err = np.interp(q, master_q_grid, baseline_Err)
                sub_I = I - interp_baseline_I
                sub_Err = np.sqrt(Err**2 + interp_baseline_Err**2)
                sub_filename = os.path.join(sub_dir, f"SUB_bl_{filename}")
                np.savetxt(sub_filename, np.c_[q, sub_I, sub_Err],
                           fmt="%.6e", header="q\tI(q) (Subtracted)\tError", comments="# ")
                sub_integral = integrate.simpson(sub_I, x=q)
                subtracted_integrals.append(sub_integral)
                baseline_integral = integrate.simpson(interp_baseline_I, x=q)
                baseline_integrals.append(baseline_integral)
            self.subtraction_finished.emit(f"Auto baseline subtraction complete.\nSaved to: {os.path.basename(sub_dir)}", subtracted_integrals, baseline_integrals)
        except Exception as e:
            self.subtraction_finished.emit(f"An error occurred: {e}", [], [])


class CustomBaselineSubtractThread(QThread):
    subtraction_finished = Signal(str, list, list) 
    
    def __init__(self, directory_path, all_files_data, files_start_region, files_end_region, x1_center, x2_center, output_directory):
        super().__init__()
        self.all_files_data = all_files_data
        self.files_start_region = files_start_region
        self.files_end_region = files_end_region
        self.x1_center = x1_center
        self.x2_center = x2_center
        self.output_directory = output_directory

    def run(self):
        try:
            sub_dir = self.output_directory
            os.makedirs(sub_dir, exist_ok=True)

            if not self.files_start_region or not self.files_end_region:
                self.subtraction_finished.emit("Start or End region is empty. No files processed.", [], [])
                return
                
            master_q_grid = self.all_files_data[0][1][:, 0]
            
            I_start_avg, Err_start_avg = _average_files(self.files_start_region, master_q_grid)
            I_end_avg, Err_end_avg = _average_files(self.files_end_region, master_q_grid)
            
            x1, x2 = self.x1_center, self.x2_center
            if x1 == x2:
                self.subtraction_finished.emit("Start and End regions cannot be the same.", [], [])
                return

            subtracted_integrals = []
            baseline_integrals = []

            for filename, data in self.all_files_data:
                q, I, Err = data[:, 0], data[:, 1], data[:, 2]
                frame_num = _parse_frame_number(filename)
                if frame_num is None:
                    print(f"Skipping {filename}, could not parse frame number.")
                    continue
                    
                factor = (frame_num - x1) / (x2 - x1)
                
                baseline_I = I_start_avg + (I_end_avg - I_start_avg) * factor
                baseline_Err_sq = (1-factor)**2 * Err_start_avg**2 + factor**2 * Err_end_avg**2
                baseline_Err = np.sqrt(np.maximum(0, baseline_Err_sq))
                
                interp_baseline_I = np.interp(q, master_q_grid, baseline_I)
                interp_baseline_Err = np.interp(q, master_q_grid, baseline_Err)
                
                sub_I = I - interp_baseline_I
                sub_Err = np.sqrt(Err**2 + interp_baseline_Err**2)
                
                sub_filename = os.path.join(sub_dir, f"SUB_bl_{filename}")
                np.savetxt(sub_filename, np.c_[q, sub_I, sub_Err],
                           fmt="%.6e", header="q\tI(q) (Subtracted)\tError", comments="# ")

                sub_integral = integrate.simpson(sub_I, x=q)
                subtracted_integrals.append(sub_integral)
                baseline_integral = integrate.simpson(interp_baseline_I, x=q)
                baseline_integrals.append(baseline_integral)

            self.subtraction_finished.emit(f"Custom baseline subtraction complete.\nSaved to: {os.path.basename(sub_dir)}", subtracted_integrals, baseline_integrals)

        except Exception as e:
            self.subtraction_finished.emit(f"An error occurred: {e}", [], [])


###############################################
#
# Main Application Class
#
###############################################

class SubMeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SubMe: Buffer Subtraction Tool (v3.6)")
        self.setGeometry(150, 150, 1000, 750)
        
        self.directory_path = None
        self.raw_files_data = []
        self.files_data = []
        self.frame_numbers = []
        self.integrals = []
        self.filename_map = {} 

        self.span_selector_avg = None
        self.vline_cust1 = None 
        self.vline_cust2 = None 
        self.baseline_plot_line = None 
        self.auto_baseline_plot_line = None
        self.subtracted_plot_lines = []

        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax_twin = None

        self.setup_ui()
        self.reset()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        control_widget = QFrame()
        control_layout = QVBoxLayout(control_widget)
        control_widget.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        control_widget.setMaximumWidth(320) # Slightly wider for new buttons

        self.browse_button = QPushButton("Select Data Folder")
        self.browse_button.clicked.connect(self.browse_directory)
        self.folder_label = QLabel("Selected Folder: None")
        self.folder_label.setWordWrap(True)
        control_layout.addWidget(self.browse_button)
        control_layout.addWidget(self.folder_label)
        control_layout.addSpacing(10)

        self.tabs = QTabWidget()
        control_layout.addWidget(self.tabs)
        
        # --- Tab 1: Average Subtraction ---
        self.tab_avg = QWidget()
        tab_avg_layout = QVBoxLayout(self.tab_avg)
        tab_avg_layout.addWidget(QLabel("<b>Method 1: Average Buffer</b>"))
        tab_avg_layout.addWidget(QLabel("Subtract an average buffer calculated\nfrom a range of frames."))
        
        avg_sub_layout = QGridLayout()
        self.avg_start_entry = QLineEdit()
        self.avg_start_entry.setPlaceholderText("Start")
        self.avg_end_entry = QLineEdit()
        self.avg_end_entry.setPlaceholderText("End")
        avg_sub_layout.addWidget(QLabel("Start Frame:"), 0, 0)
        avg_sub_layout.addWidget(self.avg_start_entry, 0, 1)
        avg_sub_layout.addWidget(QLabel("End Frame:"), 1, 0)
        avg_sub_layout.addWidget(self.avg_end_entry, 1, 1)
        
        tab_avg_layout.addLayout(avg_sub_layout)
        self.subtract_average_button = QPushButton("Subtract Average Buffer")
        self.subtract_average_button.clicked.connect(self.run_average_subtract)
        tab_avg_layout.addWidget(self.subtract_average_button)
        tab_avg_layout.addStretch()
        self.tabs.addTab(self.tab_avg, "Average")
        
        # --- Tab 2: Baseline Subtraction (Merged) ---
        self.tab_baseline = QWidget()
        tab_bl_layout = QVBoxLayout(self.tab_baseline)
        
        # ... Section: Auto Baseline ...
        grp_auto = QGroupBox("Method 2: Auto Linear Baseline")
        grp_auto_layout = QVBoxLayout(grp_auto)
        grp_auto_layout.addWidget(QLabel("Uses the first 10 and last 10 frames\nto calculate a linear baseline."))
        self.subtract_auto_button = QPushButton("Subtract Auto Baseline")
        self.subtract_auto_button.clicked.connect(self.run_auto_baseline_subtract)
        grp_auto_layout.addWidget(self.subtract_auto_button)
        tab_bl_layout.addWidget(grp_auto)
        
        tab_bl_layout.addSpacing(10)

        # ... Section: Manual Baseline ...
        grp_manual = QGroupBox("Method 3: Manual Linear Baseline")
        grp_manual_layout = QVBoxLayout(grp_manual)
        grp_manual_layout.addWidget(QLabel("Define a baseline using two specific frames."))
        
        grid_man = QGridLayout()
        
        # Row 1: Frame 1 controls
        self.cust_frame1_entry = QLineEdit()
        self.cust_frame1_entry.setAlignment(Qt.AlignCenter)
        self.cust_frame1_entry.setPlaceholderText("Frame 1")
        
        self.btn_f1_dec10 = QPushButton("<<")
        self.btn_f1_dec1 = QPushButton("<")
        self.btn_f1_inc1 = QPushButton(">")
        self.btn_f1_inc10 = QPushButton(">>")
        
        # Set widths for buttons
        for btn in [self.btn_f1_dec10, self.btn_f1_dec1, self.btn_f1_inc1, self.btn_f1_inc10]:
            btn.setMaximumWidth(30)

        grid_man.addWidget(QLabel("F1:"), 0, 0)
        grid_man.addWidget(self.btn_f1_dec10, 0, 1)
        grid_man.addWidget(self.btn_f1_dec1, 0, 2)
        grid_man.addWidget(self.cust_frame1_entry, 0, 3)
        grid_man.addWidget(self.btn_f1_inc1, 0, 4)
        grid_man.addWidget(self.btn_f1_inc10, 0, 5)

        # Row 2: Frame 2 controls
        self.cust_frame2_entry = QLineEdit()
        self.cust_frame2_entry.setAlignment(Qt.AlignCenter)
        self.cust_frame2_entry.setPlaceholderText("Frame 2")
        
        self.btn_f2_dec10 = QPushButton("<<")
        self.btn_f2_dec1 = QPushButton("<")
        self.btn_f2_inc1 = QPushButton(">")
        self.btn_f2_inc10 = QPushButton(">>")
        
        # Set widths for buttons
        for btn in [self.btn_f2_dec10, self.btn_f2_dec1, self.btn_f2_inc1, self.btn_f2_inc10]:
            btn.setMaximumWidth(30)

        grid_man.addWidget(QLabel("F2:"), 1, 0)
        grid_man.addWidget(self.btn_f2_dec10, 1, 1)
        grid_man.addWidget(self.btn_f2_dec1, 1, 2)
        grid_man.addWidget(self.cust_frame2_entry, 1, 3)
        grid_man.addWidget(self.btn_f2_inc1, 1, 4)
        grid_man.addWidget(self.btn_f2_inc10, 1, 5)
        
        grp_manual_layout.addLayout(grid_man)
        
        self.subtract_custom_button = QPushButton("Subtract Manual Baseline")
        self.subtract_custom_button.clicked.connect(self.run_custom_baseline_subtract)
        grp_manual_layout.addWidget(self.subtract_custom_button)
        
        tab_bl_layout.addWidget(grp_manual)
        tab_bl_layout.addStretch()
        
        self.tabs.addTab(self.tab_baseline, "Baseline")

        # --- Connect signals ---
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.avg_start_entry.editingFinished.connect(self.update_plot_widgets)
        self.avg_end_entry.editingFinished.connect(self.update_plot_widgets)
        
        self.cust_frame1_entry.editingFinished.connect(self.update_plot_widgets)
        self.cust_frame2_entry.editingFinished.connect(self.update_plot_widgets)
        
        # Connect new buttons with specific lambda-like slots or dedicated slots
        self.btn_f1_dec10.clicked.connect(lambda: self._adjust_frame(self.cust_frame1_entry, -10))
        self.btn_f1_dec1.clicked.connect(lambda: self._adjust_frame(self.cust_frame1_entry, -1))
        self.btn_f1_inc1.clicked.connect(lambda: self._adjust_frame(self.cust_frame1_entry, 1))
        self.btn_f1_inc10.clicked.connect(lambda: self._adjust_frame(self.cust_frame1_entry, 10))
        
        self.btn_f2_dec10.clicked.connect(lambda: self._adjust_frame(self.cust_frame2_entry, -10))
        self.btn_f2_dec1.clicked.connect(lambda: self._adjust_frame(self.cust_frame2_entry, -1))
        self.btn_f2_inc1.clicked.connect(lambda: self._adjust_frame(self.cust_frame2_entry, 1))
        self.btn_f2_inc10.clicked.connect(lambda: self._adjust_frame(self.cust_frame2_entry, 10))

        control_layout.addStretch()

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset)
        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)
        control_layout.addWidget(self.reset_button)
        control_layout.addWidget(self.quit_button)
        self.coord_label = QLabel("Cursor: (x=N/A, y=N/A)")
        control_layout.addWidget(self.coord_label)

        main_layout.addWidget(control_widget, stretch=1)
        main_layout.addWidget(self.canvas, stretch=3)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion_update_coords)

    # --- Plotting and UI Management ---

    def clear_plot_widgets(self):
        """Helper to remove all interactive widgets from the plot."""
        if self.span_selector_avg:
            self.span_selector_avg.set_visible(False)
            self.span_selector_avg = None
        
        if self.vline_cust1:
            try: self.vline_cust1.remove()
            except (ValueError, AttributeError): pass
            self.vline_cust1 = None
        if self.vline_cust2:
            try: self.vline_cust2.remove()
            except (ValueError, AttributeError): pass
            self.vline_cust2 = None
            
        if self.baseline_plot_line:
            try: self.baseline_plot_line.remove()
            except (ValueError, AttributeError): pass
            self.baseline_plot_line = None
        if self.auto_baseline_plot_line:
            try: self.auto_baseline_plot_line.remove()
            except (ValueError, AttributeError): pass
            self.auto_baseline_plot_line = None
            
    def on_tab_changed(self, index):
        self.plot_integrals()

    def update_plot_widgets(self):
        """
        Clears and redraws the correct SpanSelectors and previews
        for the currently visible tab.
        """
        self.clear_plot_widgets() # Clear all widgets first
        current_tab_index = self.tabs.currentIndex()

        if current_tab_index == 0: # Average Subtraction
            try: s = int(self.avg_start_entry.text())
            except ValueError: s = 0
            try: e = int(self.avg_end_entry.text())
            except ValueError: e = 0
            self.span_selector_avg = SpanSelector(
                self.ax, self.on_span_select_avg, 'horizontal',
                props=dict(alpha=0.3, facecolor='red'),
                interactive=True, drag_from_anywhere=True,
                useblit=False 
            )
            if e > s: self.span_selector_avg.extents = (s, e)
        
        elif current_tab_index == 1: # Baseline (Combined Tab)
            # Draw Auto Baseline preview (Magenta)
            self.draw_auto_baseline_line()

            # Draw Manual Baseline preview (Green) & Vertical Lines
            try:
                frame1 = int(self.cust_frame1_entry.text())
                if frame1 in self.frame_numbers:
                    self.vline_cust1 = self.ax.axvline(frame1, color='blue', linestyle=':', label="Man F1")
            except ValueError: pass
            
            try:
                frame2 = int(self.cust_frame2_entry.text())
                if frame2 in self.frame_numbers:
                    self.vline_cust2 = self.ax.axvline(frame2, color='green', linestyle=':', label="Man F2")
            except ValueError: pass
            
            self.draw_custom_baseline_line()
            
        self.update_legend()
        self.canvas.draw()
            
    def plot_integrals(self, subtracted_integrals=None, baseline_integrals=None):
        self.clear_plot_widgets()
        self.ax.clear()
        
        if self.ax_twin:
            self.ax_twin.remove()
            self.ax_twin = None
        
        if not self.frame_numbers:
            self.ax.set_title("Load data to see integral plot")
            self.canvas.draw()
            return
            
        self.ax.plot(self.frame_numbers, self.integrals, marker='.', linestyle='-', markersize=4, label="Raw Integral")
        
        has_plot = False
        if baseline_integrals:
            plot_frames = self.frame_numbers[:len(baseline_integrals)]
            self.ax.plot(plot_frames, baseline_integrals, linestyle='--', color='purple', label="Calculated Baseline")
            has_plot = True

        if subtracted_integrals:
            plot_frames = self.frame_numbers[:len(subtracted_integrals)]
            self.ax_twin = self.ax.twinx()
            self.ax_twin.plot(plot_frames, subtracted_integrals, marker='.', linestyle='--', markersize=4, color='orange', label="Subtracted Integral")
            self.ax_twin.set_ylabel("Subtracted Integral", color='orange')
            self.ax_twin.tick_params(axis='y', labelcolor='orange')
            min_val = np.min(subtracted_integrals)
            max_val = np.max(subtracted_integrals)
            padding = (max_val - min_val) * 0.1 if (max_val-min_val) != 0 else 1.0
            self.ax_twin.set_ylim(min_val - (padding*2), max_val + padding)
            has_plot = True
            
        self.ax.set_xlabel("Frame Number")
        self.ax.set_ylabel("Total Integral (Raw)") 
        self.ax.set_title("Integral vs. Frame Number")
        
        self.ax.grid(True, linestyle=':', alpha=0.7)
        self.fig.tight_layout()
        
        self.update_plot_widgets()
        self.update_legend()
        self.canvas.draw()

    def update_legend(self):
        handles, labels = self.ax.get_legend_handles_labels()
        if self.ax_twin: 
            lines2, labels2 = self.ax_twin.get_legend_handles_labels()
            by_label = dict(zip(labels + labels2, handles + lines2))
            self.ax.legend(by_label.values(), by_label.keys(), loc='upper left')
        else:
            by_label = dict(zip(labels, handles))
            self.ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    def draw_auto_baseline_line(self):
        if self.auto_baseline_plot_line:
            try: self.auto_baseline_plot_line.remove()
            except (ValueError, AttributeError): pass
            self.auto_baseline_plot_line = None

        if len(self.integrals) < 20 or len(self.frame_numbers) < 20:
            return

        frames_np = np.array(self.frame_numbers)
        integrals_np = np.array(self.integrals)
        
        integrals_start = integrals_np[:10]
        integrals_end = integrals_np[-10:]
        frames_start = frames_np[:10]
        frames_end = frames_np[-10:]

        y1_avg = np.mean(integrals_start)
        x1_center = np.mean(frames_start)
        y2_avg = np.mean(integrals_end)
        x2_center = np.mean(frames_end)

        if x1_center == x2_center: return

        slope = (y2_avg - y1_avg) / (x2_center - x1_center)
        intercept = y1_avg - slope * x1_center
        baseline_y_values = slope * frames_np + intercept
        
        line_artist, = self.ax.plot(frames_np, baseline_y_values, 'm--', label="Auto Baseline Preview", alpha=0.5)
        self.auto_baseline_plot_line = line_artist
        
    def draw_custom_baseline_line(self):
        if self.baseline_plot_line:
            try: self.baseline_plot_line.remove()
            except (ValueError, AttributeError): pass
            self.baseline_plot_line = None

        try:
            frame1 = int(self.cust_frame1_entry.text())
            frame2 = int(self.cust_frame2_entry.text())
        except ValueError:
            self.canvas.draw(); return # One of the fields is empty
        
        if not (frame1 in self.frame_numbers and frame2 in self.frame_numbers):
            self.canvas.draw(); return # Invalid frame numbers
            
        if frame1 == frame2:
            self.canvas.draw(); return # Frames are the same

        # Find integral values for these frames
        integral1 = self.integrals[self.frame_numbers.index(frame1)]
        integral2 = self.integrals[self.frame_numbers.index(frame2)]
        
        frames_np = np.array(self.frame_numbers)
        
        if frame1 > frame2: # Ensure p1 is always before p2
            frame1, frame2 = frame2, frame1
            integral1, integral2 = integral2, integral1

        # Calculate the line
        slope = (integral2 - integral1) / (frame2 - frame1)
        intercept = integral1 - slope * frame1
        baseline_y_values = slope * frames_np + intercept
        
        line_artist, = self.ax.plot(frames_np, baseline_y_values, 'g--', label="Manual Baseline Preview")
        self.baseline_plot_line = line_artist
        
        self.update_legend()
        self.canvas.draw()

    # --- SpanSelector Callbacks ---

    @Slot(float, float)
    def on_span_select_avg(self, xmin, xmax):
        self.avg_start_entry.blockSignals(True)
        self.avg_end_entry.blockSignals(True)
        self.avg_start_entry.setText(str(int(np.round(xmin))))
        self.avg_end_entry.setText(str(int(np.round(xmax))))
        self.avg_start_entry.blockSignals(False)
        self.avg_end_entry.blockSignals(False)

    # --- +/- button slots ---
    
    def _adjust_frame(self, entry_widget, increment):
        """Helper function to increment/decrement a frame number entry."""
        if not self.frame_numbers:
            return # No data loaded

        try:
            current_val = int(entry_widget.text())
        except ValueError:
            # If field is empty, start from the middle
            current_val = self.frame_numbers[len(self.frame_numbers) // 2]
        
        new_val = current_val + increment
        
        # Clamp value to be within the loaded frame range
        min_frame = self.frame_numbers[0]
        max_frame = self.frame_numbers[-1]
        
        if new_val < min_frame:
            new_val = min_frame
        if new_val > max_frame:
            new_val = max_frame
            
        entry_widget.setText(str(new_val))
        self.update_plot_widgets() # Real-time update
    

    # --- Core Application Logic ---

    def browse_directory(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not folder_path: return
        self.reset()
        self.directory_path = folder_path
        self.folder_label.setText(f"Folder: {os.path.basename(folder_path)}")
        self.browse_button.setText("Loading...")
        self.browse_button.setEnabled(False)
        self.worker = FileLoadingThread(folder_path)
        self.worker.loading_finished.connect(self.finish_file_loading)
        self.worker.start()

    @Slot(list)
    def finish_file_loading(self, raw_files_data):
        self.browse_button.setText("Select Data Folder")
        self.browse_button.setEnabled(True)
        if not raw_files_data:
            QMessageBox.warning(self, "No Data", "No valid .dat files could be read.")
            return
        self.raw_files_data = raw_files_data
        first_file_q = self.raw_files_data[0][1][:, 0]
        needs_conversion = np.max(first_file_q) > 2.0
        if needs_conversion:
             QMessageBox.information(self, "Unit Detection", "q-values appear to be in nm⁻¹. They will be converted to Å⁻¹.")
        self.files_data = []
        for filename, raw_data in self.raw_files_data:
            processed_data = raw_data.copy()
            if needs_conversion:
                processed_data[:, 0] *= 0.1
            self.files_data.append((filename, processed_data))
        print(f"✅ Successfully loaded and processed {len(self.files_data)} files.")
        
        if self.calculate_all_integrals_no_plot(): # 1. Calculate data
            self.auto_detect_regions()             # 2. Pre-fill fields
            self.plot_integrals()                  # 3. Plot (will read pre-filled fields)
            self.set_buttons_enabled(True)

    def calculate_all_integrals_no_plot(self):
        self.integrals = []
        self.frame_numbers = []
        self.filename_map = {}
        for filename, data in self.files_data:
            q, I = data[:, 0], data[:, 1]
            integral = integrate.simpson(I, x=q)
            frame_num = _parse_frame_number(filename)
            if frame_num is not None:
                self.integrals.append(integral)
                self.frame_numbers.append(frame_num)
                self.filename_map[frame_num] = filename
            else:
                print(f"Warning: Could not parse frame number from {filename}")
        if not self.frame_numbers:
            QMessageBox.warning(self, "Error", "Could not parse frame numbers from any files.")
            return False 
        return True 

    def calculate_all_integrals(self):
        if self.calculate_all_integrals_no_plot():
            self.plot_integrals()
            
    def auto_detect_regions(self):
        if len(self.integrals) < 50:
            return

        try:
            baseline_data = np.array(self.integrals[:30])
            baseline_mean = np.mean(baseline_data)
            baseline_std = np.std(baseline_data)
            threshold = baseline_mean + (5 * baseline_std)
            
            integrals_np = np.array(self.integrals)
            above_threshold_mask = integrals_np > threshold
            
            if not np.any(above_threshold_mask):
                return 

            above_indices = np.where(above_threshold_mask)[0]
            peak_start_index = above_indices[0]
            peak_end_index = above_indices[-1]
            
            min_frame = self.frame_numbers[0]
            max_frame = self.frame_numbers[-1]
            padding = 10 
            
            avg_start = min_frame
            avg_end = self.frame_numbers[peak_start_index] - padding
            avg_end = max(avg_start + 1, avg_end) 
            
            cust_frame1 = avg_end
            cust_frame2 = self.frame_numbers[peak_end_index] + padding
            cust_frame2 = min(max_frame, cust_frame2)

            self.avg_start_entry.setText(str(avg_start))
            self.avg_end_entry.setText(str(avg_end))
            self.cust_frame1_entry.setText(str(cust_frame1))
            self.cust_frame2_entry.setText(str(cust_frame2))
            
        except Exception as e:
            print(f"Error during peak auto-detection: {e}")

    def set_buttons_enabled(self, enabled):
        self.subtract_average_button.setEnabled(enabled)
        self.subtract_auto_button.setEnabled(enabled)
        self.subtract_custom_button.setEnabled(enabled)
        self.browse_button.setEnabled(enabled)
        
        if hasattr(self, 'btn_f1_dec1'): 
            for btn in [self.btn_f1_dec1, self.btn_f1_inc1, self.btn_f1_dec10, self.btn_f1_inc10,
                        self.btn_f2_dec1, self.btn_f2_inc1, self.btn_f2_dec10, self.btn_f2_inc10]:
                btn.setEnabled(enabled)

    # --- "Run" functions for each tab ---

    def run_average_subtract(self):
        try:
            start_frame = int(self.avg_start_entry.text())
            end_frame = int(self.avg_end_entry.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Buffer Start and End frames must be integers.")
            return
        if start_frame >= end_frame:
            QMessageBox.warning(self, "Invalid Range", "Start frame must be < End frame.")
            return
            
        files_to_average = []
        for i in range(start_frame, end_frame + 1):
            filename = self.filename_map.get(i)
            if filename:
                for f_name, data in self.files_data:
                    if f_name == filename: files_to_average.append((f_name, data)); break
        if not files_to_average:
            QMessageBox.warning(self, "No Files", "No files found matching the specified frame range.")
            return

        self.set_buttons_enabled(False)
        self.avg_worker = AverageSubtractThread(
            self.directory_path, self.files_data, files_to_average, start_frame, end_frame
        )
        self.avg_worker.subtraction_finished.connect(self.on_subtraction_finished)
        self.avg_worker.start()

    def run_auto_baseline_subtract(self):
        if len(self.files_data) < 20:
            QMessageBox.warning(self, "Not Enough Data", "Auto baseline subtraction requires at least 20 files.")
            return
        self.set_buttons_enabled(False)
        self.auto_worker = AutoBaselineSubtractThread(self.directory_path, self.files_data)
        self.auto_worker.subtraction_finished.connect(self.on_subtraction_finished)
        self.auto_worker.start()

    def run_custom_baseline_subtract(self):
        try:
            frame1 = int(self.cust_frame1_entry.text())
            frame2 = int(self.cust_frame2_entry.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Frame 1 and Frame 2 must be integers.")
            return
        
        if not (frame1 in self.frame_numbers and frame2 in self.frame_numbers):
            QMessageBox.warning(self, "Invalid Frames", "Frame numbers not found in the loaded data.")
            return
            
        if frame1 == frame2:
             QMessageBox.warning(self, "Invalid Range", "Frame 1 and Frame 2 must be different.")
             return
             
        filename1 = self.filename_map.get(frame1)
        filename2 = self.filename_map.get(frame2)
        
        curve1_data = next((data for f, data in self.files_data if f == filename1), None)
        curve2_data = next((data for f, data in self.files_data if f == filename2), None)

        if curve1_data is None or curve2_data is None:
            QMessageBox.critical(self, "Data Error", "Could not find the data for the specified frames.")
            return

        files_start_region = [(filename1, curve1_data)]
        files_end_region = [(filename2, curve2_data)]
        
        x1_center = frame1
        x2_center = frame2
        
        if x1_center > x2_center:
            x1_center, x2_center = x2_center, x1_center
            files_start_region, files_end_region = files_end_region, files_start_region

        parent_dir = os.path.dirname(self.directory_path)
        # CHANGE: Naming convention SUB-Man-BL-XX-YY
        target_folder = os.path.join(parent_dir, f"SUB-Man-BL-{x1_center}-{x2_center}")

        self.set_buttons_enabled(False)
        self.custom_worker = CustomBaselineSubtractThread(
            self.directory_path, self.files_data, 
            files_start_region, files_end_region,
            x1_center, x2_center,
            target_folder
        )
        self.custom_worker.subtraction_finished.connect(self.on_subtraction_finished)
        self.custom_worker.start()

    @Slot(str, list, list)
    def on_subtraction_finished(self, message, subtracted_integrals, baseline_integrals):
        QMessageBox.information(self, "Processing Complete", message)
        self.set_buttons_enabled(True)
        self.plot_integrals(subtracted_integrals, baseline_integrals)
        
    def _on_motion_update_coords(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                self.coord_label.setText(f"Cursor: (x={x:.1f}, y={y:.3e})")
        else:
            self.coord_label.setText("Cursor: (x=N/A, y=N/A)")

    def reset(self):
        self.directory_path = None
        self.raw_files_data = []
        self.files_data = []
        self.frame_numbers = []
        self.integrals = []
        self.filename_map = {}
        
        if hasattr(self, 'avg_start_entry'):
            self.avg_start_entry.clear()
            self.avg_end_entry.clear()
            self.cust_frame1_entry.clear()
            self.cust_frame2_entry.clear()
            self.folder_label.setText("Selected Folder: None")
            self.coord_label.setText("Cursor: (x=N/A, y=N/A)")
            self.set_buttons_enabled(False)
            self.browse_button.setEnabled(True)

        self.plot_integrals()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QMainWindow, QWidget { background-color: #f0f0f0; }
        QPushButton {
            background-color: #0078d7;
            color: white;
            border-radius: 5px;
            padding: 6px;
            font-size: 13px;
        }
        QPushButton:hover { background-color: #005a9e; }
        QPushButton:disabled { background-color: #aaaaaa; }
        
        /* Style for arrow buttons */
        QPushButton[text="<"], QPushButton[text=">"], 
        QPushButton[text="<<"], QPushButton[text=">>"] {
            font-weight: bold;
            padding: 2px;
        }
        
        QFrame, QGroupBox {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 5px;
            margin-top: 1ex; /* space for title */
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 3px;
            color: #333333;
            font-weight: bold;
        }
        
        QLabel { font-size: 12px; }
        QLineEdit { padding: 5px; border: 1px solid #cccccc; border-radius: 3px; }
        QTabWidget::pane { border-top: 1px solid #cccccc; }
        QTabBar::tab { padding: 8px; }
        QTabBar::tab:selected { background-color: #ffffff; border: 1px solid #cccccc; border-bottom: none; }
    """)
    main_window = SubMeApp()
    main_window.show()
    sys.exit(app.exec())