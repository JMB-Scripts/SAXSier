"""
SAXSting v5.0: SAXS Data Analysis & Comparison Tool
---------------------------------------------------
A GUI for high-throughput SAXS primary analysis, designed for rapid comparison,
Guinier refinement, and publication-quality data export.

--- USER MANUAL & WORKFLOW ---

1. DATA LOADING
   - Drag & Drop .dat files into the list or use the "Add .dat Files" button.
   - Unit Detection: If data is in nm^-1, it is automatically converted to A^-1 
     (a summary warning will appear if conversion occurred).

2. PRIMARY ANALYSIS (Guinier)
   - Click [Auto-Process]: Runs the "Sexier" algorithm on ALL files to automatically 
     find the optimal linear Guinier region (optimizing for qmax*Rg ~ 1.3).
   - Manual Refinement: 
     1. Select a specific file in the list.
     2. Adjust the 'qmin index' and 'qmax index' in the "Manual Refinement" box.
     3. Click [Refine Selected] to update only that file.

3. MOLECULAR WEIGHT & OLIGOMERS
   - Enter the theoretical Monomer MW (in kDa) in the "Expected MW" box.
   - Click [Calculate State] to compute the oligomeric state (e.g., 2.1 ≈ Dimer) 
     based on the Volume of Correlation (Vc) derived MW.

4. VISUALIZATION
   - Use [Raw] to see original intensities or [Superimpose] to normalize by I(0).
   - Toggle [Log-Log Plot] for the scattering curve.
   - Hover over any plot to see precise X/Y coordinates in the status bar.

5. DATA OPERATIONS
   - Average: Check specific files in the list and click [Average Checked Files] 
     to generate and save a normalized, averaged dataset.

6. SAVING & EXPORT
   - Click [Save Full Report] and select a parent directory.
   - The tool creates a smart folder (e.g., "SAXSting-01") containing:
     a) Plots (PNG & SVG) for Curves, Guinier fits, and Kratky plots.
     b) Summary Table (TXT) with all parameters and limits.
     c) /source_data_txt/ folder: Individual .txt files for every sample 
        (Curve data, Guinier Fits+Residuals, Kratky coords) for replotting in Excel/Origin.
"""

import matplotlib
matplotlib.use('QtAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from scipy import integrate
from scipy.stats import linregress
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QListWidget, QFileDialog, QMessageBox,
                             QGridLayout, QFrame, QTableWidget, QTableWidgetItem,
                             QAbstractItemView, QTabWidget, QLineEdit, QGroupBox, QListWidgetItem,
                             QHeaderView, QButtonGroup, QCheckBox, QStatusBar, QInputDialog)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor, QPalette

import os
import sys
import datetime

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
        return None
    return np.array(valid_data_rows, dtype=np.float64)


def _sexier_auto_search(q, I):
    """
    The exact algorithm from Sexier v7.2 to find the best Guinier region.
    """
    positive_mask = I > 0
    if not np.any(positive_mask): return None
        
    q_sq = q**2
    with np.errstate(invalid='ignore'):
        ln_I = np.log(I)

    best_score = -np.inf
    best_result = None

    min_pts = 5
    max_pts = 160
    max_start = 40
    
    data_len = len(q)
    search_len = min(data_len, 400) 

    for size in range(min_pts, min(max_pts, search_len)):
        for start in range(0, min(max_start, search_len - size)):
            end = start + size
            
            q_win = q[start:end]
            I_win = I[start:end]
            lnI_win = ln_I[start:end]
            qsq_win = q_sq[start:end]
            
            if np.any(np.isinf(lnI_win)) or np.any(np.isnan(lnI_win)): continue

            slope, intercept, r_value, _, _ = linregress(qsq_win, lnI_win)

            if slope >= 0: continue 

            rg = np.sqrt(-3 * slope)
            
            # Sexier Constraints: 1.1 <= qmax*Rg <= 1.5 AND qmin*Rg <= 0.8
            qmin_rg = q[start] * rg
            qmax_rg = q[end - 1] * rg
            
            if not (0.9 <= qmax_rg <= 1.6 and qmin_rg <= 0.9): 
                 continue

            score = (r_value**8) + (1 - abs(qmax_rg - 1.3))
            
            if score > best_score:
                best_score = score
                best_result = {
                    'rg': rg, 'i0': np.exp(intercept), 'r2': r_value**2,
                    'start_idx': start, 'end_idx': end,
                    'slope': slope, 'intercept': intercept,
                    'qmin': q[start], 'qmax': q[end-1],
                    'qmin_rg': qmin_rg, 'qmax_rg': qmax_rg,
                    'q_sq_data': qsq_win, 'ln_I_data': lnI_win
                }

    return best_result

def _perform_guinier_fit_manual(q, I, start_idx, end_idx):
    """Manual fit used when user specifies line numbers."""
    if start_idx < 0: start_idx = 0
    if end_idx > len(q): end_idx = len(q)
    if start_idx >= end_idx or (end_idx - start_idx) < 3: return None

    q_win = q[start_idx:end_idx]
    I_win = I[start_idx:end_idx]
    if np.any(I_win <= 0): return None

    q_sq = q_win**2
    ln_I = np.log(I_win)
    slope, intercept, r_value, _, _ = linregress(q_sq, ln_I)

    if slope >= 0: return None 

    rg = np.sqrt(-3 * slope)
    i0 = np.exp(intercept)
    
    return {
        'rg': rg, 'i0': i0, 'r2': r_value**2,
        'start_idx': start_idx, 'end_idx': end_idx,
        'slope': slope, 'intercept': intercept,
        'qmin': q[start_idx], 'qmax': q[end_idx-1],
        'qmin_rg': q[start_idx] * rg, 'qmax_rg': q[end_idx-1] * rg,
        'q_sq_data': q_sq, 'ln_I_data': ln_I
    }

def _estimate_molecular_weight(q, I, i0, rg):
    q_limit = 0.3
    mask = q <= q_limit
    if np.sum(mask) < 5: return 0
    
    q_sub = q[mask]
    I_sub = I[mask]
    
    integral = integrate.simpson(I_sub * q_sub, x=q_sub)
    if integral <= 0 or rg <= 0: return 0

    Vc = i0 / integral
    QR = Vc**2 / rg
    mw = QR / 0.1231 
    return mw


###############################################
#
# Main Application Class
#
###############################################

class FileListWidget(QListWidget):
    def __init__(self, main_window_ref=None, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.main_window_ref = main_window_ref

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls(): 
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else: event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
            
            converted_count = 0
            mw = self.main_window_ref
            if not mw:
                curr = self.parent()
                while curr:
                    if isinstance(curr, SAXStingApp):
                        mw = curr
                        break
                    curr = curr.parent()

            if mw:
                for url in event.mimeData().urls():
                    if url.isLocalFile() and url.toLocalFile().endswith('.dat'):
                        was_converted = mw.add_file_to_list(url.toLocalFile())
                        if was_converted:
                            converted_count += 1
                
                if converted_count > 0:
                    QMessageBox.information(self, "Unit Conversion", 
                                          f"{converted_count} file(s) detected in nm⁻¹ and converted to Å⁻¹.")
        else:
            event.ignore()


class SAXStingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAXSting v5.0")
        self.setGeometry(100, 100, 1600, 950)
        
        self.file_map = {} 
        self.results = {} 
        self.view_normalized = False
        
        # Styles
        self.btn_style_normal = "font-weight: bold; background-color: #e0e0e0; color: black; border: 1px solid #999; border-radius: 4px; padding: 6px;"
        self.btn_style_active = "font-weight: bold; background-color: #0078d7; color: white; border: 1px solid #005a9e; border-radius: 4px; padding: 6px;"
        self.btn_style_action = "font-weight: bold; background-color: #28a745; color: white; border-radius: 4px; padding: 6px;"
        self.btn_style_process = "font-weight: bold; font-size: 13px; background-color: #ff4444; color: white; border-radius: 4px; padding: 6px; margin: 5px 0;"
        self.btn_style_processing = "font-weight: bold; font-size: 13px; background-color: #b30000; color: white; border-radius: 4px; padding: 6px; margin: 5px 0;"
        self.btn_style_quit = "font-weight: bold; color: white; background-color: #8b0000; border-radius: 4px; padding: 6px;"

        self.init_plots()
        self.setup_ui()

    def init_plots(self):
        self.fig_saxs = Figure(figsize=(5, 5))
        self.canvas_saxs = FigureCanvasQTAgg(self.fig_saxs)
        self.ax_saxs = self.fig_saxs.add_subplot(111)

        self.fig_guinier = Figure(figsize=(5, 5))
        self.canvas_guinier = FigureCanvasQTAgg(self.fig_guinier)
        gs = self.fig_guinier.add_gridspec(4, 1, hspace=0.05)
        self.ax_guinier = self.fig_guinier.add_subplot(gs[:3, 0])
        self.ax_resid = self.fig_guinier.add_subplot(gs[3, 0], sharex=self.ax_guinier)

        self.fig_kratky = Figure(figsize=(5, 5))
        self.canvas_kratky = FigureCanvasQTAgg(self.fig_kratky)
        self.ax_kratky = self.fig_kratky.add_subplot(111)

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # --- TOP SECTION (Plots + Controls) ---
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.canvas_saxs, stretch=3)

        # --- CONTROL PANEL ---
        control_scroll = QWidget()
        control_layout = QVBoxLayout(control_scroll)
        
        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        control_frame.setLayout(control_layout)
        control_frame.setFixedWidth(380)

        # Title (Fixed Font: Arial)
        title = QLabel("SAXSting v5.0")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(title)
        
        # 1. Load Data
        grp_load = QGroupBox("1. Data Loading")
        lay_load = QVBoxLayout(grp_load)
        btn_browse = QPushButton("Add .dat Files")
        btn_browse.clicked.connect(self._browse_files)
        lay_load.addWidget(btn_browse)
        
        self.list_widget = FileListWidget(main_window_ref=self)
        self.list_widget.itemSelectionChanged.connect(self.on_file_selected)
        self.list_widget.itemChanged.connect(self.on_check_changed)
        lay_load.addWidget(self.list_widget)
        control_layout.addWidget(grp_load)

        # === AUTO PROCESS BUTTON (Sexier Algo) ===
        self.btn_process_all = QPushButton("Auto-Process")
        self.btn_process_all.setToolTip("Runs the Sexier Auto-Rg search algorithm on all files")
        self.btn_process_all.setStyleSheet(self.btn_style_process)
        self.btn_process_all.clicked.connect(self.run_auto_process_sexier)
        control_layout.addWidget(self.btn_process_all)

        # 2. Guinier Settings (Manual Individual)
        grp_guinier = QGroupBox("2. Guinier (Manual Refinement)")
        lay_guinier = QGridLayout(grp_guinier)
        
        self.lbl_selected_file = QLabel("No file selected")
        self.lbl_selected_file.setStyleSheet("color: gray; font-style: italic; font-size: 11px;")
        lay_guinier.addWidget(self.lbl_selected_file, 0, 0, 1, 2)

        self.manual_start_inp = QLineEdit()
        self.manual_end_inp = QLineEdit()
        
        lay_guinier.addWidget(QLabel("qmin index:"), 1, 0)
        lay_guinier.addWidget(self.manual_start_inp, 1, 1)
        lay_guinier.addWidget(QLabel("qmax index:"), 2, 0)
        lay_guinier.addWidget(self.manual_end_inp, 2, 1)
        
        self.btn_refine_selected = QPushButton("Refine Selected")
        self.btn_refine_selected.setToolTip("Update fit for the highlighted file only")
        self.btn_refine_selected.setStyleSheet(self.btn_style_action)
        self.btn_refine_selected.clicked.connect(self.refine_selected_file)
        lay_guinier.addWidget(self.btn_refine_selected, 3, 0, 1, 2)
        
        control_layout.addWidget(grp_guinier)
        
        # 3. Mass Settings
        grp_mass = QGroupBox("3. Molecular Weight")
        lay_mass = QGridLayout(grp_mass)
        self.expected_mw_inp = QLineEdit()
        self.expected_mw_inp.setPlaceholderText("e.g. 66.5")
        lay_mass.addWidget(QLabel("Expected MW (kDa):"), 0, 0)
        lay_mass.addWidget(self.expected_mw_inp, 0, 1)
        
        self.btn_calc_oligo = QPushButton("Calculate State")
        self.btn_calc_oligo.setStyleSheet(self.btn_style_normal)
        self.btn_calc_oligo.clicked.connect(self.update_oligomer_states)
        lay_mass.addWidget(self.btn_calc_oligo, 1, 0, 1, 2)
        
        control_layout.addWidget(grp_mass)

        # 4. View Controls
        grp_tools = QGroupBox("4. View Controls")
        lay_tools = QVBoxLayout(grp_tools)
        
        btn_layout = QHBoxLayout()
        self.btn_raw = QPushButton("Raw")
        self.btn_raw.setCheckable(True)
        self.btn_raw.setChecked(True)
        self.btn_raw.clicked.connect(lambda: self.set_view_mode(False))
        
        self.btn_super = QPushButton("Superimpose")
        self.btn_super.setCheckable(True)
        self.btn_super.clicked.connect(lambda: self.set_view_mode(True))
        
        self.view_group = QButtonGroup(self)
        self.view_group.addButton(self.btn_raw)
        self.view_group.addButton(self.btn_super)
        
        btn_layout.addWidget(self.btn_raw)
        btn_layout.addWidget(self.btn_super)
        lay_tools.addLayout(btn_layout)

        self.chk_log = QCheckBox("Log-Log Plot")
        self.chk_log.clicked.connect(self.update_plots)
        lay_tools.addWidget(self.chk_log)

        # --- RESTORED AVERAGING BUTTON ---
        self.btn_average = QPushButton("Average Checked Files")
        self.btn_average.setStyleSheet(self.btn_style_normal)
        self.btn_average.clicked.connect(self.average_normalized_files)
        lay_tools.addWidget(self.btn_average)
        
        self.btn_save = QPushButton("Save Full Report")
        self.btn_save.clicked.connect(self.save_full_analysis)
        lay_tools.addWidget(self.btn_save)
        
        control_layout.addWidget(grp_tools)
        
        control_layout.addStretch()

        # 5. Quit / Reset
        hbox_utils = QHBoxLayout()
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_all)
        self.btn_quit = QPushButton("Quit")
        self.btn_quit.setStyleSheet(self.btn_style_quit)
        self.btn_quit.clicked.connect(self.close)
        
        hbox_utils.addWidget(self.btn_reset)
        hbox_utils.addWidget(self.btn_quit)
        control_layout.addLayout(hbox_utils)
        
        top_layout.addWidget(control_frame, stretch=0)

        # --- RIGHT TABS ---
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(self.canvas_guinier, "Guinier Fit")
        self.right_tabs.addTab(self.canvas_kratky, "Kratky")
        top_layout.addWidget(self.right_tabs, stretch=3)

        main_layout.addLayout(top_layout, stretch=6)

        # --- TABLE RESULTS ---
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "File", "Rg (Å)", "I(0)", "MW (kDa)", "State", 
            "qmin index", "qmax index", "qmin·Rg", "qmax·Rg"
        ])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        main_layout.addWidget(self.table, stretch=2)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.canvas_saxs.mpl_connect('motion_notify_event', self.update_cursor)
        self.canvas_guinier.mpl_connect('motion_notify_event', self.update_cursor)
        self.canvas_kratky.mpl_connect('motion_notify_event', self.update_cursor)
        
        self.set_view_mode(False)

    def update_cursor(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.status_bar.showMessage(f"Cursor: x={x:.4f}, y={y:.4e}")
        else:
            self.status_bar.clearMessage()

    # --- LOGIC: FILE HANDLING ---

    def add_file_to_list(self, full_path):
        filename = os.path.basename(full_path)
        if filename in self.file_map: return False

        data = _read_dat_file(full_path)
        if data is None: return False

        converted = False
        if np.max(data[:,0]) > 3.0:
            data[:,0] *= 0.1 
            converted = True

        self.file_map[filename] = {'path': full_path, 'data': data}
        
        item = QListWidgetItem(filename)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        item.setData(Qt.UserRole, full_path) 
        self.list_widget.addItem(item)
        
        return converted

    def _browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Dat Files", "", "Data (*.dat)")
        converted_count = 0
        for f in files:
            was_converted = self.add_file_to_list(f)
            if was_converted:
                converted_count += 1
        
        if converted_count > 0:
            QMessageBox.information(self, "Unit Conversion", 
                                  f"{converted_count} file(s) detected in nm⁻¹ and converted to Å⁻¹.")

    def reset_all(self):
        self.list_widget.clear()
        self.file_map.clear()
        self.results.clear()
        self.table.setRowCount(0)
        self.ax_saxs.clear()
        self.ax_guinier.clear()
        self.ax_resid.clear()
        self.ax_kratky.clear()
        self.canvas_saxs.draw()
        self.canvas_guinier.draw()
        self.canvas_kratky.draw()
        self.status_bar.showMessage("Ready")

    # --- LOGIC: PROCESSING ---

    def run_auto_process_sexier(self):
        """Runs the Sexier Auto-Search algorithm on all files."""
        # Visual Update
        self.btn_process_all.setText("Processing...")
        self.btn_process_all.setStyleSheet(self.btn_style_processing)
        self.btn_process_all.setEnabled(False)
        QApplication.processEvents()

        try:
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                filename = item.text()
                data = self.file_map[filename]['data']
                q, I = data[:,0], data[:,1]
                
                res = _sexier_auto_search(q, I)
                
                if res:
                    self._calculate_derived_params(filename, res)
                    self.results[filename] = res
                else:
                    self.results[filename] = {'error': 'Fit failed'}
            
            self.update_table()
            self.update_plots()
        
        finally:
            # Always reset button
            self.btn_process_all.setText("Auto-Process")
            self.btn_process_all.setStyleSheet(self.btn_style_process)
            self.btn_process_all.setEnabled(True)

    def refine_selected_file(self):
        items = self.list_widget.selectedItems()
        if not items: 
            QMessageBox.information(self, "Info", "Please select a file name in the list first.")
            return
        
        filename = items[0].text()
        try:
            s_idx = int(self.manual_start_inp.text())
            e_idx = int(self.manual_end_inp.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Indices must be integers.")
            return

        data = self.file_map[filename]['data']
        q, I = data[:,0], data[:,1]
        
        res = _perform_guinier_fit_manual(q, I, s_idx, e_idx)
        if res:
            self._calculate_derived_params(filename, res)
            self.results[filename] = res
            self.update_table()
            self.update_plots()
        else:
            QMessageBox.warning(self, "Fit Failed", "Invalid range or positive slope.")

    def average_normalized_files(self):
        """Averages the normalized (I/I0) intensity of all checked files."""
        # 1. Gather valid checked files
        valid_indices = []
        for i in range(self.list_widget.count()):
            if self.list_widget.item(i).checkState() == Qt.Checked:
                filename = self.list_widget.item(i).text()
                if filename in self.results and 'i0' in self.results[filename] and self.results[filename]['i0'] > 0:
                    valid_indices.append(i)
        
        if not valid_indices:
            QMessageBox.warning(self, "Error", "No valid checked files found (must be processed with I0).")
            return

        # 2. Master Q-axis (take from first valid file)
        first_item = self.list_widget.item(valid_indices[0])
        first_file = first_item.text()
        master_q = self.file_map[first_file]['data'][:, 0]
        
        sum_I_norm = np.zeros_like(master_q)
        count = 0
        
        # 3. Interpolate and Sum
        for idx in valid_indices:
            filename = self.list_widget.item(idx).text()
            data = self.file_map[filename]['data']
            res = self.results[filename]
            
            q_curr = data[:, 0]
            i_curr = data[:, 1]
            i0 = res['i0']
            
            i_norm = i_curr / i0
            
            # Interpolate onto master_q
            i_interp = np.interp(master_q, q_curr, i_norm, left=0, right=0)
            sum_I_norm += i_interp
            count += 1
            
        avg_I = sum_I_norm / count
        
        # 4. Save
        default_name = "Averaged_Normalized_Curves.dat"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Averaged Data", default_name, "Data Files (*.dat)")
        if file_path:
            try:
                header = f"q(A-1)\tI_norm_avg\t(Averaged from {count} files)"
                np.savetxt(file_path, np.column_stack((master_q, avg_I, np.zeros_like(avg_I))), header=header, fmt='%.6e', delimiter='\t')
                QMessageBox.information(self, "Success", f"Saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save file:\n{e}")

    def update_oligomer_states(self):
        for filename, res in self.results.items():
            if 'mw' in res:
                try:
                    expected_mw = float(self.expected_mw_inp.text()) * 1000 
                    if expected_mw > 0:
                        n = res['mw'] / expected_mw
                        res['state_str'] = f"{n:.1f} ({int(round(n))}-mer)"
                    else:
                        res['state_str'] = "-"
                except:
                    res['state_str'] = "-"
        self.update_table()

    def _calculate_derived_params(self, filename, res):
        data = self.file_map[filename]['data']
        mw = _estimate_molecular_weight(data[:,0], data[:,1], res['i0'], res['rg'])
        res['mw'] = mw
        
        try:
            expected_mw = float(self.expected_mw_inp.text()) * 1000 
            if expected_mw > 0:
                n = mw / expected_mw
                res['state_str'] = f"{n:.1f} ({int(round(n))}-mer)"
            else:
                res['state_str'] = "-"
        except:
            res['state_str'] = "-"

    # --- LOGIC: UI UPDATES ---

    def on_file_selected(self):
        items = self.list_widget.selectedItems()
        if not items:
            self.lbl_selected_file.setText("No file selected")
            self.manual_start_inp.setText("")
            self.manual_end_inp.setText("")
            return

        filename = items[0].text()
        self.lbl_selected_file.setText(f"Editing: {filename}")

        if filename in self.results and 'start_idx' in self.results[filename]:
            res = self.results[filename]
            self.manual_start_inp.setText(str(res['start_idx']))
            self.manual_end_inp.setText(str(res['end_idx']))
        else:
            self.manual_start_inp.setText("10")
            self.manual_end_inp.setText("50")

    def on_check_changed(self, item):
        self.update_plots()

    def set_view_mode(self, normalized):
        self.view_normalized = normalized
        if normalized:
            self.btn_super.setStyleSheet(self.btn_style_active)
            self.btn_raw.setStyleSheet(self.btn_style_normal)
        else:
            self.btn_raw.setStyleSheet(self.btn_style_active)
            self.btn_super.setStyleSheet(self.btn_style_normal)
        self.update_plots()

    def update_table(self):
        self.table.setRowCount(0)
        self.table.setSortingEnabled(False)
        
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            filename = item.text()
            
            if filename not in self.results: continue
            res = self.results[filename]
            
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            self.table.setItem(row, 0, QTableWidgetItem(filename))
            
            if 'error' in res:
                self.table.setItem(row, 1, QTableWidgetItem("Failed"))
                continue

            self.table.setItem(row, 1, QTableWidgetItem(f"{res['rg']:.2f}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{res['i0']:.2e}"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{res['mw']/1000:.1f}")) 
            self.table.setItem(row, 4, QTableWidgetItem(res.get('state_str', '-')))
            
            self.table.setItem(row, 5, QTableWidgetItem(f"{res['start_idx']}"))
            self.table.setItem(row, 6, QTableWidgetItem(f"{res['end_idx']}"))
            
            self.table.setItem(row, 7, QTableWidgetItem(f"{res['qmin_rg']:.2f}"))
            self.table.setItem(row, 8, QTableWidgetItem(f"{res['qmax_rg']:.2f}"))

        self.table.setSortingEnabled(True)

    def update_plots(self):
        self.ax_saxs.clear()
        self.ax_guinier.clear()
        self.ax_resid.clear()
        self.ax_kratky.clear()
        
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        is_log_log = self.chk_log.isChecked()

        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() != Qt.Checked: continue
            
            filename = item.text()
            if filename not in self.file_map: continue
            
            data = self.file_map[filename]['data']
            q, I = data[:,0], data[:,1]
            color = colors[i % len(colors)]
            
            # 1. SAXS Plot
            y_plot = I
            if self.view_normalized and filename in self.results and 'i0' in self.results[filename]:
                y_plot = I / self.results[filename]['i0']
            
            self.ax_saxs.plot(q, y_plot, label=filename, color=color, alpha=0.8)
            
            # 2. Analysis Plots
            if filename in self.results and 'rg' in self.results[filename]:
                res = self.results[filename]
                
                # Guinier
                x_g = res['q_sq_data']
                y_g = res['ln_I_data']
                fit = res['slope'] * x_g + res['intercept']
                
                self.ax_guinier.plot(x_g, y_g, '.', color=color)
                self.ax_guinier.plot(x_g, fit, '-', color=color)
                self.ax_resid.plot(x_g, y_g - fit, 'o-', ms=2, color=color)
                
                # Kratky
                rg_curr = res['rg']
                i0_curr = res['i0']
                qk = q * rg_curr
                ik = (qk**2) * (I / i0_curr)
                
                self.ax_kratky.plot(qk, ik, color=color, label=filename)

        # Formatting
        self.ax_saxs.set_xlabel("q (Å⁻¹)")
        self.ax_saxs.set_ylabel("Intensity")
        self.ax_saxs.set_yscale("log")
        if is_log_log: self.ax_saxs.set_xscale("log")
        else: self.ax_saxs.set_xscale("linear")
        self.ax_saxs.grid(True, alpha=0.3)
        
        # Legend Check
        handles, labels = self.ax_saxs.get_legend_handles_labels()
        if handles and self.list_widget.count() < 10: 
            self.ax_saxs.legend(fontsize='small')

        self.ax_guinier.set_ylabel("ln(I)")
        self.ax_guinier.set_title("Guinier Fits")
        self.ax_resid.set_xlabel("q²")
        self.ax_resid.set_ylabel("Res")
        self.ax_resid.axhline(0, c='k', lw=1)
        
        self.ax_kratky.set_xlabel("q·Rg")
        self.ax_kratky.set_ylabel("(qRg)² I/I0")
        self.ax_kratky.set_title("Dimensionless Kratky")
        self.ax_kratky.axhline(1.1, ls='--', c='gray')
        self.ax_kratky.axvline(1.73, ls='--', c='gray')
        self.ax_kratky.set_xlim(0, 6)
        self.ax_kratky.set_ylim(0, 3.5)

        self.canvas_saxs.draw()
        self.canvas_guinier.draw()
        self.canvas_kratky.draw()

    def save_full_analysis(self):
        parent_dir = QFileDialog.getExistingDirectory(self, "Select Parent Directory for Output")
        if not parent_dir: return
        
        counter = 1
        while True:
            folder_name = f"SAXSting-{counter:02d}"
            out_dir = os.path.join(parent_dir, folder_name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                break
            counter += 1
        
        self.fig_saxs.savefig(os.path.join(out_dir, "01_Comparison_Curves.png"), dpi=300)
        self.fig_saxs.savefig(os.path.join(out_dir, "01_Comparison_Curves.svg"))
        
        self.fig_guinier.savefig(os.path.join(out_dir, "02_Guinier_Fits.png"), dpi=300)
        self.fig_guinier.savefig(os.path.join(out_dir, "02_Guinier_Fits.svg"))
        
        self.fig_kratky.savefig(os.path.join(out_dir, "03_Kratky_Plots.png"), dpi=300)
        self.fig_kratky.savefig(os.path.join(out_dir, "03_Kratky_Plots.svg"))
        
        with open(os.path.join(out_dir, "00_Results_Summary.txt"), 'w', encoding='utf-8') as f:
            headers = [self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())]
            f.write("\t".join(headers) + "\n")
            
            for row in range(self.table.rowCount()):
                data = [self.table.item(row, col).text() if self.table.item(row, col) else "" for col in range(self.table.columnCount())]
                f.write("\t".join(data) + "\n")

        txt_dir = os.path.join(out_dir, "source_data_txt")
        os.makedirs(txt_dir, exist_ok=True)
        
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            filename = item.text()
            if filename not in self.results or 'error' in self.results[filename]: continue
            
            res = self.results[filename]
            data = self.file_map[filename]['data']
            q, I = data[:,0], data[:,1]
            
            base_name = os.path.splitext(filename)[0]
            
            i0 = res['i0']
            curve_data = np.column_stack((q, I, I/i0))
            np.savetxt(os.path.join(txt_dir, f"{base_name}_Curve.txt"), curve_data, 
                       header="q(A-1)\tI(q)\tI(q)/I0", fmt="%.6e", delimiter="\t")
            
            q_sq = res['q_sq_data']
            ln_I_exp = res['ln_I_data']
            ln_I_fit = res['slope'] * q_sq + res['intercept']
            resid = ln_I_exp - ln_I_fit
            guinier_data = np.column_stack((q_sq, ln_I_exp, ln_I_fit, resid))
            np.savetxt(os.path.join(txt_dir, f"{base_name}_Guinier.txt"), guinier_data,
                       header="q^2\tln(I_exp)\tln(I_fit)\tResiduals", fmt="%.6e", delimiter="\t")
            
            rg = res['rg']
            qRg = q * rg
            dimless_I = (qRg**2) * (I/i0)
            kratky_data = np.column_stack((qRg, dimless_I))
            np.savetxt(os.path.join(txt_dir, f"{base_name}_Kratky.txt"), kratky_data,
                       header="q*Rg\t(qRg)^2*I/I0", fmt="%.6e", delimiter="\t")

        QMessageBox.information(self, "Saved", f"Analysis saved to:\n{out_dir}\n\n(Includes SVG plots and .txt source data)")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    p = app.palette()
    p.setColor(QPalette.Window, QColor(245, 245, 245))
    app.setPalette(p)
    
    w = SAXStingApp()
    w.show()
    sys.exit(app.exec())