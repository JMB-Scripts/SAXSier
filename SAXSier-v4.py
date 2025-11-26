import sys
import os
import platform

# --- FIX: Persist Matplotlib Cache ---
# This function sets the Matplotlib cache directory.
def set_mpl_cache_dir():
    # 1. Check if running as a PyInstaller bundle
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Look for the cache we bundled in Step 2
        bundled_cache = os.path.join(sys._MEIPASS, 'pkg_mpl_cache')
        if os.path.exists(bundled_cache):
            # Use the internal cache (Fast, no rebuilding)
            os.environ['MPLCONFIGDIR'] = bundled_cache
            return

    # 2. Fallback for Development Mode (local machine)
    # If we are running straight python, use the user's folder
    app_name = "SAXSier"
    home = os.path.expanduser("~")
    
    if sys.platform.startswith('win'):
        base_dir = os.getenv('APPDATA') or os.path.join(home, 'AppData', 'Roaming')
    elif sys.platform == 'darwin':
        base_dir = os.path.join(home, 'Library', 'Application Support')
    else:
        base_dir = os.path.join(home, '.config')

    mpl_cache_dir = os.path.join(base_dir, app_name, 'mpl_cache')
    
    try:
        os.makedirs(mpl_cache_dir, exist_ok=True)
        os.environ['MPLCONFIGDIR'] = mpl_cache_dir
    except Exception as e:
        print(f"Warning: Could not set custom Matplotlib cache: {e}")

# Run this BEFORE imports
set_mpl_cache_dir()
# ----------------------

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QLabel, QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from ragtime import SAXSAnalysisApp
from sexier import MainWindow as SexierMainWindow
from sub_me import SubMeApp
from saxsting import SAXStingApp

class SAXSLauncher(QMainWindow):
    """
    Main launcher window for the SAXS analysis suite.
    Provides access to Ragtime, Sexier, SAXSting and SubMe tools.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JMB-Scripts: SAXSier")
        self.setGeometry(300, 300, 500, 600) 

        self.opened_windows = []

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel("- SAXSier, Analysis Suite -")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # --- Ragtime Tool ---
        main_layout.addWidget(self.create_tool_section(
            "- Ragtime -",
            "Performs Guinier and MW analysis in deep, on a series of profiles from SEC-SAXS to plot I(0)vsRg and I(0)vsMW.",
            self.launch_ragtime
        ))

        # --- Sexier Tool ---
        main_layout.addWidget(self.create_tool_section(
            "- Sexier -",
            "Performs a detailed single-profile analysis Guinier, Kratky, MW, BIFT.",
            self.launch_sexier
        ))
        # --- SAXSting Tool ---

        main_layout.addWidget(self.create_tool_section(
            "- SAXSting -",
            "Superimposes multiple SAXS curves analyze Rg,I(0), MW and Kratky",
            self.launch_saxting
        ))

        # --- SubMe Tool ---
        main_layout.addWidget(self.create_tool_section(
            "- SubMe -",
            "Performs buffer subtraction for SEC-SAXS using an averaged region or a linear baseline.",
            self.launch_subme
        ))

        main_layout.addStretch()

        # Quit button
        quit_button = QPushButton("Quit Application")
        quit_button.setObjectName("quit_button")
        quit_button.clicked.connect(self.close)
        main_layout.addWidget(quit_button)

        self.setCentralWidget(main_widget)

    def create_tool_section(self, title, description, launch_function):
        """Helper function to create a consistent section for each tool."""
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(frame)

        button = QPushButton(title)
        button.clicked.connect(launch_function)
        
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #444;")

        layout.addWidget(button)
        layout.addWidget(desc_label)
        
        return frame

    def launch_subme(self):
        subme_window = SubMeApp()
        self.opened_windows.append(subme_window)
        subme_window.show()

    def launch_ragtime(self):
        ragtime_window = SAXSAnalysisApp()
        self.opened_windows.append(ragtime_window)
        ragtime_window.show()

    def launch_sexier(self):
        sexier_window = SexierMainWindow()
        self.opened_windows.append(sexier_window)
        sexier_window.show()

    def launch_saxting(self):
        saxting_window = SAXStingApp()
        self.opened_windows.append(saxting_window)
        saxting_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QMainWindow { background-color: #f0f0f0; }
        QPushButton {
            background-color: #0078d7;
            color: white;
            border-radius: 5px;
            padding: 8px;
            font-size: 14px;
        }
        QPushButton:hover { background-color: #005a9e; }
        QPushButton#quit_button { background-color: #dc3545; }
        QPushButton#quit_button:hover { background-color: #c82333; }
        QFrame {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 5px;
        }
        QLabel { font-size: 12px; }
    """)
    launcher = SAXSLauncher()
    launcher.show()
    sys.exit(app.exec())