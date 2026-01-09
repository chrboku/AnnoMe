import sys
import os
from collections import defaultdict, OrderedDict
import io
import json
import traceback
import tomllib
import re
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QComboBox,
    QLineEdit,
    QFileDialog,
    QGroupBox,
    QScrollArea,
    QCheckBox,
    QTextEdit,
    QSplitter,
    QMessageBox,
    QProgressDialog,
    QListWidgetItem,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QSlider,
    QMenu,
    QSpinBox,
    QDialog,
    QDialogButtonBox,
    QTabWidget,
    QAction,
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap, QImage, QColor, QBrush
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
import math
import pandas as pd
import numpy as np
import tempfile
from pprint import pprint
import pickle
import inspect

# Import classification functions
from .Classification import (
    generate_embeddings,
    add_all_metadata,
    train_and_classify,
    predict,
    show_dataset_overview,
    generate_prediction_overview,
    set_random_seeds,
)

# Import MGF parsing
from .Filters import parse_mgf_file

# Global color constants for file type categories
COLOR_TRAIN_RELEVANT = QColor(197, 216, 157)  # Light green
COLOR_TRAIN_OTHER = QColor(210, 83, 83)  # Peach
COLOR_VALIDATION_RELEVANT = QColor(156, 171, 132)  # Light blue
COLOR_VALIDATION_OTHER = QColor(158, 59, 59)  # Light yellow
COLOR_INFERENCE = QColor(69, 104, 130)  # Lavender


class CollapsibleSection(QWidget):
    """A collapsible section widget with a toggle button."""

    toggled = pyqtSignal(object)  # Signal emitted when section is toggled

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.toggle_button = QPushButton(f"▼ {title}")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.clicked.connect(self.toggle)

        self.content_area = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_area.setLayout(self.content_layout)

        layout = QVBoxLayout()
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_area)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def toggle(self):
        checked = self.toggle_button.isChecked()
        self.content_area.setVisible(checked)
        text = self.toggle_button.text()
        self.toggle_button.setText(text.replace("▼" if checked else "▶", "▶" if not checked else "▼"))
        self.toggled.emit(self)

    def collapse(self):
        """Collapse this section without emitting signal."""
        if self.toggle_button.isChecked():
            self.toggle_button.setChecked(False)
            self.content_area.setVisible(False)
            text = self.toggle_button.text()
            self.toggle_button.setText(text.replace("▼", "▶"))

    def expand(self):
        """Expand this section without emitting signal."""
        if not self.toggle_button.isChecked():
            self.toggle_button.setChecked(True)
            self.content_area.setVisible(True)
            text = self.toggle_button.text()
            self.toggle_button.setText(text.replace("▶", "▼"))

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)


class EmbeddingWorker(QThread):
    """Worker thread for generating embeddings."""

    finished = pyqtSignal(object)  # Emits the dataframe
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, datasets, data_to_add, mgf_files):
        super().__init__()
        self.datasets = datasets
        self.data_to_add = data_to_add
        self.mgf_files = mgf_files

    def run(self):
        import sys

        try:
            self.progress.emit("Generating embeddings...")

            df = generate_embeddings(self.datasets, self.data_to_add)

            self.progress.emit("Adding metadata...")

            df = add_all_metadata(self.datasets, df)

            self.progress.emit("Adding all MGF metadata fields...")

            # Add all metadata fields from MGF entries to the dataframe
            # Match by source (filename) and name (feature ID)
            for idx, row in df.iterrows():
                source = row.get("source", "")
                name = row.get("name", "")

                if source in self.mgf_files:
                    entries = self.mgf_files[source]["entries"]
                    # Find matching entry by name
                    for entry in entries:
                        entry_name = entry.get("name", entry.get("feature_id", entry.get("title", "")))
                        if entry_name == name:
                            # Add all fields from entry that aren't already in df
                            for key, value in entry.items():
                                # Skip special keys and keys that already exist
                                if key not in ["$$spectrumdata", "$$spectrumData", "peaks"] and key not in df.columns:
                                    df.at[idx, key] = value
                            break

            self.finished.emit(df)
        except Exception as e:
            error_msg = f"Error generating embeddings: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.error.emit(error_msg)


class TrainingWorker(QThread):
    """Worker thread for training classifiers."""

    finished = pyqtSignal(object, object, object, object, object, object)  # df_train, df_validation, df_inference, df_metrics, trained_classifiers, pivot_tables
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    log = pyqtSignal(str)  # Emits log messages

    def __init__(self, df_subset, subsets, output_dir, classifiers=None, min_prediction_threshold=120):
        super().__init__()
        self.df_subset = df_subset
        self.subsets = subsets
        self.output_dir = output_dir
        self.classifiers = classifiers
        self.min_prediction_threshold = min_prediction_threshold

    def run(self):
        import sys

        try:
            self.progress.emit("Training classifiers and predicting...\nSee console for further details")

            df_train, df_validation, df_inference, df_metrics, trained_classifiers = train_and_classify(self.df_subset, subsets=self.subsets, output_dir=self.output_dir, classifiers=self.classifiers)

            # Generate prediction overviews and capture long and pivot tables
            long_tables = {}
            pivot_tables = {}
            for subset_name in self.subsets:
                xxx = self.df_subset.copy()
                df_subset_infe = predict(xxx, trained_classifiers[subset_name][1], subset_name, trained_classifiers[subset_name][0])
                long_table, pivot_table = generate_prediction_overview(xxx, df_subset_infe, self.output_dir, file_prefix=subset_name, min_prediction_threshold=self.min_prediction_threshold)
                long_tables[subset_name] = long_table
                pivot_tables[subset_name] = pivot_table

            self.finished.emit(df_train, df_validation, df_inference, df_metrics, trained_classifiers, (long_tables, pivot_tables))
        except Exception as e:
            error_msg = f"Error training classifiers: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.error.emit(error_msg)


class SpectrumViewer(QDialog):
    """Dialog for viewing spectrum details."""

    def __init__(self, spectrum_data, meta_data, prediction_data, parent=None):
        super().__init__(parent)
        self.spectrum_data = spectrum_data
        self.meta_data = meta_data
        self.prediction_data = prediction_data

        self.setWindowTitle("Spectrum Viewer")
        self.setGeometry(100, 100, 1200, 800)

        layout = QVBoxLayout()

        # Top section: Metadata and Prediction side by side
        top_widget = QWidget()
        top_layout = QHBoxLayout()

        # Metadata panel
        meta_group = QGroupBox("Metadata")
        meta_layout = QVBoxLayout()
        meta_text = QTextEdit()
        meta_text.setReadOnly(True)
        meta_html = "<table border='1' cellpadding='5' style='width:100%'>"
        for key, value in self.meta_data.items():
            meta_html += f"<tr><td><b>{key}</b></td><td>{value}</td></tr>"
        meta_html += "</table>"
        meta_text.setHtml(meta_html)
        meta_layout.addWidget(meta_text)
        meta_group.setLayout(meta_layout)
        top_layout.addWidget(meta_group)

        # Prediction panel
        pred_group = QGroupBox("Classification Results")
        pred_layout = QVBoxLayout()
        pred_text = QTextEdit()
        pred_text.setReadOnly(True)
        if self.prediction_data:
            pred_html = "<table border='1' cellpadding='5' style='width:100%'>"
            for key, value in self.prediction_data.items():
                pred_html += f"<tr><td><b>{key}</b></td><td>{value}</td></tr>"
            pred_html += "</table>"
            pred_text.setHtml(pred_html)
        else:
            pred_text.setHtml("<p>No classification results available</p>")
        pred_layout.addWidget(pred_text)
        pred_group.setLayout(pred_layout)
        top_layout.addWidget(pred_group)

        top_widget.setLayout(top_layout)
        layout.addWidget(top_widget)

        # Bottom section: Spectrum visualization
        spectrum_group = QGroupBox("Spectrum")
        spectrum_layout = QVBoxLayout()

        try:
            import matplotlib

            matplotlib.use("Qt5Agg")
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure

            # Create matplotlib figure
            fig = Figure(figsize=(10, 4))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            # Plot spectrum
            if self.spectrum_data is not None:
                import numpy as np

                # spectrum_data is a 2D array with [0,:] = m/z and [1,:] = intensity
                if isinstance(self.spectrum_data, (list, tuple)) and len(self.spectrum_data) == 2:
                    mz_values = self.spectrum_data[0]
                    intensity_values = self.spectrum_data[1]
                elif hasattr(self.spectrum_data, "shape") and len(self.spectrum_data.shape) == 2:
                    # numpy array
                    mz_values = self.spectrum_data[0, :]
                    intensity_values = self.spectrum_data[1, :]
                else:
                    # Fallback: assume it's a list of [mz, intensity] pairs
                    mz_values = [peak[0] for peak in self.spectrum_data]
                    intensity_values = [peak[1] for peak in self.spectrum_data]

                ax.vlines(mz_values, 0, intensity_values, colors="blue", linewidth=1.5)
                ax.set_xlabel("m/z", fontsize=12)
                ax.set_ylabel("Intensity", fontsize=12)
                ax.set_title("MS/MS Spectrum", fontsize=14)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
            else:
                ax.text(0.5, 0.5, "No spectrum data available", ha="center", va="center", fontsize=14)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

            spectrum_layout.addWidget(canvas)
        except ImportError:
            spectrum_label = QLabel("Matplotlib not available for spectrum visualization")
            spectrum_label.setAlignment(Qt.AlignCenter)
            spectrum_layout.addWidget(spectrum_label)
        except Exception as e:
            spectrum_label = QLabel(f"Error creating spectrum plot: {str(e)}")
            spectrum_label.setAlignment(Qt.AlignCenter)
            spectrum_layout.addWidget(spectrum_label)

        spectrum_group.setLayout(spectrum_layout)
        layout.addWidget(spectrum_group)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self.setLayout(layout)


class ClassificationGUI(QMainWindow):
    """Main GUI for Flavonoid Compound Classification."""

    def __init__(self):
        super().__init__()
        self.mgf_files = {}  # filename -> {data, entries, unique_smiles, type}
        self.all_meta_keys = set()
        self.subsets = []  # List of subset dictionaries
        self.df_embeddings = None
        self.df_subset = None
        self.df_train = None
        self.df_validation = None
        self.df_inference = None
        self.df_metrics = None
        self.trained_classifiers = None
        self.subset_results = {}  # subset_name -> {df_train, df_validation, df_inference, df_metrics, trained_classifiers}
        self.sections = []
        self.classifiers_config = None  # ML classifiers configuration
        self.min_prediction_threshold = 120  # Default min prediction threshold
        self.pending_config_data = None  # For load_full_configuration workflow

        set_random_seeds(42)

        self.init_ui()

    def get_version(self):
        """Read version from pyproject.toml"""
        try:
            # Get the path to pyproject.toml (go up from src/AnnoMe to project root)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            toml_path = os.path.join(project_root, "pyproject.toml")

            if os.path.exists(toml_path):
                with open(toml_path, "rb") as f:
                    data = tomllib.load(f)
                    return data.get("project", {}).get("version", "unknown")
        except Exception:
            pass
        return "unknown"

    def init_ui(self):
        version = self.get_version()
        self.setWindowTitle(f"Flavonoid Compound Classification GUI (v{version})")
        self.setGeometry(50, 50, 1600, 900)

        # Create menu bar
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        load_config_action = QAction("Load Configuration", self)
        load_config_action.triggered.connect(self.load_full_configuration)
        file_menu.addAction(load_config_action)

        save_config_action = QAction("Save Configuration", self)
        save_config_action.triggered.connect(self.save_full_configuration)
        file_menu.addAction(save_config_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll.setWidget(scroll_content)

        layout = QVBoxLayout()
        scroll_content.setLayout(layout)

        # Section 1: Load MGF Files
        self.section1 = CollapsibleSection("1. Load MGF Files")
        self.section1.toggled.connect(self.on_section_toggled)
        self.init_section1()
        layout.addWidget(self.section1)
        self.sections.append(self.section1)

        # Section 2: Generate Embeddings
        self.section2 = CollapsibleSection("2. Generate Embeddings")
        self.section2.toggled.connect(self.on_section_toggled)
        self.section2.collapse()
        self.init_section2()
        layout.addWidget(self.section2)
        self.sections.append(self.section2)

        # Section 3: Define Metadata Subsets
        self.section3 = CollapsibleSection("3. Define Metadata Subsets")
        self.section3.toggled.connect(self.on_section_toggled)
        self.section3.collapse()
        self.init_section3()
        layout.addWidget(self.section3)
        self.sections.append(self.section3)

        # Section 4: Train Classifiers
        self.section4 = CollapsibleSection("4. Train Classifiers")
        self.section4.toggled.connect(self.on_section_toggled)
        self.section4.collapse()
        self.init_section4()
        layout.addWidget(self.section4)
        self.sections.append(self.section4)

        # Section 5: Inspect Classification Results
        self.section5 = CollapsibleSection("5. Inspect Classification Results")
        self.section5.toggled.connect(self.on_section_toggled)
        self.section5.collapse()
        self.init_section5()
        layout.addWidget(self.section5)
        self.sections.append(self.section5)

        # Section 6: Inspect Individual Spectra
        self.section6 = CollapsibleSection("6. Inspect Individual Spectra")
        self.section6.toggled.connect(self.on_section_toggled)
        self.section6.collapse()
        self.init_section6()
        layout.addWidget(self.section6)
        self.sections.append(self.section6)

        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        main_widget.setLayout(main_layout)

    def on_section_toggled(self, toggled_section):
        """Handle section toggle to ensure only one section is open at a time."""
        if toggled_section.toggle_button.isChecked():
            # Collapse all other sections
            for section in self.sections:
                if section != toggled_section:
                    section.collapse()

    def init_section1(self):
        """Initialize the MGF file loading section."""
        content = QWidget()
        main_layout = QHBoxLayout()

        # Help text on the left (25%)
        help_group = QGroupBox("Help")
        help_layout = QVBoxLayout()
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
            <h3>Load MGF Files</h3>
            <p>Load MSMS spectra from MGF files and assign dataset types for classification.</p>
            <h4>File Types:</h4>
            <ul>
                <li><b>train - relevant:</b> Training data with compounds of interest</li>
                <li><b>train - other:</b> Training data with other compounds</li>
                <li><b>validation - relevant:</b> Validation data with relevant compounds</li>
                <li><b>validation - other:</b> Validation data with other compounds</li>
                <li><b>inference:</b> Unknown data for prediction</li>
            </ul>
            <h4>Steps:</h4>
            <ol>
                <li>Click <b>Load MGF File(s)</b> to select one or more MGF files</li>
                <li>Review statistics (entries, unique SMILES)</li>
                <li>Assign appropriate type to each file using dropdown</li>
                <li>Remove unwanted files with <b>Remove Selected File(s)</b></li>
            </ol>
            <p><b>Tip:</b> You need both training data (train-relevant and train-other) and validation data for proper classifier training.</p>
        """)
        help_layout.addWidget(help_text)
        help_group.setLayout(help_layout)
        help_group.setMaximumWidth(400)
        main_layout.addWidget(help_group, 1)  # 25% width

        # Controls on the right (75%)
        controls = QWidget()
        layout = QVBoxLayout()

        # File table
        layout.addWidget(QLabel("Loaded MGF Files:"))
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(4)
        self.file_table.setHorizontalHeaderLabels(["File Name", "# Entries", "# Unique SMILES", "Type"])
        self.file_table.horizontalHeader().setStretchLastSection(False)
        self.file_table.setSelectionBehavior(QAbstractItemView.SelectRows)

        header = self.file_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.Interactive)
        header.setSectionResizeMode(3, QHeaderView.Interactive)

        layout.addWidget(self.file_table, 1)  # Stretch to fill available space

        # Buttons below the table
        button_layout = QHBoxLayout()
        load_btn = QPushButton("Load MGF File(s)")
        load_btn.clicked.connect(self.load_mgf_files)
        button_layout.addWidget(load_btn)

        remove_files_btn = QPushButton("Remove Selected File(s)")
        remove_files_btn.clicked.connect(self.remove_selected_files)
        button_layout.addWidget(remove_files_btn)

        button_layout.addStretch()

        export_config_btn = QPushButton("Export Configuration")
        export_config_btn.clicked.connect(self.export_file_configuration)
        button_layout.addWidget(export_config_btn)

        import_config_btn = QPushButton("Import Configuration")
        import_config_btn.clicked.connect(self.import_file_configuration)
        button_layout.addWidget(import_config_btn)

        layout.addLayout(button_layout)

        controls.setLayout(layout)
        main_layout.addWidget(controls, 3)  # 75% width

        content.setLayout(main_layout)
        self.section1.add_widget(content)

    def init_section2(self):
        """Initialize the embeddings generation section."""
        content = QWidget()
        main_layout = QHBoxLayout()

        # Help text on the left (25%)
        help_group = QGroupBox("Help")
        help_layout = QVBoxLayout()
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
            <h3>Generate Embeddings</h3>
            <p>Convert MSMS spectra to numerical vectors using the MS2DeepScore deep learning model.</p>
            <h4>What are Embeddings?</h4>
            <p>Embeddings are numerical representations of spectra that capture their chemical similarity. 
            They enable machine learning classifiers to learn patterns and make predictions.</p>
            <h4>Process:</h4>
            <ol>
                <li>Loads all MGF files and their spectra</li>
                <li>Uses MS2DeepScore model to generate embeddings</li>
                <li>Attaches all metadata from MGF files to each embedding</li>
                <li>Saves results to pickle file for reuse</li>
            </ol>
            <h4>Performance:</h4>
            <p>Generation may take several minutes for large datasets. Progress will be displayed.</p>
            <p><b>Note:</b> Once generated, embeddings are saved and can be reused without regeneration.</p>
        """)
        help_layout.addWidget(help_text)
        help_group.setLayout(help_layout)
        help_group.setMaximumWidth(400)
        main_layout.addWidget(help_group, 1)  # 25% width

        # Controls on the right (75%)
        controls = QWidget()
        layout = QVBoxLayout()

        # Output directory selection
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Directory:"))
        self.output_dir_input = QLineEdit("./output/classification_results/")
        output_layout.addWidget(self.output_dir_input)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(browse_btn)
        layout.addLayout(output_layout)

        # Generate embeddings button
        self.generate_embeddings_btn = QPushButton("Generate Embeddings")
        self.generate_embeddings_btn.clicked.connect(self.generate_embeddings_clicked)
        layout.addWidget(self.generate_embeddings_btn)

        # Status label
        self.embedding_status = QLabel("Status: Not started")
        layout.addWidget(self.embedding_status)

        layout.addStretch()

        controls.setLayout(layout)
        main_layout.addWidget(controls, 3)  # 75% width

        content.setLayout(main_layout)
        self.section2.add_widget(content)

    def init_section3(self):
        """Initialize the metadata subset definition section."""
        content = QWidget()
        main_layout = QHBoxLayout()

        # Help text on the left (25%)
        help_group = QGroupBox("Help")
        help_layout = QVBoxLayout()
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
            <h3>Define Metadata Subsets</h3>
            <p>Create custom subsets based on spectrum metadata to filter your data.</p>
            <p><b>Note:</b> Generate embeddings first (Section 2) to access metadata fields.</p>
            <h4>Metadata Overview:</h4>
            <ul>
                <li>Click <b>Refresh Metadata Overview</b> to populate the table</li>
                <li>Rows show metadata keys, columns show MGF files</li>
                <li>Numbers indicate unique values for each key/file combination</li>
                <li>Click any cell to view the actual values and their frequencies</li>
            </ul>
            <h4>Custom Subsets:</h4>
            <p>Define subsets using Python syntax. All metadata is in the <code>meta</code> dictionary.</p>
            <p><b>Examples:</b></p>
            <code>meta.get('ionmode') == 'positive'</code><br>
            <code>abs(float(meta.get('CE', 0)) - 20) < 1</code><br>
            <code>meta.get('fragmentation_method') == 'hcd' and meta.get('ionmode') == 'positive'</code>
            <h4>Workflow:</h4>
            <ol>
                <li>Enter subset expression</li>
                <li>Click <b>Check Syntax</b> to validate</li>
                <li>Click <b>Add Subset</b> to apply</li>
                <li>View match counts across file types</li>
                <li>Edit or delete subsets as needed</li>
            </ol>
        """)
        help_layout.addWidget(help_text)
        help_group.setLayout(help_layout)
        help_group.setMaximumWidth(400)
        main_layout.addWidget(help_group, 1)  # 25% width

        # Controls on the right (75%)
        controls = QWidget()
        layout = QVBoxLayout()

        # Refresh button
        refresh_btn = QPushButton("Refresh Metadata Overview")
        refresh_btn.clicked.connect(self.refresh_metadata_overview)
        layout.addWidget(refresh_btn)

        # Splitter for meta overview and value list
        splitter = QSplitter(Qt.Horizontal)

        # Meta keys table
        meta_widget = QWidget()
        meta_layout = QVBoxLayout()
        meta_layout.addWidget(QLabel("Metadata Overview (rows: meta-keys, columns: MGF files):"))
        self.meta_table = QTableWidget()
        self.meta_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.meta_table.itemSelectionChanged.connect(self.on_meta_cell_selected)
        # Connect vertical header click to show all values across all samples
        self.meta_table.verticalHeader().sectionClicked.connect(self.on_meta_key_clicked)
        meta_layout.addWidget(self.meta_table)
        meta_widget.setLayout(meta_layout)
        splitter.addWidget(meta_widget)

        # Value list
        value_widget = QWidget()
        value_layout = QVBoxLayout()
        self.value_list_label = QLabel("Unique Values for Selected Meta-Key:")
        value_layout.addWidget(self.value_list_label)
        self.value_list = QListWidget()
        value_layout.addWidget(self.value_list)
        value_widget.setLayout(value_layout)
        splitter.addWidget(value_widget)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)  # Stretch to fill available space

        # Subset definition
        subset_group = QGroupBox("Define Custom Subset")
        subset_layout = QVBoxLayout()

        subset_help = QLabel("Define subsets using Python syntax. All meta-values are in the 'meta' dictionary.\nExample: abs(float(meta.get('CE', 0))-20) < 1 and meta.get('ionmode') == 'pos'")
        subset_help.setWordWrap(True)
        subset_layout.addWidget(subset_help)

        # Subset name input
        subset_name_layout = QHBoxLayout()
        subset_name_layout.addWidget(QLabel("Subset Name:"))
        self.subset_name_input = QLineEdit()
        self.subset_name_input.setPlaceholderText("e.g., 'Positive Mode CE=20'")
        subset_name_layout.addWidget(self.subset_name_input)
        subset_layout.addLayout(subset_name_layout)

        # Subset expression input
        subset_input_layout = QHBoxLayout()
        subset_input_layout.addWidget(QLabel("Subset Expression:"))
        self.subset_input = QLineEdit()
        subset_input_layout.addWidget(self.subset_input)

        check_btn = QPushButton("Check Syntax")
        check_btn.clicked.connect(self.check_subset_syntax)
        subset_input_layout.addWidget(check_btn)

        add_subset_btn = QPushButton("Add Subset")
        add_subset_btn.clicked.connect(self.add_subset)
        subset_input_layout.addWidget(add_subset_btn)

        subset_layout.addLayout(subset_input_layout)
        subset_group.setLayout(subset_layout)
        layout.addWidget(subset_group)

        # Subset list table
        layout.addWidget(QLabel("Active Subsets:"))
        self.subset_table = QTableWidget()
        self.subset_table.setColumnCount(7)
        self.subset_table.setHorizontalHeaderLabels(["Name", "Subset Expression", "validation-relevant", "validation-other", "inference", "train-relevant", "train-other"])
        header = self.subset_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        for i in range(2, 7):
            header.setSectionResizeMode(i, QHeaderView.Interactive)
        layout.addWidget(self.subset_table)

        # Subset management buttons
        subset_mgmt_layout = QHBoxLayout()
        edit_subset_btn = QPushButton("Edit Selected Subset")
        edit_subset_btn.clicked.connect(self.edit_selected_subset)
        subset_mgmt_layout.addWidget(edit_subset_btn)

        delete_subset_btn = QPushButton("Delete Selected Subset")
        delete_subset_btn.clicked.connect(self.delete_selected_subset)
        subset_mgmt_layout.addWidget(delete_subset_btn)

        subset_mgmt_layout.addStretch()

        export_subsets_btn = QPushButton("Export Subsets")
        export_subsets_btn.clicked.connect(self.export_subsets)
        subset_mgmt_layout.addWidget(export_subsets_btn)

        import_subsets_btn = QPushButton("Import Subsets")
        import_subsets_btn.clicked.connect(self.import_subsets)
        subset_mgmt_layout.addWidget(import_subsets_btn)

        layout.addLayout(subset_mgmt_layout)

        controls.setLayout(layout)
        main_layout.addWidget(controls, 3)  # 75% width

        content.setLayout(main_layout)
        self.section3.add_widget(content)

    def init_section4(self):
        """Initialize the classifier training section."""
        content = QWidget()
        main_layout = QHBoxLayout()

        # Help text on the left (25%)
        help_group = QGroupBox("Help")
        help_layout = QVBoxLayout()
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
            <h3>Train Classifiers</h3>
            <p>Train machine learning classifiers on the generated embeddings with optional metadata subsets.</p>
            <h4>Prerequisites:</h4>
            <ul>
                <li>Embeddings must be generated (Section 2)</li>
                <li>Optionally define metadata subsets (Section 3)</li>
            </ul>
            <h4>Configuration:</h4>
            <ol>
                <li><b>Classifier Configuration:</b> Define which ML algorithms to use (or use defaults)</li>
                <li><b>Min Prediction Threshold:</b> Minimum votes needed for relevant classification</li>
            </ol>
            <h4>Training Process:</h4>
            <ul>
                <li>Splits data into training/validation sets</li>
                <li>Trains multiple classifier algorithms</li>
                <li>Generates predictions and performance metrics</li>
                <li>Saves trained models and results</li>
            </ul>
            <p><b>Note:</b> Training can take time depending on dataset size and number of classifiers.</p>
        """)
        help_layout.addWidget(help_text)
        help_group.setLayout(help_layout)
        help_group.setMaximumWidth(400)
        main_layout.addWidget(help_group, 1)  # 25% width

        # Controls on the right (75%)
        controls = QWidget()
        layout = QVBoxLayout()

        # ML Classifiers Configuration header
        layout.addWidget(QLabel("\nClassifier Configuration (Python dict):"))

        # Load Default button on the right side above text input
        load_default_layout = QHBoxLayout()
        load_default_layout.addStretch()
        load_default_btn = QPushButton("Load Default")
        load_default_btn.setMaximumWidth(120)
        load_default_btn.clicked.connect(self.load_default_classifiers_config)
        load_default_btn.setEnabled(False)  # Disabled until embeddings are generated
        self.load_default_classifiers_btn = load_default_btn
        load_default_layout.addWidget(load_default_btn)
        layout.addLayout(load_default_layout)

        self.classifiers_config_text = QTextEdit()
        self.classifiers_config_text.setPlaceholderText(
            "Leave empty to use default configuration, or enter Python dict code here. The variable 'classifiers' as a dictionary of sklearn classifiers must be generated."
        )
        self.classifiers_config_text.setMinimumHeight(150)
        font = self.classifiers_config_text.font()
        font.setFamily("Courier New")
        self.classifiers_config_text.setFont(font)
        self.classifiers_config_text.setEnabled(False)  # Disabled until embeddings are generated
        layout.addWidget(self.classifiers_config_text, 1)  # Stretch to fill available space

        # Min Prediction Threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Min Prediction Threshold:"))
        self.min_prediction_threshold_input = QSpinBox()
        self.min_prediction_threshold_input.setMinimum(1)
        self.min_prediction_threshold_input.setMaximum(1000)
        self.min_prediction_threshold_input.setValue(120)
        self.min_prediction_threshold_input.setToolTip("Minimum number of classifier votes needed to classify a compound as relevant")
        self.min_prediction_threshold_input.setEnabled(False)  # Disabled until embeddings are generated
        threshold_layout.addWidget(self.min_prediction_threshold_input)
        threshold_layout.addStretch()
        layout.addLayout(threshold_layout)

        # Save/Load Configuration buttons
        config_buttons_layout = QHBoxLayout()
        config_buttons_layout.addStretch()
        save_config_btn = QPushButton("Save Configuration")
        save_config_btn.setMaximumWidth(150)
        save_config_btn.clicked.connect(self.save_classifier_configuration)
        config_buttons_layout.addWidget(save_config_btn)

        load_config_btn = QPushButton("Load Configuration")
        load_config_btn.setMaximumWidth(150)
        load_config_btn.clicked.connect(self.load_classifier_configuration)
        config_buttons_layout.addWidget(load_config_btn)
        layout.addLayout(config_buttons_layout)

        # Train button
        self.train_btn = QPushButton("Train and Classify")
        self.train_btn.clicked.connect(self.train_classifiers_clicked)
        self.train_btn.setEnabled(False)
        layout.addWidget(self.train_btn)

        # Training status
        self.training_status = QLabel("Status: Not started")
        layout.addWidget(self.training_status)

        layout.addStretch()

        controls.setLayout(layout)
        main_layout.addWidget(controls, 3)  # 75% width

        content.setLayout(main_layout)
        self.section4.add_widget(content)

    def init_section5(self):
        """Initialize the results inspection section."""
        content = QWidget()
        main_layout = QHBoxLayout()

        # Subset list on the left
        subset_widget = QWidget()
        subset_layout = QVBoxLayout()
        subset_layout.addWidget(QLabel("<b>Trained Subsets:</b>"))
        self.subset_results_list = QListWidget()
        self.subset_results_list.itemClicked.connect(self.on_subset_result_selected)
        subset_layout.addWidget(self.subset_results_list)
        subset_widget.setLayout(subset_layout)
        subset_widget.setMaximumWidth(250)
        main_layout.addWidget(subset_widget)

        # Results display on the right
        results_widget = QWidget()
        results_layout = QVBoxLayout()

        # Export buttons
        button_layout = QHBoxLayout()
        export_results_btn = QPushButton("Export Results to Excel")
        export_results_btn.clicked.connect(self.export_results)
        button_layout.addWidget(export_results_btn)

        export_classifiers_btn = QPushButton("Export Classifiers")
        export_classifiers_btn.clicked.connect(self.export_classifiers)
        button_layout.addWidget(export_classifiers_btn)
        button_layout.addStretch()
        results_layout.addLayout(button_layout)

        # Results table
        self.current_subset_label = QLabel("Select a subset to view results")
        results_layout.addWidget(self.current_subset_label)
        self.results_table = QTableWidget()
        self.results_table.setSortingEnabled(True)
        results_layout.addWidget(self.results_table, 1)  # Stretch to fill available space

        results_widget.setLayout(results_layout)
        main_layout.addWidget(results_widget)

        content.setLayout(main_layout)
        self.section5.add_widget(content)

    def init_section6(self):
        """Initialize the individual spectra inspection section."""
        content = QWidget()
        main_layout = QHBoxLayout()

        # Help text on the left (25%)
        help_group = QGroupBox("Help")
        help_layout = QVBoxLayout()
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
            <h3>Inspect Individual Spectra</h3>
            <p>Browse and examine individual MSMS spectra with their metadata and predictions.</p>
            <h4>Navigation:</h4>
            <ol>
                <li>Select an MGF file from the left list</li>
                <li>Browse spectra in that file from the right list</li>
                <li>Double-click or click <b>View Details</b> to open viewer</li>
            </ol>
            <h4>Spectrum Viewer:</h4>
            <ul>
                <li><b>Metadata Tab:</b> All spectrum metadata fields</li>
                <li><b>Prediction Results:</b> Classification predictions (if available)</li>
                <li><b>Spectrum Tab:</b> Visualization (future feature)</li>
            </ul>
            <h4>Export:</h4>
            <p>Export individual spectrum data to Excel for external analysis.</p>
            <p><b>Tip:</b> Use this to investigate misclassified spectra and understand why the classifier made certain predictions.</p>
        """)
        help_layout.addWidget(help_text)
        help_group.setLayout(help_layout)
        help_group.setMaximumWidth(400)
        main_layout.addWidget(help_group, 1)  # 25% width

        # Controls on the right (75%)
        controls = QWidget()
        layout = QVBoxLayout()

        # Splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left: MGF file list
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Select MGF File:"))
        self.spectrum_file_list = QListWidget()
        self.spectrum_file_list.itemSelectionChanged.connect(self.on_spectrum_file_selected)
        left_layout.addWidget(self.spectrum_file_list)
        left_widget.setLayout(left_layout)
        splitter.addWidget(left_widget)

        # Right: Spectrum list
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Select Spectrum:"))
        self.spectrum_list = QListWidget()
        self.spectrum_list.itemDoubleClicked.connect(self.view_spectrum_details)
        right_layout.addWidget(self.spectrum_list)

        view_btn = QPushButton("View Details")
        view_btn.clicked.connect(self.view_spectrum_details)
        right_layout.addWidget(view_btn)

        right_widget.setLayout(right_layout)
        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)  # Stretch to fill available space

        # Export button
        export_individual_btn = QPushButton("Export Selected Spectrum to Excel")
        export_individual_btn.clicked.connect(self.export_individual_spectrum)
        layout.addWidget(export_individual_btn)

        controls.setLayout(layout)
        main_layout.addWidget(controls, 3)  # 75% width

        content.setLayout(main_layout)
        self.section6.add_widget(content)

    # Section 1 methods
    def load_mgf_files(self):
        """Load MGF files."""
        files, _ = QFileDialog.getOpenFileNames(self, "Select MGF Files", "", "MGF Files (*.mgf);;All Files (*)")

        if not files:
            return

        progress = QProgressDialog("Loading MGF files...", "Cancel", 0, len(files), self)
        progress.setWindowModality(Qt.WindowModal)

        for i, file_path in enumerate(files):
            if progress.wasCanceled():
                break

            progress.setValue(i)
            progress.setLabelText(f"Loading {os.path.basename(file_path)}...")
            QApplication.processEvents()

            try:
                # Parse MGF file
                entries = parse_mgf_file(file_path, check_required_keys=False)

                # Count unique SMILES
                smiles_set = set()
                for entry in entries:
                    smiles = entry.get("smiles", "")
                    if smiles:
                        smiles_set.add(smiles)

                # Collect all meta keys (excluding special keys)
                for entry in entries:
                    # Filter out special keys that aren't metadata
                    meta_keys = {k for k in entry.keys() if k not in ["$$spectrumdata", "peaks"]}
                    self.all_meta_keys.update(meta_keys)

                filename = os.path.basename(file_path)
                self.mgf_files[filename] = {
                    "path": file_path,
                    "entries": entries,
                    "num_entries": len(entries),
                    "unique_smiles": len(smiles_set),
                    "type": "inference",  # Default type
                }

            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load {file_path}:\n{str(e)}")

        progress.setValue(len(files))
        self.update_file_table()
        QMessageBox.information(self, "Success", f"Loaded {len(files)} MGF file(s)")

    def update_file_table(self):
        """Update the file table with loaded MGF files."""
        self.file_table.setRowCount(len(self.mgf_files))

        for row, (filename, data) in enumerate(self.mgf_files.items()):
            # Filename
            self.file_table.setItem(row, 0, QTableWidgetItem(filename))

            # Number of entries
            self.file_table.setItem(row, 1, QTableWidgetItem(str(data["num_entries"])))

            # Unique SMILES
            self.file_table.setItem(row, 2, QTableWidgetItem(str(data["unique_smiles"])))

            # Type dropdown with color coding
            type_combo = QComboBox()
            type_combo.addItems(["validation - relevant", "validation - other", "inference", "train - relevant", "train - other"])
            type_combo.setCurrentText(data["type"])
            type_combo.currentTextChanged.connect(lambda text, fn=filename: self.on_file_type_changed(fn, text))

            # Set background color based on type
            file_type = data["type"]
            if file_type == "train - relevant":
                type_combo.setStyleSheet(f"background-color: rgb({COLOR_TRAIN_RELEVANT.red()}, {COLOR_TRAIN_RELEVANT.green()}, {COLOR_TRAIN_RELEVANT.blue()});")
            elif file_type == "train - other":
                type_combo.setStyleSheet(f"background-color: rgb({COLOR_TRAIN_OTHER.red()}, {COLOR_TRAIN_OTHER.green()}, {COLOR_TRAIN_OTHER.blue()});")
            elif file_type == "validation - relevant":
                type_combo.setStyleSheet(f"background-color: rgb({COLOR_VALIDATION_RELEVANT.red()}, {COLOR_VALIDATION_RELEVANT.green()}, {COLOR_VALIDATION_RELEVANT.blue()});")
            elif file_type == "validation - other":
                type_combo.setStyleSheet(f"background-color: rgb({COLOR_VALIDATION_OTHER.red()}, {COLOR_VALIDATION_OTHER.green()}, {COLOR_VALIDATION_OTHER.blue()});")
            elif file_type == "inference":
                type_combo.setStyleSheet(f"background-color: rgb({COLOR_INFERENCE.red()}, {COLOR_INFERENCE.green()}, {COLOR_INFERENCE.blue()});")

            self.file_table.setCellWidget(row, 3, type_combo)

        # Auto-resize columns to contents
        self.file_table.resizeColumnsToContents()

    def on_file_type_changed(self, filename, new_type):
        """Handle file type change."""
        if filename in self.mgf_files:
            self.mgf_files[filename]["type"] = new_type
            # Update the combo box color
            for row in range(self.file_table.rowCount()):
                if self.file_table.item(row, 0).text() == filename:
                    combo = self.file_table.cellWidget(row, 3)
                    if combo:
                        if new_type == "train - relevant":
                            combo.setStyleSheet(f"background-color: rgb({COLOR_TRAIN_RELEVANT.red()}, {COLOR_TRAIN_RELEVANT.green()}, {COLOR_TRAIN_RELEVANT.blue()});")
                        elif new_type == "train - other":
                            combo.setStyleSheet(f"background-color: rgb({COLOR_TRAIN_OTHER.red()}, {COLOR_TRAIN_OTHER.green()}, {COLOR_TRAIN_OTHER.blue()});")
                        elif new_type == "validation - relevant":
                            combo.setStyleSheet(f"background-color: rgb({COLOR_VALIDATION_RELEVANT.red()}, {COLOR_VALIDATION_RELEVANT.green()}, {COLOR_VALIDATION_RELEVANT.blue()});")
                        elif new_type == "validation - other":
                            combo.setStyleSheet(f"background-color: rgb({COLOR_VALIDATION_OTHER.red()}, {COLOR_VALIDATION_OTHER.green()}, {COLOR_VALIDATION_OTHER.blue()});")
                        elif new_type == "inference":
                            combo.setStyleSheet(f"background-color: rgb({COLOR_INFERENCE.red()}, {COLOR_INFERENCE.green()}, {COLOR_INFERENCE.blue()});")
                    break

    def remove_selected_files(self):
        """Remove selected MGF files."""
        selected_rows = set(item.row() for item in self.file_table.selectedItems())

        if not selected_rows:
            QMessageBox.warning(self, "Warning", "Please select file(s) to remove")
            return

        # Get filenames to remove
        filenames_to_remove = []
        for row in selected_rows:
            filename_item = self.file_table.item(row, 0)
            if filename_item:
                filenames_to_remove.append(filename_item.text())

        # Confirm removal
        if len(filenames_to_remove) == 1:
            msg = f"Are you sure you want to remove '{filenames_to_remove[0]}'?"
        else:
            msg = f"Are you sure you want to remove {len(filenames_to_remove)} files?"

        reply = QMessageBox.question(self, "Confirm Removal", msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            # Remove from dictionary and update meta keys
            for filename in filenames_to_remove:
                if filename in self.mgf_files:
                    del self.mgf_files[filename]

            # Recollect all meta keys from remaining files
            self.all_meta_keys = set()
            for data in self.mgf_files.values():
                for entry in data["entries"]:
                    meta_keys = {k for k in entry.keys() if k not in ["$$spectrumdata", "peaks"]}
                    self.all_meta_keys.update(meta_keys)

            # Update table
            self.update_file_table()

            QMessageBox.information(self, "Success", f"Removed {len(filenames_to_remove)} file(s)")

    def export_file_configuration(self):
        """Export MGF file configuration to JSON."""
        if not self.mgf_files:
            QMessageBox.warning(self, "Warning", "No MGF files loaded to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Export File Configuration", "file_configuration.json", "JSON Files (*.json);;All Files (*)")

        if not file_path:
            return

        try:
            # Create export data (only path and type, not the full entries)
            export_data = {"files": [{"filename": filename, "path": data["path"], "type": data["type"]} for filename, data in self.mgf_files.items()]}

            with open(file_path, "w") as f:
                json.dump(export_data, f, indent=2)

            QMessageBox.information(self, "Success", f"Configuration exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export configuration:\n{str(e)}")

    def import_file_configuration(self):
        """Import MGF file configuration from JSON."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Import File Configuration", "", "JSON Files (*.json);;All Files (*)")

        if not file_path:
            return

        try:
            with open(file_path, "r") as f:
                import_data = json.load(f)

            if "files" not in import_data:
                QMessageBox.warning(self, "Error", "Invalid configuration file format")
                return

            # Load the MGF files
            progress = QProgressDialog("Loading MGF files...", "Cancel", 0, len(import_data["files"]), self)
            progress.setWindowModality(Qt.WindowModal)

            loaded_count = 0
            failed_files = []

            for i, file_info in enumerate(import_data["files"]):
                if progress.wasCanceled():
                    break

                progress.setValue(i)
                progress.setLabelText(f"Loading {file_info['filename']}...")
                QApplication.processEvents()

                mgf_path = file_info["path"]
                file_type = file_info["type"]

                # Check if file exists
                if not os.path.exists(mgf_path):
                    failed_files.append(f"{file_info['filename']} (file not found)")
                    continue

                try:
                    # Parse MGF file
                    entries = parse_mgf_file(mgf_path, check_required_keys=False)

                    # Count unique SMILES
                    smiles_set = set()
                    for entry in entries:
                        smiles = entry.get("smiles", "")
                        if smiles:
                            smiles_set.add(smiles)

                    # Collect all meta keys
                    for entry in entries:
                        meta_keys = {k for k in entry.keys() if k not in ["$$spectrumdata", "peaks"]}
                        self.all_meta_keys.update(meta_keys)

                    filename = os.path.basename(mgf_path)
                    self.mgf_files[filename] = {"path": mgf_path, "entries": entries, "num_entries": len(entries), "unique_smiles": len(smiles_set), "type": file_type}
                    loaded_count += 1

                except Exception as e:
                    failed_files.append(f"{file_info['filename']} ({str(e)})")

            progress.setValue(len(import_data["files"]))
            self.update_file_table()

            # Show summary
            summary = f"Loaded {loaded_count} file(s)"
            if failed_files:
                summary += f"\n\nFailed to load {len(failed_files)} file(s):\n"
                summary += "\n".join(failed_files[:5])
                if len(failed_files) > 5:
                    summary += f"\n... and {len(failed_files) - 5} more"

            QMessageBox.information(self, "Import Complete", summary)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to import configuration:\n{str(e)}")

    # Section 2 methods
    def refresh_metadata_overview(self):
        """Refresh the metadata overview table using embeddings dataframe."""
        if self.df_embeddings is None:
            QMessageBox.warning(self, "Warning", "Please generate embeddings first (Section 2)")
            return

        # Get unique sources (MGF files) from the dataframe
        sources = sorted(self.df_embeddings["source"].unique())

        # Get all metadata columns (exclude standard embedding-related columns)
        exclude_cols = {"source", "name", "smiles", "inchikey", "ionmode", "precursor_mz", "adduct", "label", "num_peaks", "embeddings", "predicted_class"}
        all_keys = sorted([col for col in self.df_embeddings.columns if col not in exclude_cols])

        self.meta_table.setRowCount(len(all_keys))
        self.meta_table.setColumnCount(len(sources))
        self.meta_table.setHorizontalHeaderLabels(sources)
        self.meta_table.setVerticalHeaderLabels(all_keys)

        # Configure column resize modes to Interactive
        header = self.meta_table.horizontalHeader()
        for i in range(len(sources)):
            header.setSectionResizeMode(i, QHeaderView.Interactive)

        # Count unique values for each meta-key in each source
        for col, source in enumerate(sources):
            # Filter dataframe for this source
            source_df = self.df_embeddings[self.df_embeddings["source"] == source]

            for row, key in enumerate(all_keys):
                # Get non-null values for this key
                values = source_df[key].dropna().astype(str).unique()
                missing_count = source_df[key].isna().sum()

                # Show count of unique values
                item = QTableWidgetItem(str(len(values)))
                item.setData(Qt.UserRole, {"source": source, "key": key, "values": set(values), "missing": missing_count})
                self.meta_table.setItem(row, col, item)

        # Auto-resize columns to contents
        self.meta_table.resizeColumnsToContents()

        QMessageBox.information(self, "Success", "Metadata overview refreshed from embeddings")

    def on_meta_cell_selected(self):
        """Handle meta cell selection to show unique values."""
        selected = self.meta_table.selectedItems()
        if not selected:
            return

        item = selected[0]
        data = item.data(Qt.UserRole)

        if not data:
            return

        if self.df_embeddings is None:
            return

        self.value_list.clear()

        # Get the values and their counts
        source = data["source"]
        key = data["key"]

        # Filter dataframe for this source
        source_df = self.df_embeddings[self.df_embeddings["source"] == source]

        # Get value counts
        value_counts = source_df[key].value_counts().to_dict()
        missing_count = source_df[key].isna().sum()

        # Update label to show it's for a specific file
        self.value_list_label.setText(f"Unique Values for '{key}' in '{source}':")

        # Add missing values
        if missing_count > 0:
            self.value_list.addItem(f"<missing>: {missing_count}")

        # Add other values (convert to string for display)
        for value, count in sorted(value_counts.items(), key=lambda x: str(x[0])):
            self.value_list.addItem(f"{value}: {count}")

    def on_meta_key_clicked(self, row):
        """Handle meta-field name click to show all unique values across all samples."""
        if self.df_embeddings is None:
            return

        # Get the meta-key for this row
        exclude_cols = {"source", "name", "smiles", "inchikey", "ionmode", "precursor_mz", "adduct", "label", "num_peaks", "embeddings", "predicted_class"}
        all_keys = sorted([col for col in self.df_embeddings.columns if col not in exclude_cols])

        if row >= len(all_keys):
            return

        key = all_keys[row]

        self.value_list.clear()

        # Update label to show it's across all samples
        self.value_list_label.setText(f"Unique Values for '{key}' across ALL samples:")

        # Get value counts across all sources
        value_counts = self.df_embeddings[key].value_counts().to_dict()
        missing_count = self.df_embeddings[key].isna().sum()

        # Add missing values
        if missing_count > 0:
            self.value_list.addItem(f"<missing>: {missing_count}")

        # Add other values (convert to string for display)
        for value, count in sorted(value_counts.items(), key=lambda x: str(x[0])):
            self.value_list.addItem(f"{value}: {count}")

    def check_subset_syntax(self):
        """Check if the subset syntax is valid."""
        subset_expr = self.subset_input.text().strip()

        if not subset_expr:
            QMessageBox.warning(self, "Warning", "Please enter a subset expression")
            return

        if self.df_embeddings is None:
            QMessageBox.warning(self, "Warning", "Please generate embeddings first (Section 2)")
            return

        # Test with metadata from embeddings
        exclude_cols = {"source", "name", "smiles", "inchikey", "ionmode", "precursor_mz", "adduct", "label", "num_peaks", "embeddings", "predicted_class"}
        test_meta = {col: "test_value" for col in self.df_embeddings.columns if col not in exclude_cols}
        test_meta["CE"] = "20.0"
        test_meta["ionmode"] = "pos"

        try:
            # Try to compile and evaluate
            compiled = compile(subset_expr, "<string>", "eval")
            result = eval(compiled, {"meta": test_meta, "abs": abs, "float": float, "int": int, "str": str})
            QMessageBox.information(self, "Success", "Subset syntax is valid!")
        except Exception as e:
            QMessageBox.warning(self, "Syntax Error", f"Invalid subset syntax:\n{str(e)}")

    def add_subset(self):
        """Add a subset to the subset list."""
        subset_name = self.subset_name_input.text().strip()
        subset_expr = self.subset_input.text().strip()

        if not subset_name:
            QMessageBox.warning(self, "Warning", "Please enter a subset name")
            return

        if not subset_expr:
            QMessageBox.warning(self, "Warning", "Please enter a subset expression")
            return

        if self.df_embeddings is None:
            QMessageBox.warning(self, "Warning", "Please generate embeddings first (Section 2)")
            return

        # Test the subset with a sample from the embeddings
        exclude_cols = {"source", "name", "smiles", "inchikey", "ionmode", "precursor_mz", "adduct", "label", "num_peaks", "embeddings", "predicted_class"}
        test_meta = {col: "test_value" for col in self.df_embeddings.columns if col not in exclude_cols}
        try:
            compiled = compile(subset_expr, "<string>", "eval")
            eval(compiled, {"meta": test_meta, "abs": abs, "float": float, "int": int, "str": str})
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Invalid subset:\n{str(e)}")
            return

        # Count matches for each type using the embeddings dataframe
        type_counts = defaultdict(int)

        # Get file types from mgf_files (mapping source to type)
        file_type_map = {filename: data["type"] for filename, data in self.mgf_files.items()}

        # Iterate through embeddings dataframe
        for idx, row in self.df_embeddings.iterrows():
            source = row.get("source", "")
            file_type = file_type_map.get(source, "unknown")

            # Create meta dict from row (exclude special columns)
            meta = {col: row[col] for col in self.df_embeddings.columns if col not in exclude_cols}

            try:
                compiled = compile(subset_expr, "<string>", "eval")
                if eval(compiled, {"meta": meta, "abs": abs, "float": float, "int": int, "str": str}):
                    type_counts[file_type] += 1
            except:
                pass

        # Add to subsets list
        self.subsets.append({"name": subset_name, "expression": subset_expr, "counts": type_counts})

        # Update table
        self.update_subset_table()
        # Do not clear the subset input fields

    def update_subset_table(self):
        """Update the subset table."""
        self.subset_table.setRowCount(len(self.subsets))

        type_order = ["validation - relevant", "validation - other", "inference", "train - relevant", "train - other"]

        for row, subset_data in enumerate(self.subsets):
            # Name
            self.subset_table.setItem(row, 0, QTableWidgetItem(subset_data.get("name", "Unnamed")))

            # Expression
            self.subset_table.setItem(row, 1, QTableWidgetItem(subset_data["expression"]))

            # Counts for each type with color coding
            for col, file_type in enumerate(type_order, start=2):
                count = subset_data["counts"].get(file_type, 0)
                item = QTableWidgetItem(str(count))

                # Set background color based on file type
                if file_type == "train - relevant":
                    item.setBackground(QBrush(COLOR_TRAIN_RELEVANT))
                elif file_type == "train - other":
                    item.setBackground(QBrush(COLOR_TRAIN_OTHER))
                elif file_type == "validation - relevant":
                    item.setBackground(QBrush(COLOR_VALIDATION_RELEVANT))
                elif file_type == "validation - other":
                    item.setBackground(QBrush(COLOR_VALIDATION_OTHER))
                elif file_type == "inference":
                    item.setBackground(QBrush(COLOR_INFERENCE))

                self.subset_table.setItem(row, col, item)

        # Auto-resize columns to contents
        self.subset_table.resizeColumnsToContents()

    def edit_selected_subset(self):
        """Edit the selected subset."""
        selected_rows = list(set(item.row() for item in self.subset_table.selectedItems()))

        if not selected_rows:
            QMessageBox.warning(self, "Warning", "Please select a subset to edit")
            return

        if len(selected_rows) > 1:
            QMessageBox.warning(self, "Warning", "Please select only one subset to edit")
            return

        row = selected_rows[0]
        if row < 0 or row >= len(self.subsets):
            return

        # Get the subset name and expression and put them in the text fields
        subset_name = self.subsets[row].get("name", "")
        subset_expr = self.subsets[row]["expression"]
        self.subset_name_input.setText(subset_name)
        self.subset_input.setText(subset_expr)

        # Delete the subset from the list
        del self.subsets[row]
        self.update_subset_table()

        QMessageBox.information(self, "Edit Subset", "Subset loaded into text fields for editing.\nModify it and click 'Add Subset' when done.")

    def delete_selected_subset(self):
        """Delete the selected subset(s)."""
        selected_rows = set(item.row() for item in self.subset_table.selectedItems())

        if not selected_rows:
            QMessageBox.warning(self, "Warning", "Please select a subset to delete")
            return

        # Delete subsets (in reverse order to maintain indices)
        for row in sorted(selected_rows, reverse=True):
            del self.subsets[row]

        self.update_subset_table()

    def export_subsets(self):
        """Export subsets to JSON."""
        if not self.subsets:
            QMessageBox.warning(self, "Warning", "No subsets defined to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Export Subsets", "subsets.json", "JSON Files (*.json);;All Files (*)")

        if not file_path:
            return

        try:
            # Export subsets (names and expressions, counts will be recalculated on import)
            export_data = {"subsets": [{"name": s.get("name", "Unnamed"), "expression": s["expression"]} for s in self.subsets]}

            with open(file_path, "w") as f:
                json.dump(export_data, f, indent=2)

            QMessageBox.information(self, "Success", f"Subsets exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export subsets:\n{str(e)}")

    def import_subsets(self):
        """Import subsets from JSON."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Subsets", "", "JSON Files (*.json);;All Files (*)")

        if not file_path:
            return

        if self.df_embeddings is None:
            QMessageBox.warning(self, "Warning", "Please generate embeddings first (Section 2)")
            return

        try:
            with open(file_path, "r") as f:
                import_data = json.load(f)

            # Support both old "filters" and new "subsets" format for backward compatibility
            if "subsets" in import_data:
                subset_list = import_data["subsets"]
            elif "filters" in import_data:
                subset_list = import_data["filters"]
            else:
                QMessageBox.warning(self, "Error", "Invalid subset file format")
                return

            # Import subsets and recalculate counts using embeddings dataframe
            imported_count = 0
            failed_subsets = []

            # Get metadata columns from embeddings
            exclude_cols = {"source", "name", "smiles", "inchikey", "ionmode", "precursor_mz", "adduct", "label", "num_peaks", "embeddings", "predicted_class"}

            # Get file types from mgf_files (mapping source to type)
            file_type_map = {filename: data["type"] for filename, data in self.mgf_files.items()}

            for subset_info in subset_list:
                subset_name = subset_info.get("name", "Imported Subset")
                subset_expr = subset_info["expression"]

                # Test the subset with metadata from embeddings
                test_meta = {col: "test_value" for col in self.df_embeddings.columns if col not in exclude_cols}
                try:
                    compiled = compile(subset_expr, "<string>", "eval")
                    eval(compiled, {"meta": test_meta, "abs": abs, "float": float, "int": int, "str": str})
                except Exception as e:
                    failed_subsets.append(f"{subset_name}: {subset_expr} ({str(e)})")
                    continue

                # Count matches for each type using the embeddings dataframe
                type_counts = defaultdict(int)

                for idx, row in self.df_embeddings.iterrows():
                    source = row.get("source", "")
                    file_type = file_type_map.get(source, "unknown")

                    # Create meta dict from row (exclude special columns)
                    meta = {col: row[col] for col in self.df_embeddings.columns if col not in exclude_cols}

                    try:
                        compiled = compile(subset_expr, "<string>", "eval")
                        if eval(compiled, {"meta": meta, "abs": abs, "float": float, "int": int, "str": str}):
                            type_counts[file_type] += 1
                    except:
                        pass

                # Add to subsets list with recalculated counts
                self.subsets.append({"name": subset_name, "expression": subset_expr, "counts": type_counts})
                imported_count += 1

            self.update_subset_table()

            # Show summary
            summary = f"Imported {imported_count} subset(s)"
            if failed_subsets:
                summary += f"\n\nFailed to import {len(failed_subsets)} subset(s):\n"
                summary += "\n".join(failed_subsets[:3])
                if len(failed_subsets) > 3:
                    summary += f"\n... and {len(failed_subsets) - 3} more"

            QMessageBox.information(self, "Import Complete", summary)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to import subsets:\n{str(e)}")

    # Section 3 methods
    def browse_output_dir(self):
        """Browse for output directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_input.setText(directory)

    def generate_embeddings_clicked(self):
        """Generate embeddings from loaded datasets."""
        if not self.mgf_files:
            QMessageBox.warning(self, "Warning", "No MGF files loaded")
            return

        # Prepare datasets
        datasets = []
        for filename, data in self.mgf_files.items():
            datasets.append({"name": filename, "type": data["type"], "file": data["path"], "fragmentation_method": "fragmentation_method", "colour": "#80BF02"})

        # Prepare data_to_add
        data_to_add = OrderedDict(
            [
                ("name", ["feature_id", "name", "title", "compound_name"]),
                ("formula", ["formula"]),
                ("smiles", ["smiles"]),
                ("adduct", ["adduct", "precursor_type"]),
                ("ionMode", ["ionmode"]),
                ("RTINSECONDS", ["rtinseconds", "retention_time"]),
                ("precursor_mz", ["pepmass", "precursor_mz"]),
                ("fragmentation_method", ["fragmentation_method", "fragmentation_mode"]),
                ("CE", ["collision_energy"]),
            ]
        )

        # Create progress dialog
        self.progress_dialog = QProgressDialog("Generating embeddings...\nSee console for further details", "Cancel", 0, 0, self)
        self.embedding_status.setText("Status: Generating embeddings in progress...")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)

        # Start worker thread
        self.embedding_worker = EmbeddingWorker(datasets, data_to_add, self.mgf_files)
        self.embedding_worker.progress.connect(self.on_embedding_progress)
        self.embedding_worker.finished.connect(self.on_embeddings_generated)
        self.embedding_worker.error.connect(self.on_embedding_error)
        self.embedding_worker.start()

        self.generate_embeddings_btn.setEnabled(False)

    def on_embedding_progress(self, message):
        """Update progress dialog with message."""
        self.progress_dialog.setLabelText(message)

    def on_embeddings_generated(self, df):
        """Handle embeddings generation completion."""
        self.df_embeddings = df
        self.progress_dialog.close()
        self.embedding_status.setText(f"Status: Embeddings generated ({len(df)} entries)")
        self.train_btn.setEnabled(True)
        self.generate_embeddings_btn.setEnabled(True)

        # Enable classifier configuration inputs
        self.classifiers_config_text.setEnabled(True)
        self.load_default_classifiers_btn.setEnabled(True)
        self.min_prediction_threshold_input.setEnabled(True)

        # Save embeddings
        output_dir = self.output_dir_input.text()
        os.makedirs(output_dir, exist_ok=True)
        pickle_file = os.path.join(output_dir, "df_embeddings.pkl")
        df.to_pickle(pickle_file)

        # If we're in the middle of loading a full configuration, continue the workflow
        if self.pending_config_data is not None:
            # Navigate to subsets section
            self.go_to_section(self.section3)
            self.continue_loading_configuration()
        else:
            QMessageBox.information(self, "Success", f"Embeddings generated and saved to:\n{pickle_file}")

    def on_embedding_error(self, error_msg):
        """Handle embedding generation error."""
        self.progress_dialog.close()
        self.embedding_status.setText("Status: Error occurred")
        self.generate_embeddings_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", error_msg)

    def load_default_classifiers_config(self):
        """Load the default classifiers configuration into the text box."""
        default_config = """from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import VotingClassifier

classifiers = {
    "Nearest Neighbors n=3": KNeighborsClassifier(3),
    "Nearest Neighbors n=10": KNeighborsClassifier(10),
    "Linear SVM": SVC(kernel="linear", C=0.025, random_state=42),
    #"RBF SVM": SVC(gamma=2, C=1, random_state=42),
    # "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42),
    "Neural Net": MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "LDA": LinearDiscriminantAnalysis(solver="svd", store_covariance=True, n_components=1),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    # "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Bagging Classifier": BaggingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Ridge Classifier": RidgeClassifier(random_state=42),
    "Voting Classifier (soft)": VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(random_state=42, max_iter=1000)),
            ("rf", RandomForestClassifier(random_state=42)),
            ("gnb", GaussianNB()),
        ],
        voting="soft",
    ),
}"""
        self.classifiers_config_text.setPlainText(default_config)

    def save_classifier_configuration(self):
        """Save the classifier configuration and threshold to JSON file."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Classifier Configuration", "", "JSON Files (*.json);;All Files (*)")

        if not file_path:
            return

        try:
            config_data = {"classifier_configuration": self.classifiers_config_text.toPlainText(), "min_prediction_threshold": self.min_prediction_threshold_input.value()}

            with open(file_path, "w") as f:
                json.dump(config_data, f, indent=4)

            QMessageBox.information(self, "Success", f"Configuration saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save configuration:\n{str(e)}")

    def load_classifier_configuration(self):
        """Load classifier configuration and threshold from JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Classifier Configuration", "", "JSON Files (*.json);;All Files (*)")

        if not file_path:
            return

        try:
            with open(file_path, "r") as f:
                config_data = json.load(f)

            # Load classifier configuration text
            if "classifier_configuration" in config_data:
                self.classifiers_config_text.setPlainText(config_data["classifier_configuration"])

            # Load min prediction threshold
            if "min_prediction_threshold" in config_data:
                self.min_prediction_threshold_input.setValue(config_data["min_prediction_threshold"])

            QMessageBox.information(self, "Success", f"Configuration loaded from:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load configuration:\n{str(e)}")

    def save_full_configuration(self):
        """Save complete workflow configuration (MGF files, output dir, subsets, classifier config)."""
        if not self.mgf_files:
            QMessageBox.warning(self, "Warning", "No MGF files loaded. Nothing to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Full Configuration", "", "JSON Files (*.json);;All Files (*)")

        if not file_path:
            return

        try:
            # Collect all configuration data
            config_data = {
                "mgf_files": [],
                "output_directory": self.output_dir_input.text(),
                "subsets": self.subsets,
                "classifier_configuration": self.classifiers_config_text.toPlainText(),
                "min_prediction_threshold": self.min_prediction_threshold_input.value(),
            }

            # Save MGF file paths and types
            for filename, data in self.mgf_files.items():
                config_data["mgf_files"].append({"path": data["path"], "type": data["type"]})

            with open(file_path, "w") as f:
                json.dump(config_data, f, indent=4)

            QMessageBox.information(self, "Success", f"Full configuration saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save configuration:\n{str(e)}")

    def load_full_configuration(self):
        """Load complete workflow configuration and automatically execute workflow steps."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Full Configuration", "", "JSON Files (*.json);;All Files (*)")

        if not file_path:
            return

        try:
            with open(file_path, "r") as f:
                config_data = json.load(f)

            # Clear existing data
            self.mgf_files.clear()
            self.subsets.clear()
            self.df_embeddings = None
            self.pending_config_data = config_data

            # Step 1: Load MGF files
            if "mgf_files" in config_data:
                progress = QProgressDialog("Loading MGF files...", "Cancel", 0, len(config_data["mgf_files"]), self)
                progress.setWindowModality(Qt.WindowModal)

                for i, file_info in enumerate(config_data["mgf_files"]):
                    if progress.wasCanceled():
                        break

                    progress.setValue(i)
                    file_path_mgf = file_info["path"]
                    progress.setLabelText(f"Loading {os.path.basename(file_path_mgf)}...")
                    QApplication.processEvents()

                    try:
                        # Parse MGF file
                        entries = parse_mgf_file(file_path_mgf, check_required_keys=False)

                        # Count unique SMILES
                        smiles_set = set()
                        for entry in entries:
                            smiles = entry.get("smiles", "")
                            if smiles:
                                smiles_set.add(smiles)

                        # Collect all meta keys
                        for entry in entries:
                            meta_keys = {k for k in entry.keys() if k not in ["$$spectrumdata", "peaks"]}
                            self.all_meta_keys.update(meta_keys)

                        filename = os.path.basename(file_path_mgf)
                        self.mgf_files[filename] = {
                            "path": file_path_mgf,
                            "entries": entries,
                            "num_entries": len(entries),
                            "unique_smiles": len(smiles_set),
                            "type": file_info.get("type", "inference"),
                        }
                    except Exception as e:
                        QMessageBox.warning(self, "Error", f"Failed to load {file_path_mgf}:\n{str(e)}")

                progress.setValue(len(config_data["mgf_files"]))
                self.update_file_table()

            # Step 2: Set output directory
            if "output_directory" in config_data:
                self.output_dir_input.setText(config_data["output_directory"])

            # Step 3: Generate embeddings automatically (workflow continues in on_embeddings_generated)
            if self.mgf_files:
                # Navigate to embeddings section
                self.go_to_section(self.section2)
                self.generate_embeddings_clicked()
            else:
                self.pending_config_data = None

        except Exception as e:
            self.pending_config_data = None
            QMessageBox.critical(self, "Error", f"Failed to load full configuration:\n{str(e)}")

    def continue_loading_configuration(self):
        """Continue loading configuration after embeddings are generated."""
        if self.pending_config_data is None or self.df_embeddings is None:
            return

        try:
            config_data = self.pending_config_data

            # Step 4: Refresh metadata overview
            self.refresh_metadata_overview()

            # Step 5: Load subsets and recalculate counts
            if "subsets" in config_data:
                # Get metadata columns from embeddings
                exclude_cols = {"source", "name", "smiles", "inchikey", "ionmode", "precursor_mz", "adduct", "label", "num_peaks", "embeddings", "predicted_class"}

                # Get file types from mgf_files (mapping source to type)
                file_type_map = {filename: data["type"] for filename, data in self.mgf_files.items()}

                for subset_info in config_data["subsets"]:
                    subset_name = subset_info.get("name", "Unnamed")
                    subset_expr = subset_info["expression"]

                    # Count matches for each type using the embeddings dataframe
                    type_counts = defaultdict(int)

                    for idx, row in self.df_embeddings.iterrows():
                        source = row.get("source", "")
                        file_type = file_type_map.get(source, "unknown")

                        # Create meta dict from row (exclude special columns)
                        meta = {col: row[col] for col in self.df_embeddings.columns if col not in exclude_cols}

                        try:
                            compiled = compile(subset_expr, "<string>", "eval")
                            if eval(compiled, {"meta": meta, "abs": abs, "float": float, "int": int, "str": str}):
                                type_counts[file_type] += 1
                        except:
                            pass

                    # Add to subsets list with recalculated counts
                    self.subsets.append({"name": subset_name, "expression": subset_expr, "counts": type_counts})

                self.update_subset_table()

            # Step 6: Load classifier configuration
            if "classifier_configuration" in config_data:
                self.classifiers_config_text.setPlainText(config_data["classifier_configuration"])

            if "min_prediction_threshold" in config_data:
                self.min_prediction_threshold_input.setValue(config_data["min_prediction_threshold"])

            # Navigate to training section
            self.go_to_section(self.section4)

            QMessageBox.information(
                self,
                "Success",
                f"Configuration loaded and workflow executed:\n"
                f"- Loaded {len(self.mgf_files)} MGF files\n"
                f"- Generated embeddings ({len(self.df_embeddings)} entries)\n"
                f"- Loaded {len(self.subsets)} subsets\n"
                f"- Loaded classifier configuration\n\n"
                f"Ready to train classifiers!",
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to complete configuration loading:\n{str(e)}")
        finally:
            self.pending_config_data = None

    def show_about(self):
        """Show about dialog with GUI description and version."""
        version = self.get_version()
        about_text = f"""<h2>Flavonoid Compound Classification GUI</h2>
        <p><b>Version:</b> {version}</p>
        
        <p>This application provides a complete workflow for classifying flavonoid compounds using mass spectrometry data:</p>
        
        <ol>
        <li><b>Load MGF Files:</b> Import mass spectrometry data files and assign dataset types</li>
        <li><b>Generate Embeddings:</b> Convert spectra to embeddings using MS2DeepScore</li>
        <li><b>Define Metadata Subsets:</b> Create subsets based on metadata criteria</li>
        <li><b>Train Classifiers:</b> Configure and train machine learning classifiers</li>
        <li><b>Inspect Results:</b> View classification results and performance metrics</li>
        <li><b>Inspect Spectra:</b> Examine individual spectra and predictions</li>
        </ol>
        
        <p>Use <b>File → Save/Load Configuration</b> to persist your workflow settings across sessions.</p>
        """

        QMessageBox.about(self, "About", about_text)

    def train_classifiers_clicked(self):
        """Train classifiers on the loaded datasets."""
        if self.df_embeddings is None:
            QMessageBox.warning(self, "Warning", "Please generate embeddings first")
            return

        if not self.subsets:
            QMessageBox.warning(self, "Warning", "Please define at least one subset before training")
            return

        # Parse classifiers configuration
        classifiers_config_text = self.classifiers_config_text.toPlainText().strip()
        if classifiers_config_text:
            try:
                # Evaluate the user's config
                exec(classifiers_config_text)

                if "classifiers" not in locals().keys():
                    QMessageBox.warning(self, "Error", "Classifiers configuration must define a 'classifiers' variable")
                    return
                self.classifiers_config = locals()["classifiers"]
                if not isinstance(self.classifiers_config, dict):
                    QMessageBox.warning(self, "Error", "Classifiers configuration must be a dictionary")
                    return
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Invalid classifiers configuration:\n{str(e)}")
                return
        else:
            # Use default (None will trigger default in train_and_classify)
            self.classifiers_config = None

        # Get min prediction threshold
        self.min_prediction_threshold = self.min_prediction_threshold_input.value()

        output_dir = self.output_dir_input.text()
        os.makedirs(output_dir, exist_ok=True)

        # Clear previous results
        self.subset_results.clear()
        self.current_subset_index = 0

        # Train classifiers for each subset separately
        self.train_next_subset(output_dir)

    def train_next_subset(self, output_dir):
        """Train classifiers for the next subset in the queue."""
        if self.current_subset_index >= len(self.subsets):
            # All subsets trained
            self.training_status.setText(f"Status: Training completed for {len(self.subsets)} subset(s), output written to {output_dir}.")
            self.train_btn.setEnabled(True)
            self.populate_subset_results_list()
            QMessageBox.information(self, "Success", f"Classifiers trained for all {len(self.subsets)} subset(s)\nOutput written to {output_dir}.")
            return

        subset_data = self.subsets[self.current_subset_index]
        subset_name = subset_data["name"]
        subset_expr = subset_data["expression"]

        # Create subset function
        def subset_func(row):
            meta = {k: v for k, v in row.items() if k not in ["$$spectrumdata", "peaks", "embeddings"]}
            try:
                compiled = compile(subset_expr, "<string>", "eval")
                res = eval(compiled, {"meta": meta, "abs": abs, "float": float, "int": int, "str": str})
                # print(f"Evaluating subset '{subset_name}' for row {meta} => {res} with function {inspect.getsource(subset_func)}")
                return res
            except Exception as ex:
                print(f"Error evaluating subset expression for row {meta}: {ex}")
                return False

        subsets_dict = {subset_name: subset_func}

        subset_output_dir = os.path.join(output_dir, subset_name.replace(" ", "_"))
        os.makedirs(subset_output_dir, exist_ok=True)

        # Create progress dialog
        self.progress_dialog = QProgressDialog(f"Training classifiers", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)

        # Start worker thread with subset_name as parameter
        self.current_training_subset = subset_name  # Store in instance variable
        self.training_worker = TrainingWorker(self.df_embeddings, subsets_dict, subset_output_dir, classifiers=self.classifiers_config, min_prediction_threshold=self.min_prediction_threshold)
        self.training_worker.progress.connect(self.on_training_progress)
        self.training_worker.finished.connect(self.on_training_finished)
        self.training_worker.error.connect(self.on_training_error)
        self.training_worker.start()

        self.train_btn.setEnabled(False)

    def on_training_progress(self, message):
        """Update progress dialog with message."""
        self.progress_dialog.setLabelText(message)

    def on_training_finished(self, df_train, df_validation, df_inference, df_metrics, trained_classifiers, tables):
        """Handle training completion."""
        subset_name = self.current_training_subset
        long_tables, pivot_tables = tables

        # Store results for this subset including long_table and pivot_table
        self.subset_results[subset_name] = {
            "df_train": df_train,
            "df_validation": df_validation,
            "df_inference": df_inference,
            "df_metrics": df_metrics,
            "trained_classifiers": trained_classifiers,
            "long_table": long_tables.get(subset_name),
            "pivot_table": pivot_tables.get(subset_name),
        }

        self.progress_dialog.close()

        # Move to next subset
        self.current_subset_index += 1
        output_dir = self.output_dir_input.text()
        self.train_next_subset(output_dir)

        # Populate spectrum file list for section 5 after all training
        if self.current_subset_index >= len(self.subsets):
            self.populate_spectrum_file_list()

    def on_training_error(self, error_msg):
        """Handle training error."""
        self.progress_dialog.close()
        self.training_status.setText("Status: Error occurred")
        self.train_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", error_msg)

        # Continue with next subset despite error
        self.current_subset_index += 1
        output_dir = self.output_dir_input.text()
        self.train_next_subset(output_dir)

    def populate_subset_results_list(self):
        """Populate the subset results list."""
        self.subset_results_list.clear()
        for subset_name in sorted(self.subset_results.keys()):
            self.subset_results_list.addItem(subset_name)

    # Section 4 methods
    def on_subset_result_selected(self, item):
        """Handle subset selection to display results."""
        subset_name = item.text()
        if subset_name not in self.subset_results:
            return

        self.current_subset_label.setText(f"<b>Results for Subset: {subset_name}</b>")
        self.refresh_results_table_for_subset(subset_name)

    def refresh_results_table_for_subset(self, subset_name):
        """Refresh the results table for a specific subset using the long table from generate_prediction_overview."""
        if subset_name not in self.subset_results:
            return

        results = self.subset_results[subset_name]
        long_table = results.get("long_table")

        if long_table is None or long_table.empty:
            QMessageBox.warning(self, "Warning", "No long table results available for this subset")
            return

        # Use the long table which has the "classification:relevant" column
        df = long_table.copy()

        # Ensure the classification:relevant column exists
        if "classification:relevant" not in df.columns:
            QMessageBox.warning(self, "Warning", "The long table does not contain 'classification:relevant' column")
            return

        # Aggregate by source and type, counting relevant vs not relevant
        # Create a binary relevant flag: relevant if "classification:relevant" == "relevant", otherwise not relevant
        df["is_relevant"] = df["classification:relevant"] == "relevant"

        # Group by source and type, then count relevant and not relevant
        summary = df.groupby(["source", "type"]).agg(total=("is_relevant", "count"), relevant_count=("is_relevant", "sum")).reset_index()

        # Calculate not relevant count
        summary["not_relevant_count"] = summary["total"] - summary["relevant_count"]

        # Get file types for display
        file_types = {}
        for filename, data in self.mgf_files.items():
            file_types[filename] = data["type"]

        # Set up the table
        self.results_table.setRowCount(len(summary))
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels(["MGF File", "Type", "Prediction: Other (Count)", "Prediction: Other (%)", "Prediction: Relevant (Count)", "Prediction: Relevant (%)"])

        # Populate the table
        for row_idx, row in summary.iterrows():
            source = row["source"]
            file_type = file_types.get(source, "unknown")
            total = row["total"]
            not_relevant = row.get("not_relevant_count", 0)
            relevant = row.get("relevant_count", 0)

            not_relevant_pct = (not_relevant / total * 100) if total > 0 else 0
            relevant_pct = (relevant / total * 100) if total > 0 else 0

            # Determine if this is inference data
            is_inference = "inference" in file_type
            expected_type = "relevant" if "relevant" in file_type else "other"

            # Source column
            self.results_table.setItem(row_idx, 0, QTableWidgetItem(source))

            # Type column with color coding
            type_item = QTableWidgetItem(file_type)
            if file_type == "train - relevant":
                type_item.setBackground(QBrush(COLOR_TRAIN_RELEVANT))
            elif file_type == "train - other":
                type_item.setBackground(QBrush(COLOR_TRAIN_OTHER))
            elif file_type == "validation - relevant":
                type_item.setBackground(QBrush(COLOR_VALIDATION_RELEVANT))
            elif file_type == "validation - other":
                type_item.setBackground(QBrush(COLOR_VALIDATION_OTHER))
            elif "inference" in file_type:
                type_item.setBackground(QBrush(COLOR_INFERENCE))
            self.results_table.setItem(row_idx, 1, type_item)

            # Define match/mismatch colors
            COLOR_MATCH = QColor(144, 238, 144)  # Light green
            COLOR_MISMATCH = QColor(255, 182, 193)  # Light red

            # Not relevant count
            item = QTableWidgetItem(str(int(not_relevant)))
            if is_inference:
                item.setBackground(QBrush(COLOR_INFERENCE))
            elif expected_type == "other":
                item.setBackground(QBrush(COLOR_MATCH))
            else:
                item.setBackground(QBrush(COLOR_MISMATCH))
            self.results_table.setItem(row_idx, 2, item)

            # Not relevant %
            item = QTableWidgetItem(f"{not_relevant_pct:.1f}%")
            if is_inference:
                item.setBackground(QBrush(COLOR_INFERENCE))
            elif expected_type == "other":
                item.setBackground(QBrush(COLOR_MATCH))
            else:
                item.setBackground(QBrush(COLOR_MISMATCH))
            self.results_table.setItem(row_idx, 3, item)

            # Relevant count
            item = QTableWidgetItem(str(int(relevant)))
            if is_inference:
                item.setBackground(QBrush(COLOR_INFERENCE))
            elif expected_type == "relevant":
                item.setBackground(QBrush(COLOR_MATCH))
            else:
                item.setBackground(QBrush(COLOR_MISMATCH))
            self.results_table.setItem(row_idx, 4, item)

            # Relevant %
            item = QTableWidgetItem(f"{relevant_pct:.1f}%")
            if is_inference:
                item.setBackground(QBrush(COLOR_INFERENCE))
            elif expected_type == "relevant":
                item.setBackground(QBrush(COLOR_MATCH))
            else:
                item.setBackground(QBrush(COLOR_MISMATCH))
            self.results_table.setItem(row_idx, 5, item)

        # Auto-resize columns
        self.results_table.resizeColumnsToContents()

    def export_results(self):
        """Export results to Excel."""
        if not self.subset_results:
            QMessageBox.warning(self, "Warning", "No results to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "Excel Files (*.xlsx);;All Files (*)")

        if not file_path:
            return

        try:
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                for subset_name, results in self.subset_results.items():
                    sheet_prefix = subset_name[:20].replace(" ", "_")  # Limit sheet name length
                    if results["df_train"] is not None:
                        results["df_train"].to_excel(writer, sheet_name=f"{sheet_prefix}_train"[:31])
                    if results["df_validation"] is not None:
                        results["df_validation"].to_excel(writer, sheet_name=f"{sheet_prefix}_val"[:31])
                    if results["df_inference"] is not None:
                        results["df_inference"].to_excel(writer, sheet_name=f"{sheet_prefix}_inf"[:31])
                    if results["df_metrics"] is not None:
                        results["df_metrics"].to_excel(writer, sheet_name=f"{sheet_prefix}_metrics"[:31])

            QMessageBox.information(self, "Success", f"Results exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export results:\n{str(e)}")

    def export_classifiers(self):
        """Export trained classifiers to pickle files."""
        if not self.subset_results:
            QMessageBox.warning(self, "Warning", "No trained classifiers to export")
            return

        directory = QFileDialog.getExistingDirectory(self, "Select Directory to Save Classifiers")
        if not directory:
            return

        try:
            import pickle

            exported_count = 0

            for subset_name, results in self.subset_results.items():
                if results["trained_classifiers"] is not None:
                    safe_name = subset_name.replace(" ", "_").replace("/", "_")
                    classifier_file = os.path.join(directory, f"classifier_{safe_name}.pkl")

                    with open(classifier_file, "wb") as f:
                        pickle.dump(results["trained_classifiers"], f)

                    exported_count += 1

            QMessageBox.information(self, "Success", f"Exported {exported_count} classifier file(s) to:\n{directory}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export classifiers:\n{str(e)}")

    # Section 5 methods
    def populate_spectrum_file_list(self):
        """Populate the spectrum file list."""
        self.spectrum_file_list.clear()
        for filename in sorted(self.mgf_files.keys()):
            self.spectrum_file_list.addItem(filename)

    def on_spectrum_file_selected(self):
        """Handle spectrum file selection."""
        selected_items = self.spectrum_file_list.selectedItems()
        if not selected_items:
            return

        filename = selected_items[0].text()
        data = self.mgf_files.get(filename)

        if not data:
            return

        self.spectrum_list.clear()
        entries = data["entries"]

        for i, entry in enumerate(entries):
            # Keys are stored directly in entry, not nested in params
            name = entry.get("name", entry.get("title", f"Spectrum {i + 1}"))
            self.spectrum_list.addItem(name)

    def view_spectrum_details(self):
        """View details of selected spectrum."""
        if not self.spectrum_file_list.selectedItems() or not self.spectrum_list.selectedItems():
            QMessageBox.warning(self, "Warning", "Please select a file and spectrum")
            return

        filename = self.spectrum_file_list.selectedItems()[0].text()
        spectrum_idx = self.spectrum_list.currentRow()

        data = self.mgf_files.get(filename)
        if not data or spectrum_idx < 0 or spectrum_idx >= len(data["entries"]):
            return

        entry = data["entries"][spectrum_idx]
        # Extract metadata directly from entry (keys are not nested in params)
        meta_data = {k: v for k, v in entry.items() if k not in ["$$spectrumdata", "$$spectrumData", "peaks"]}

        # Get spectrum data - stored in $$spectrumData with [0,:] = m/z and [1,:] = intensity
        spectrum_data = entry.get("$$spectrumData", entry.get("$$spectrumdata", None))

        # Get prediction data if available from subset results
        prediction_data = {}
        for subset_name, results in self.subset_results.items():
            for df_name, df in [("train", results["df_train"]), ("validation", results["df_validation"]), ("inference", results["df_inference"])]:
                if df is not None and not df.empty:
                    # Try to find this spectrum in the dataframe
                    # This is a placeholder - actual matching would depend on how data is structured
                    prediction_data[f"{subset_name} ({df_name})"] = "Results would appear here"

        dialog = SpectrumViewer(spectrum_data, meta_data, prediction_data, self)
        dialog.exec_()

    def export_individual_spectrum(self):
        """Export individual spectrum to Excel."""
        if not self.spectrum_file_list.selectedItems() or not self.spectrum_list.selectedItems():
            QMessageBox.warning(self, "Warning", "Please select a file and spectrum")
            return

        filename = self.spectrum_file_list.selectedItems()[0].text()
        spectrum_idx = self.spectrum_list.currentRow()

        data = self.mgf_files.get(filename)
        if not data or spectrum_idx < 0 or spectrum_idx >= len(data["entries"]):
            return

        entry = data["entries"][spectrum_idx]
        # Extract metadata directly from entry
        meta_data = {k: v for k, v in entry.items() if k not in ["$$spectrumdata", "peaks"]}

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Spectrum", "", "Excel Files (*.xlsx);;All Files (*)")

        if not file_path:
            return

        try:
            df = pd.DataFrame([meta_data])
            df.to_excel(file_path, index=False)
            QMessageBox.information(self, "Success", f"Spectrum exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export spectrum:\n{str(e)}")

    # Helper methods
    def go_to_section(self, section):
        """Navigate to a specific section."""
        for s in self.sections:
            s.collapse()
        section.expand()


def main():
    app = QApplication(sys.argv)
    window = ClassificationGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
