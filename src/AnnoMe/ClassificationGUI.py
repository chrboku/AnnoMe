import sys
import os
import argparse
from collections import defaultdict, OrderedDict
import io
import json
import traceback
import tomllib
import re
import hashlib
import multiprocessing
from joblib import Parallel, delayed
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
    QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QPixmap, QImage, QColor, QBrush
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
import math
import pandas as pd
import polars as pl
import numpy as np
import tempfile
from pprint import pprint
import pickle
import inspect
from matchms import Spectrum

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
COLOR_VALIDATION_RELEVANT = QColor(156, 171, 132)  # Muted green
COLOR_VALIDATION_OTHER = QColor(158, 59, 59)  # Dark red
COLOR_INFERENCE = QColor(69, 104, 130)  # Dark blue


def calculate_file_hash(file_path, algorithm="sha256"):
    """Calculate hash of a file for integrity checking.

    Args:
        file_path: Path to the file to hash
        algorithm: Hash algorithm to use ('md5' or 'sha256')

    Returns:
        Hex string of the file hash, or None if file doesn't exist
    """
    if not os.path.exists(file_path):
        return None

    hash_obj = hashlib.sha256() if algorithm == "sha256" else hashlib.md5()

    try:
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {file_path}: {e}")
        return None


def _always_true_filter(row):
    """Trivial row filter that accepts every row.

    Used as the subset function when the DataFrame has already been
    pre-filtered outside of ``train_and_classify``.
    """
    return True


def _parallel_train_job(df_filtered, subset_name, classifier_set_name, classifier_set_config, output_dir):
    """Execute a single (subset, classifier_set) training job.

    Designed to be called via ``joblib.Parallel`` in a separate process.
    The *df_filtered* DataFrame must already be filtered for the given subset.

    stdout and stderr are captured and returned in the result dict so the
    GUI can display them in per-job log tabs.
    """
    import sys
    import datetime

    # Redirect stdout and stderr to capture all output
    log_capture = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = log_capture
    sys.stderr = log_capture

    start_time = datetime.datetime.now()
    print(f"=== Job started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    try:
        # Each worker process needs its own random seed
        set_random_seeds(42)

        # Build a trivial subset dict – data is already pre-filtered
        subsets = {subset_name: _always_true_filter}

        # Wrap the single classifier set (or None for defaults)
        if classifier_set_name is not None and classifier_set_config is not None:
            classifiers_to_compare = {classifier_set_name: classifier_set_config}
            effective_set_name = classifier_set_name
        else:
            classifiers_to_compare = None
            effective_set_name = "default_set"

        display_key = f"{effective_set_name} // {subset_name}"

        # Give each job its own sub-directory for train_and_classify artefacts
        job_output_dir = os.path.join(
            output_dir,
            display_key.replace(" // ", "_").replace(" ", "_"),
        )
        os.makedirs(job_output_dir, exist_ok=True)

        df_train, df_validation, df_inference, df_metrics, trained_classifiers = train_and_classify(
            df_filtered,
            subsets=subsets,
            output_dir=job_output_dir,
            classifiers_to_compare=classifiers_to_compare,
        )

        # --- prediction overview -------------------------------------------
        long_table = None
        pivot_table = None

        for combined_key in trained_classifiers.keys():
            subset_fn_tc = trained_classifiers[combined_key][0]
            classifiers_list = trained_classifiers[combined_key][1]
            min_threshold = trained_classifiers[combined_key][2] if len(trained_classifiers[combined_key]) > 2 else 120

            df_working_copy = df_filtered.copy()
            df_subset_infe = predict(
                df_working_copy,
                classifiers_list,
                subset_name,
                subset_fn_tc,
            )
            long_table, pivot_table = generate_prediction_overview(
                df_working_copy,
                df_subset_infe,
                output_dir,
                file_prefix=display_key.replace(" // ", "_"),
                min_prediction_threshold=min_threshold,
            )
            break  # only one key expected per job

        end_time = datetime.datetime.now()
        duration = end_time - start_time
        total_seconds = int(duration.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\n=== Job finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
        print(f"=== Duration: {hours:02d}:{minutes:02d}:{seconds:02d} ===")

        return {
            "display_key": display_key,
            "subset_name": subset_name,
            "df_train": df_train,
            "df_validation": df_validation,
            "df_inference": df_inference,
            "df_metrics": df_metrics,
            "trained_classifiers": trained_classifiers,
            "long_table": long_table,
            "pivot_table": pivot_table,
            "log": log_capture.getvalue(),
        }
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


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

            # copy input file to avoid any problems between matching MS2Spectra imported data and parse_mgf_file imported data
            original_file_paths = {}
            dfs = []

            for ds in self.datasets:
                original_file_paths[ds["name"]] = ds["file"]

                cur_id = 0
                filename = os.path.basename(ds["file"])
                # path to save embeddings cache
                mgf_pickle_file = ds["file"] + "_embeddings.pickle"
                # get has of input mgf file
                mgf_hash = calculate_file_hash(ds["file"], algorithm="sha256")

                # if pickle file exists and the hash matches the calculated has, load the embeddings from there
                if os.path.exists(mgf_pickle_file):
                    with open(mgf_pickle_file, "rb") as f:
                        cached_hash, df = pickle.load(f)
                        if cached_hash == mgf_hash:
                            print(f"Loaded embeddings from cache for {filename}")
                            dfs.append(df)
                        else:
                            print(f"Hash mismatch for {filename}, regenerating embeddings")
                else:
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mgf", mode="w", encoding="utf-8")
                    with open(ds["file"], "r") as fin:
                        lines = []
                        for line in fin:
                            if line.lower().startswith("begin ions"):
                                if len(lines) > 0:
                                    lines.insert(0, f"AnnoMe_internal_ID={filename}___{cur_id}\n")
                                    cur_id += 1
                                    temp_file.write(line)
                                    for l in lines:
                                        temp_file.write(l)
                                    lines = []
                            else:
                                if not line.lower().startswith("annome_internal_id"):
                                    lines.append(line)

                        if len(lines) > 0:
                            lines.insert(0, f"AnnoMe_internal_ID={filename}___{cur_id}\n")
                            cur_id += 1
                            temp_file.write("BEGIN IONS\n")
                            for l in lines:
                                temp_file.write(l)

                    temp_file.close()
                    ds["file"] = temp_file.name

                    try:
                        df = generate_embeddings([ds], self.data_to_add)
                        df = add_all_metadata([ds], df)
                        dfs.append(df)
                    except Exception as e:
                        error_msg = f"Error generating embeddings for {filename}: {str(e)}\n{traceback.format_exc()}"
                        print(error_msg)
                        self.error.emit(error_msg)
                        return
                    finally:
                        # Clean up temporary files and restore original paths
                        try:
                            os.remove(ds["file"])
                        except OSError:
                            pass
                        ds["file"] = original_file_paths[ds["name"]]

                    # write tuple of hash and df to pickle file
                    with open(mgf_pickle_file, "wb") as f:
                        pickle.dump((mgf_hash, df), f)

            df = pd.concat(dfs, axis=0, ignore_index=True)
            self.finished.emit(df)

        except Exception as e:
            error_msg = f"Error generating embeddings: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.error.emit(error_msg)


class TrainingWorker(QThread):
    """Worker thread that dispatches parallel training jobs via joblib."""

    finished = pyqtSignal(object)  # list of result dicts from _parallel_train_job
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    log = pyqtSignal(str)  # Emits log messages

    def __init__(self, df_subset, subsets, output_dir, classifiers_to_compare=None, n_jobs=-1):
        super().__init__()
        self.df_subset = df_subset
        self.subsets = subsets
        self.output_dir = output_dir
        self.classifiers_to_compare = classifiers_to_compare
        self.n_jobs = n_jobs

    def run(self):
        import sys

        try:
            # ----- pre-filter the DataFrame once per subset ---------------
            self.progress.emit("Pre-filtering data for each subset…\nSee console for further details")
            filtered_dfs = {}
            for subset_name, subset_fn in self.subsets.items():
                mask = self.df_subset.apply(subset_fn, axis=1)
                filtered_dfs[subset_name] = self.df_subset[mask].reset_index(drop=True)

            # ----- build list of (subset × classifier_set) jobs ----------
            jobs = []
            if self.classifiers_to_compare is not None:
                for subset_name in self.subsets:
                    for cls_name, cls_config in self.classifiers_to_compare.items():
                        jobs.append(
                            (
                                filtered_dfs[subset_name],
                                subset_name,
                                cls_name,
                                cls_config,
                            )
                        )
            else:
                # No user config → each job will use the built-in default
                for subset_name in self.subsets:
                    jobs.append(
                        (
                            filtered_dfs[subset_name],
                            subset_name,
                            None,
                            None,
                        )
                    )

            total = len(jobs)
            self.progress.emit(f"Running {total} training job(s) with up to {self.n_jobs} parallel worker(s)…\nSee console for further details")

            # ----- dispatch parallel jobs via joblib (process-based) -----
            results = Parallel(
                n_jobs=self.n_jobs,
                backend="loky",
            )(
                delayed(_parallel_train_job)(
                    df_filt,
                    sname,
                    csname,
                    csconfig,
                    self.output_dir,
                )
                for df_filt, sname, csname, csconfig in jobs
            )

            self.finished.emit(results)
        except Exception as e:
            error_msg = f"Error training classifiers: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.error.emit(error_msg)


class SpectrumViewer(QDialog):
    """Dialog for viewing spectrum details."""

    def __init__(self, spectrum_data, meta_data, parent=None):
        super().__init__(parent)
        self.spectrum_data = spectrum_data
        self.meta_data = meta_data

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

        top_widget.setLayout(top_layout)
        layout.addWidget(top_widget)

        # Bottom section: Spectrum visualization
        spectrum_group = QGroupBox("Spectrum")
        spectrum_layout = QVBoxLayout()

        try:
            import matplotlib

            matplotlib.use("Qt5Agg")
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            from matplotlib.figure import Figure

            # Create matplotlib figure
            fig = Figure(figsize=(10, 4))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            # Add matplotlib toolbar
            toolbar = NavigationToolbar(canvas, self)
            spectrum_layout.addWidget(toolbar)

            # Plot spectrum
            if self.spectrum_data is not None:
                # spectrum_data is a 2D array with [0,:] = m/z and [1,:] = intensity
                if isinstance(self.spectrum_data, Spectrum):
                    mz_values = self.spectrum_data.peaks.mz
                    intensity_values = self.spectrum_data.peaks.intensities
                elif isinstance(self.spectrum_data, (list, tuple)) and len(self.spectrum_data) == 2:
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
    """Main GUI for AnnoMe Classification."""

    EXCLUDE_COLS = frozenset({"source", "name", "smiles", "inchikey", "ionmode", "precursor_mz", "adduct", "label", "num_peaks", "embeddings", "predicted_class"})

    def __init__(self, config_path=None, auto_start_training=False):
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
        self.subset_results = {}  # combined_key (classifier_set // subset) -> {df_train, df_validation, df_inference, df_metrics, trained_classifiers, long_table, pivot_table, filter_fn}
        self.subset_filter_functions = {}  # subset_name -> filter_function
        self.classifiers_config = None  # ML classifiers configuration
        self.pending_config_data = None  # For load_full_configuration workflow
        self._editing_subset_index = None  # Track which subset is being edited
        self.expected_embeddings_hash = None  # Expected hash from loaded configuration file
        self.expected_embeddings_path = None  # Expected path from loaded configuration file
        self._auto_start_training = auto_start_training  # Automatically start training after config load
        self._results_combined_summary = None  # Full combined_summary for the results histogram

        set_random_seeds(42)

        self.init_ui()

        # Schedule automatic config loading after the window is shown
        if config_path is not None:
            QTimer.singleShot(200, lambda: self.load_full_configuration(file_path=config_path))

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
        self.setWindowTitle(f"AnnoMe Classification (v{version})")
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

        # Create tab widget with tabs on the left
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.West)

        # Create sections as tab pages
        self.section1 = QWidget()
        self.init_section1()
        self.tab_widget.addTab(self.section1, "1. Load MGF Files")

        self.section2 = QWidget()
        self.init_section2()
        self.tab_widget.addTab(self.section2, "2. Generate Embeddings")

        self.section3 = QWidget()
        self.init_section3()
        self.tab_widget.addTab(self.section3, "3. Define Metadata Subsets")

        self.section4 = QWidget()
        self.init_section4()
        self.tab_widget.addTab(self.section4, "4. Train Classifiers")

        self.section5 = QWidget()
        self.init_section5()
        self.tab_widget.addTab(self.section5, "5. Inspect Classification Results")

        self.section6 = QWidget()
        self.init_section6()
        self.tab_widget.addTab(self.section6, "6. Inspect Individual Spectra")

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tab_widget)
        main_widget.setLayout(main_layout)

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
            <h4>Table Columns:</h4>
            <ul>
                <li><b># Parsed:</b> Total entries parsed from the MGF file</li>
                <li><b># With Embeddings:</b> Entries for which embeddings were calculated (shown after Step 2)</li>
                <li><b># Entries:</b> Currently available entries (after filtering)</li>
                <li><b># Unique SMILES:</b> Number of unique chemical structures</li>
            </ul>
            <h4>Steps:</h4>
            <ol>
                <li>Click <b>Load MGF File(s)</b> to select one or more MGF files</li>
                <li>Review statistics (parsed entries, unique SMILES)</li>
                <li>Assign appropriate type to each file using dropdown</li>
                <li>Remove unwanted files with <b>Remove Selected File(s)</b></li>
            </ol>
            <p><b>Tip:</b> You need both training data (train-relevant and train-other) and validation data for proper classifier training.</p>
            <p><b>Note:</b> Not all spectra will be used, due to limitations in the embedding generation. Once the embeddings have been generated, only those spectra will remain, for which an embedding was successfully calculated. The table will be updated to show these counts.</p>
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
        self.file_table.setColumnCount(6)
        self.file_table.setHorizontalHeaderLabels(["File Name", "# Parsed", "# With Embeddings", "# Entries", "# Unique SMILES", "Type"])
        self.file_table.horizontalHeader().setStretchLastSection(False)
        self.file_table.setSelectionBehavior(QAbstractItemView.SelectRows)

        header = self.file_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.Interactive)
        header.setSectionResizeMode(3, QHeaderView.Interactive)
        header.setSectionResizeMode(4, QHeaderView.Interactive)
        header.setSectionResizeMode(5, QHeaderView.Interactive)

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

        # Set up scroll area for this section
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)

        section_layout = QVBoxLayout()
        section_layout.addWidget(scroll)
        self.section1.setLayout(section_layout)

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
            <p>Convert MSMS spectra to numerical vectors using the MS2DeepScore [1] deep learning model.</p>
                          
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
            <br>
            <h4>References:</h4>
            <p>[1] MS2DeepScore - a novel deep learning similarity measure to compare tandem mass spectra<br> Florian Huber, Sven van der Burg, Justin J.J. van der Hooft, Lars Ridder, 13, Article number: 84 (2021), Journal of Cheminformatics, doi: <a href = \"https://doi.org/10.1186/s13321-021-00558-4\">https://doi.org/10.1186/s13321-021-00558-4</a></p>
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

        # Set up scroll area for this section
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)

        section_layout = QVBoxLayout()
        section_layout.addWidget(scroll)
        self.section2.setLayout(section_layout)

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
            <p><b>Examples:</b></p><br>
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
            <br>
            <p><b>Tip:</b>Save and load pre-defined sets using the 'Export Subsets' and 'Import Subsets' buttons in the right, bottom corner</p>
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

        # Set up scroll area for this section
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)

        section_layout = QVBoxLayout()
        section_layout.addWidget(scroll)
        self.section3.setLayout(section_layout)

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
                <li>metadata subsets are defined (Section 3)</li>
            </ul>
            <h4>Configuration:</h4>
            <ol>
                <li><b>Classifier Configuration:</b> Define which ML algorithms to use (or use defaults)</li>
            </ol>
            <h4>Writing Classifier Code:</h4>
            <p>The classifier configuration must generate a Python dictionary named <code>classifiers_to_compare</code> with the following structure:</p>
            <ul>
                <li><b>Keys:</b> Classifier set names (strings, e.g., "default_set", "tree_based")</li>
                <li><b>Values:</b> Dictionaries containing classifier configuration with required fields</li>
            </ul>
            <p><b>Required fields for each classifier set dictionary:</b></p>
            <ul>
                <li><b>"description":</b> String describing the classifier set</li>
                <li><b>"classifiers":</b> Dictionary mapping classifier names to initialized sklearn classifier objects</li>
                <li><b>"cross_validation":</b> sklearn cross-validation object (e.g., StratifiedKFold), or None for default (StratifiedKFold with 10 splits). Each classifier will be trained n times, where n is the number of splits defined in the cross-validation object (e.g., n_splits=10 means each classifier is trained 10 times)</li>
                <li><b>"min_prediction_threshold":</b> Integer threshold for minimum predictions required</li>
            </ul>
            <p><b>Example structure:</b></p>
            <pre>
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

classifiers_to_compare = {
    "default_set": {
        "description": "Default set of classifiers",
        "classifiers": {
            "Random Forest": RandomForestClassifier(n_estimators=100),
            "SVM": SVC(kernel='rbf'),
            "Logistic Regression": LogisticRegression()
        },
        "cross_validation": StratifiedKFold(n_splits=10, shuffle=True),
        "min_prediction_threshold": 100
    },
    "tree_based": {
        "description": "Tree-based methods",
        "classifiers": {
            "Random Forest": RandomForestClassifier(),
            "Extra Trees": ExtraTreesClassifier()
        },
        "cross_validation": None,  # Uses default
        "min_prediction_threshold": 50
    }
}
            </pre>
            <p><b>Available sklearn algorithms:</b> Any classifier from scikit-learn can be used, including RandomForestClassifier, SVC, LogisticRegression, KNeighborsClassifier, GradientBoostingClassifier, MLPClassifier, DecisionTreeClassifier, etc.</p>
            <h4>Training Process:</h4>
            <ul>
                <li>Splits data into training/validation sets</li>
                <li>Trains multiple classifier algorithms</li>
                <li>Generates predictions and performance metrics</li>
                <li>Saves trained models and results</li>
            </ul>
            <h4>Prediction Logic:</h4>
            <p>With <b>n</b> classifier objects and a cross-validation fold of <b>k</b>, there will be <b>n × k</b> models total. For prediction:</p>
            <ul>
                <li>If at least <b>min_prediction_threshold</b> of these models predict a "relevant" class, the spectrum is classified as <b>relevant</b></li>
                <li>Otherwise, the spectrum is classified as <b>other</b></li>
            </ul>
            <p><b>Example:</b> With 3 classifiers and StratifiedKFold(n_splits=10), you get 3 × 10 = 30 models. If min_prediction_threshold=15, a spectrum needs at least 15 models to predict "relevant" to be classified as relevant.</p>
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
            "Leave empty to use default configuration, or enter Python dict code here. The variable 'classifiers_to_compare' as a dictionary of classifier sets must be generated. Each key is a set name, and each value is a dict of sklearn classifiers."
        )
        self.classifiers_config_text.setMinimumHeight(150)
        font = self.classifiers_config_text.font()
        font.setFamily("Courier New")
        self.classifiers_config_text.setFont(font)
        self.classifiers_config_text.setEnabled(False)  # Disabled until embeddings are generated
        layout.addWidget(self.classifiers_config_text, 1)  # Stretch to fill available space

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

        # Parallel workers control
        parallel_layout = QHBoxLayout()
        parallel_layout.addWidget(QLabel("Parallel Workers:"))
        self.n_jobs_spinbox = QSpinBox()
        self.n_jobs_spinbox.setMinimum(1)
        self.n_jobs_spinbox.setMaximum(multiprocessing.cpu_count() * 2)
        self.n_jobs_spinbox.setValue(multiprocessing.cpu_count())
        self.n_jobs_spinbox.setToolTip("Number of parallel processes for training.\nEach (subset × classifier-set) combination runs as a separate process.\nDefaults to the number of CPU cores.")
        parallel_layout.addWidget(self.n_jobs_spinbox)
        parallel_layout.addStretch()
        layout.addLayout(parallel_layout)

        # Train button
        self.train_btn = QPushButton("Train and Classify")
        self.train_btn.clicked.connect(self.train_classifiers_clicked)
        self.train_btn.setEnabled(False)
        layout.addWidget(self.train_btn)

        # Training status
        self.training_status = QLabel("Status: Not started")
        layout.addWidget(self.training_status)

        # Vertical splitter between the upper config area and the log tabs
        # Wrap everything above in a top widget, log tabs in the bottom widget
        upper_widget = QWidget()
        upper_widget.setLayout(layout)

        self.training_log_tabs = QTabWidget()
        self.training_log_tabs.setTabPosition(QTabWidget.North)
        self.training_log_tabs.setVisible(False)  # hidden until first run

        self.section4_splitter = QSplitter(Qt.Vertical)
        self.section4_splitter.addWidget(upper_widget)
        self.section4_splitter.addWidget(self.training_log_tabs)
        self.section4_splitter.setStretchFactor(0, 3)
        self.section4_splitter.setStretchFactor(1, 2)

        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(self.section4_splitter)
        controls.setLayout(controls_layout)
        main_layout.addWidget(controls, 3)  # 75% width

        content.setLayout(main_layout)

        # Set up scroll area for this section
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)

        section_layout = QVBoxLayout()
        section_layout.addWidget(scroll)
        self.section4.setLayout(section_layout)

    def init_section5(self):
        """Initialize the results inspection section."""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(6, 6, 6, 6)

        # Export buttons at the top
        button_layout = QHBoxLayout()
        export_results_btn = QPushButton("Export Results to Excel")
        export_results_btn.clicked.connect(self.export_results)
        button_layout.addWidget(export_results_btn)

        export_classifiers_btn = QPushButton("Export Classifiers")
        export_classifiers_btn.clicked.connect(self.export_classifiers)
        button_layout.addWidget(export_classifiers_btn)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        # Horizontal splitter: table on left, histogram on right (user-resizable)
        self.results_splitter = QSplitter(Qt.Horizontal)
        self.results_splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Left panel – results table
        self.results_table = QTableWidget()
        self.results_table.setSortingEnabled(True)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.itemSelectionChanged.connect(self.on_results_table_selection_changed)
        self.results_splitter.addWidget(self.results_table)

        # Right panel – vertically scrollable histogram
        self._results_histogram_available = False
        try:
            import matplotlib

            matplotlib.use("Qt5Agg")
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            from matplotlib.figure import Figure

            hist_panel = QWidget()
            hist_panel_layout = QVBoxLayout()
            hist_panel_layout.setContentsMargins(0, 0, 0, 0)

            self.results_figure = Figure()
            self.results_canvas = FigureCanvas(self.results_figure)

            self.results_hist_toolbar = NavigationToolbar(self.results_canvas, self)
            hist_panel_layout.addWidget(self.results_hist_toolbar)

            # QScrollArea with vertical-only scroll – canvas height is managed
            # manually in update_results_histogram so all subplots are visible.
            self.results_hist_scroll = QScrollArea()
            self.results_hist_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.results_hist_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.results_hist_scroll.setWidgetResizable(False)  # we control canvas size
            self.results_hist_scroll.setWidget(self.results_canvas)
            hist_panel_layout.addWidget(self.results_hist_scroll, 1)

            hist_panel.setLayout(hist_panel_layout)
            self.results_splitter.addWidget(hist_panel)
            self._results_histogram_available = True
        except ImportError:
            placeholder = QLabel("Matplotlib not available for histogram visualisation")
            placeholder.setAlignment(Qt.AlignCenter)
            self.results_splitter.addWidget(placeholder)

        # Default split: 50 % table / 50 % histogram
        self.results_splitter.setStretchFactor(0, 1)
        self.results_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(self.results_splitter, 1)

        self.section5.setLayout(main_layout)

    def init_section6(self):
        """Initialize the individual spectra inspection section."""
        content = QWidget()
        main_layout = QVBoxLayout()

        # Main horizontal splitter for all four panels
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Panel 1: Subset selector and MGF file list (leftmost)
        file_widget = QWidget()
        file_layout = QVBoxLayout()

        # Add subset selector (list)
        file_layout.addWidget(QLabel("Subset:"))
        self.spectrum_subset_list = QListWidget()
        self.spectrum_subset_list.itemSelectionChanged.connect(self.on_subset_selected_for_spectra)
        file_layout.addWidget(self.spectrum_subset_list)

        file_layout.addWidget(QLabel("MGF Files:"))
        self.spectrum_file_list = QListWidget()
        self.spectrum_file_list.itemSelectionChanged.connect(self.on_spectrum_file_selected)
        file_layout.addWidget(self.spectrum_file_list)
        file_widget.setLayout(file_layout)
        main_splitter.addWidget(file_widget)

        # Panel 2: Spectrum table
        table_widget = QWidget()
        table_layout = QVBoxLayout()
        table_layout.addWidget(QLabel("Spectra:"))
        self.spectrum_table = QTableWidget()
        self.spectrum_table.setColumnCount(8)
        self.spectrum_table.setHorizontalHeaderLabels(["ID", "m/z", "RT (s)", "CE", "Source", "Frag. Method", "Ion Mode", "Classification"])
        self.spectrum_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.spectrum_table.setSortingEnabled(True)
        self.spectrum_table.itemSelectionChanged.connect(self.on_spectrum_selected)

        # Set column resize modes
        header = self.spectrum_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.Interactive)
        header.setSectionResizeMode(3, QHeaderView.Interactive)
        header.setSectionResizeMode(4, QHeaderView.Interactive)
        header.setSectionResizeMode(5, QHeaderView.Interactive)
        header.setSectionResizeMode(6, QHeaderView.Interactive)
        header.setSectionResizeMode(7, QHeaderView.Interactive)

        table_layout.addWidget(self.spectrum_table)
        table_widget.setLayout(table_layout)
        main_splitter.addWidget(table_widget)

        # Panel 3: Metadata with classification on top
        meta_widget = QWidget()
        meta_layout = QVBoxLayout()
        meta_layout.addWidget(QLabel("Metadata:"))

        # Classification label on top
        self.spectrum_classification_label = QLabel("Classification: -")
        self.spectrum_classification_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #f0f0f0;")
        self.spectrum_classification_label.setAlignment(Qt.AlignCenter)
        meta_layout.addWidget(self.spectrum_classification_label)

        # Metadata text
        self.spectrum_meta_text = QTextEdit()
        self.spectrum_meta_text.setReadOnly(True)
        self.spectrum_meta_text.setHtml("<p>Select a spectrum to view details</p>")
        meta_layout.addWidget(self.spectrum_meta_text)
        meta_widget.setLayout(meta_layout)
        main_splitter.addWidget(meta_widget)

        # Panel 4: Spectrum plot (rightmost)
        spectrum_widget = QWidget()
        spectrum_layout = QVBoxLayout()
        spectrum_layout.addWidget(QLabel("MS/MS Spectrum:"))

        try:
            import matplotlib

            matplotlib.use("Qt5Agg")
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            from matplotlib.figure import Figure

            self.spectrum_figure = Figure(figsize=(6, 4))
            self.spectrum_canvas = FigureCanvas(self.spectrum_figure)
            self.spectrum_ax = self.spectrum_figure.add_subplot(111)
            self.spectrum_ax.text(0.5, 0.5, "Select a spectrum to view", ha="center", va="center", fontsize=12)
            self.spectrum_ax.set_xlim(0, 1)
            self.spectrum_ax.set_ylim(0, 1)

            # Add matplotlib toolbar
            self.spectrum_toolbar = NavigationToolbar(self.spectrum_canvas, self)
            spectrum_layout.addWidget(self.spectrum_toolbar)

            spectrum_layout.addWidget(self.spectrum_canvas, 1)  # Stretch factor to fill vertical space
        except ImportError:
            spectrum_label = QLabel("Matplotlib not available for spectrum visualization")
            spectrum_label.setAlignment(Qt.AlignCenter)
            spectrum_layout.addWidget(spectrum_label, 1)  # Stretch factor to fill vertical space

        spectrum_widget.setLayout(spectrum_layout)
        main_splitter.addWidget(spectrum_widget)

        # Set stretch factors: file list (1), table (2), metadata (2), spectrum (2)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 2)
        main_splitter.setStretchFactor(2, 2)
        main_splitter.setStretchFactor(3, 2)

        # Add splitter with expanding size policy to fill vertical space
        main_layout.addWidget(main_splitter, stretch=1)

        # Export button at bottom (no stretch)
        export_individual_btn = QPushButton("Export Selected Spectrum to Excel")
        export_individual_btn.clicked.connect(self.export_individual_spectrum)
        main_layout.addWidget(export_individual_btn, stretch=0)

        # Set layout directly on section6 (no scroll area needed, splitter handles it)
        self.section6.setLayout(main_layout)

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
                    meta_keys = {k for k in entry.keys() if k not in ["peaks"]}
                    self.all_meta_keys.update(meta_keys)

                filename = os.path.basename(file_path)
                self.mgf_files[filename] = {
                    "path": file_path,
                    "entries": entries,
                    "num_entries": len(entries),
                    "num_parsed": len(entries),
                    "num_with_embeddings": None,
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

            # Number of parsed entries
            self.file_table.setItem(row, 1, QTableWidgetItem(str(data.get("num_parsed", data["num_entries"]))))

            # Number with embeddings
            num_with_emb = data.get("num_with_embeddings")
            emb_text = str(num_with_emb) if num_with_emb is not None else "-"
            self.file_table.setItem(row, 2, QTableWidgetItem(emb_text))

            # Number of entries (after filtering)
            self.file_table.setItem(row, 3, QTableWidgetItem(str(data["num_entries"])))

            # Unique SMILES
            self.file_table.setItem(row, 4, QTableWidgetItem(str(data["unique_smiles"])))

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

            self.file_table.setCellWidget(row, 5, type_combo)

        # Auto-resize columns to contents
        self.file_table.resizeColumnsToContents()

    def on_file_type_changed(self, filename, new_type):
        """Handle file type change."""
        if filename in self.mgf_files:
            self.mgf_files[filename]["type"] = new_type
            # Update the combo box color
            for row in range(self.file_table.rowCount()):
                if self.file_table.item(row, 0).text() == filename:
                    combo = self.file_table.cellWidget(row, 5)
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
                    meta_keys = {k for k in entry.keys() if k not in ["peaks"]}
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
                        meta_keys = {k for k in entry.keys() if k not in ["peaks"]}
                        self.all_meta_keys.update(meta_keys)

                    filename = file_info["filename"]
                    self.mgf_files[filename] = {
                        "path": mgf_path,
                        "entries": entries,
                        "num_entries": len(entries),
                        "num_parsed": len(entries),
                        "num_with_embeddings": None,
                        "unique_smiles": len(smiles_set),
                        "type": file_type,
                    }
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
        all_keys = sorted([col for col in self.df_embeddings.columns if col not in self.EXCLUDE_COLS])

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
        all_keys = sorted([col for col in self.df_embeddings.columns if col not in self.EXCLUDE_COLS])

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

    def _count_subset_matches(self, subset_expr):
        """Count matches for a subset expression across all embeddings, grouped by file type.

        Compiles the expression once and uses vectorized aggregation for performance.
        """
        type_counts = defaultdict(int)
        if self.df_embeddings is None:
            return type_counts

        file_type_map = {filename: data["type"] for filename, data in self.mgf_files.items()}
        meta_cols = [col for col in self.df_embeddings.columns if col not in self.EXCLUDE_COLS]
        compiled = compile(subset_expr, "<string>", "eval")
        eval_globals = {"abs": abs, "float": float, "int": int, "str": str}

        def eval_row(row):
            meta = {col: row[col] for col in meta_cols}
            try:
                return bool(eval(compiled, {**eval_globals, "meta": meta}))
            except Exception:
                return False

        mask = self.df_embeddings.apply(eval_row, axis=1)
        matched_sources = self.df_embeddings.loc[mask, "source"]
        for source, count in matched_sources.value_counts().items():
            file_type = file_type_map.get(source, "unknown")
            type_counts[file_type] += count

        return type_counts

    def check_subset_syntax(self):
        """Check if the subset syntax is valid."""
        subset_expr = self.subset_input.text().strip()

        if not subset_expr:
            QMessageBox.warning(self, "Warning", "Please enter a subset expression")
            return

        if self.df_embeddings is None:
            QMessageBox.warning(self, "Warning", "Please generate embeddings first (Section 2)")
            return

        # Test with actual metadata from the first row of embeddings
        meta_cols = [col for col in self.df_embeddings.columns if col not in self.EXCLUDE_COLS]
        test_row = self.df_embeddings.iloc[0]
        test_meta = {col: test_row[col] for col in meta_cols}

        try:
            # Try to compile and evaluate
            compiled = compile(subset_expr, "<string>", "eval")
            result = eval(compiled, {"meta": test_meta, "abs": abs, "float": float, "int": int, "str": str})
            QMessageBox.information(self, "Success", f"Subset syntax is valid! (test result on first spectrum: {result})")
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

        # Test the subset with actual data from the first row of embeddings
        meta_cols = [col for col in self.df_embeddings.columns if col not in self.EXCLUDE_COLS]
        test_row = self.df_embeddings.iloc[0]
        test_meta = {col: test_row[col] for col in meta_cols}
        try:
            compiled = compile(subset_expr, "<string>", "eval")
            eval(compiled, {"meta": test_meta, "abs": abs, "float": float, "int": int, "str": str})
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Invalid subset:\n{str(e)}")
            return

        # Count matches for each type using the embeddings dataframe
        type_counts = self._count_subset_matches(subset_expr)

        # Add or replace subset depending on editing mode
        new_subset = {"name": subset_name, "expression": subset_expr, "counts": type_counts}
        if self._editing_subset_index is not None:
            self.subsets[self._editing_subset_index] = new_subset
            self._editing_subset_index = None
        else:
            self.subsets.append(new_subset)

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

        # Store the index being edited (subset stays in list until replaced)
        self._editing_subset_index = row

        # Load values into text fields
        subset_name = self.subsets[row].get("name", "")
        subset_expr = self.subsets[row]["expression"]
        self.subset_name_input.setText(subset_name)
        self.subset_input.setText(subset_expr)

        QMessageBox.information(self, "Edit Subset", "Subset loaded into text fields for editing.\nModify and click 'Add Subset' to save changes.\nThe original subset is preserved until replaced.")

    def delete_selected_subset(self):
        """Delete the selected subset(s)."""
        selected_rows = set(item.row() for item in self.subset_table.selectedItems())

        if not selected_rows:
            QMessageBox.warning(self, "Warning", "Please select a subset to delete")
            return

        # Adjust editing index if a deletion would invalidate it
        if self._editing_subset_index is not None:
            if self._editing_subset_index in selected_rows:
                # The subset being edited is being deleted — cancel editing
                self._editing_subset_index = None
            else:
                # Adjust for deletions before the editing index
                shift = sum(1 for r in selected_rows if r < self._editing_subset_index)
                self._editing_subset_index -= shift

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
        """Generate embeddings from loaded datasets or load from cache if available."""
        if not self.mgf_files:
            QMessageBox.warning(self, "Warning", "No MGF files loaded")
            return

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

        # Set busy cursor during embedding generation
        QApplication.setOverrideCursor(Qt.WaitCursor)

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
        QApplication.restoreOverrideCursor()
        self.progress_dialog.close()

        # Verify that the number of rows matches the total number of MGF entries
        total_mgf_entries = sum(data["num_entries"] for data in self.mgf_files.values())

        if len(df) != total_mgf_entries:
            self.embedding_status.setText("Status: Error - Data integrity check failed ✗")
            _integrity_msg = QMessageBox(self)
            _integrity_msg.setIcon(QMessageBox.Warning)
            _integrity_msg.setWindowTitle("Warning")
            _integrity_msg.setText(
                f"Data integrity check failed:\nExpected {total_mgf_entries} entries from parsing the MGF files, but got {len(df)} entries in embeddings. Only those spectra with valid embeddings are included further."
            )
            _ok_btn = _integrity_msg.addButton(QMessageBox.Ok)
            self._exec_with_autoclose(_integrity_msg, _ok_btn)

            # get list of all ids to use from df
            ids_to_use = set(df["AnnoMe_internal_ID"].tolist())
            # remove all spectra where a corresponding internal id is not present in the embeddings
            for filename, mgf_file in self.mgf_files.items():
                original_count = mgf_file["num_entries"]
                mgf_file["entries"] = [entry for i, entry in enumerate(mgf_file["entries"]) if entry["AnnoMe_internal_ID"] in ids_to_use]
                mgf_file["num_entries"] = len(mgf_file["entries"])
                mgf_file["num_with_embeddings"] = len(mgf_file["entries"])
                # recalculate unique smiles
                smiles_set = set()
                for entry in mgf_file["entries"]:
                    smiles = entry.get("smiles", "")
                    if smiles:
                        smiles_set.add(smiles)
                mgf_file["unique_smiles"] = len(smiles_set)

                print(
                    f"{mgf_file['path']} had {original_count} entries, but it was possible to calculate the MS2DeepScore embeddings for only {mgf_file['num_entries']} entries. The other entries will be removed."
                )
        else:
            # All embeddings calculated successfully - update num_with_embeddings for all files
            for filename, mgf_file in self.mgf_files.items():
                mgf_file["num_with_embeddings"] = mgf_file["num_entries"]

        # Update the file table to reflect the embedding counts
        self.update_file_table()

        # All checks passed, proceed normally
        self.df_embeddings = df
        self.embedding_status.setText(f"Status: Embeddings generated ({len(df)} entries) - Data integrity verified ✓")
        self.train_btn.setEnabled(True)
        self.generate_embeddings_btn.setEnabled(True)

        # Enable classifier configuration inputs
        self.classifiers_config_text.setEnabled(True)
        self.load_default_classifiers_btn.setEnabled(True)

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
            QMessageBox.information(
                self,
                "Success",
                f"Embeddings generated and saved to:\n{pickle_file}\n\n"
                f"💡 Tip: Save your project configuration via the menu bar (File → Save Project)\n"
                f"to easily skip the embedding generation step next time you load this project.",
            )

    def on_embedding_error(self, error_msg):
        """Handle embedding generation error."""
        QApplication.restoreOverrideCursor()
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
from sklearn.model_selection import StratifiedKFold

classifiers_to_compare = {
    "default_set": {
        "description": "Default set of diverse classifiers for comparison",
        "classifiers": {
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
        },
        "cross_validation": StratifiedKFold(n_splits=10, random_state=42, shuffle=True),  # Optional: if not specified, uses StratifiedKFold with 10 splits
        "min_prediction_threshold": 100,
    }
}

# Example with multiple classifier sets for comparison:
# classifiers_to_compare = {
#     "knn_variants": {
#         "description": "KNNs with different k values are used", 
#         "classifiers": {
#             "KNN n=3": KNeighborsClassifier(3),
#             "KNN n=5": KNeighborsClassifier(5),
#             "KNN n=10": KNeighborsClassifier(10),
#         },
#         "cross_validation": StratifiedKFold(n_splits=5, random_state=42, shuffle=True),  # 5-fold CV
#         "min_prediction_threshold": 12,
#     },
#     "tree_based": {
#         "description": "Tree-based methods are used", 
#         "classifiers": {
#             "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
#             "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42),
#             "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
#         },
#         # "cross_validation": None,  # Optional: if omitted or None, uses default StratifiedKFold with 10 splits
#         "min_prediction_threshold": 25,
#     },
#     "linear_models": {
#         "description": "Linear models are used", 
#         "classifiers": {
#             "Linear SVM": SVC(kernel="linear", C=0.025, random_state=42),
#             "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
#             "Ridge Classifier": RidgeClassifier(random_state=42),
#         },
#         "cross_validation": StratifiedKFold(n_splits=3, random_state=42, shuffle=True),  # 3-fold CV
#         "min_prediction_threshold": 7,
#     }
# }"""
        self.classifiers_config_text.setPlainText(default_config)

    def save_classifier_configuration(self):
        """Save the classifier configuration and threshold to JSON file."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Classifier Configuration", "", "JSON Files (*.json);;All Files (*)")

        if not file_path:
            return

        try:
            config_data = {"classifier_configuration": self.classifiers_config_text.toPlainText()}

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
            }

            # Save MGF file paths and types
            for filename, data in self.mgf_files.items():
                config_data["mgf_files"].append({"path": data["path"], "type": data["type"]})

            # Calculate and save embeddings pickle file hash if it exists
            output_dir = self.output_dir_input.text()
            if output_dir:
                pickle_file = os.path.join(output_dir, "df_embeddings.pkl")
                if os.path.exists(pickle_file):
                    file_hash = calculate_file_hash(pickle_file)
                    if file_hash:
                        config_data["embeddings_pickle_path"] = pickle_file
                        config_data["embeddings_pickle_hash"] = file_hash
                        print(f"Saved embeddings hash: {file_hash}")

            with open(file_path, "w") as f:
                json.dump(config_data, f, indent=4)

            QMessageBox.information(self, "Success", f"Full configuration saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save configuration:\n{str(e)}")

    def load_full_configuration(self, file_path=None):
        """Load complete workflow configuration and automatically execute workflow steps.

        Parameters
        ----------
        file_path : str, optional
            Path to the configuration JSON file. When *None* (default) a file-open
            dialog is shown so the user can pick the file interactively.
        """
        if file_path is None:
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
            # Don't clear expected hash yet - we'll set it from config data below

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
                            meta_keys = {k for k in entry.keys() if k not in ["peaks"]}
                            self.all_meta_keys.update(meta_keys)

                        filename = os.path.basename(file_path_mgf)
                        self.mgf_files[filename] = {
                            "path": file_path_mgf,
                            "entries": entries,
                            "num_entries": len(entries),
                            "num_parsed": len(entries),
                            "num_with_embeddings": None,
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

            # Step 2.5: Store expected embeddings hash and path from configuration
            if "embeddings_pickle_hash" in config_data and "embeddings_pickle_path" in config_data:
                self.expected_embeddings_hash = config_data["embeddings_pickle_hash"]
                self.expected_embeddings_path = config_data["embeddings_pickle_path"]
                print(f"Stored expected embeddings hash from configuration: {self.expected_embeddings_hash[:16]}...")

            # Step 3: Try to load embeddings from pickle if hash matches, otherwise generate
            embeddings_loaded = False
            if self.mgf_files:
                # Check if configuration includes embeddings pickle hash
                if "embeddings_pickle_hash" in config_data and "embeddings_pickle_path" in config_data:
                    pickle_path = config_data["embeddings_pickle_path"]
                    expected_hash = config_data["embeddings_pickle_hash"]

                    # Verify the pickle file exists and hash matches
                    if os.path.exists(pickle_path):
                        actual_hash = calculate_file_hash(pickle_path)

                        if actual_hash == expected_hash:
                            try:
                                # Load embeddings from pickle
                                print(f"Loading embeddings from cached pickle file: {pickle_path}")
                                df = pd.read_pickle(pickle_path)

                                # Verify data integrity
                                total_mgf_entries = sum(data["num_entries"] for data in self.mgf_files.values())

                                if len(df) == total_mgf_entries:
                                    # Successfully loaded embeddings
                                    self.df_embeddings = df
                                    self.embedding_status.setText(f"Status: Embeddings loaded from cache ({len(df)} entries) ✓")
                                    self.train_btn.setEnabled(True)
                                    self.generate_embeddings_btn.setEnabled(True)
                                    self.classifiers_config_text.setEnabled(True)
                                    self.load_default_classifiers_btn.setEnabled(True)
                                    embeddings_loaded = True

                                    # Clear expected hash after successful load
                                    self.expected_embeddings_hash = None
                                    self.expected_embeddings_path = None

                                    # Navigate to embeddings section
                                    self.go_to_section(self.section2)

                                    QMessageBox.information(self, "Embeddings Loaded", f"Embeddings loaded from cache:\n{pickle_path}\n\nHash verified successfully. Skipping embedding generation.")

                                    # Continue with workflow
                                    self.continue_loading_configuration()
                                else:
                                    print(f"Warning: Embeddings count mismatch. Expected {total_mgf_entries}, got {len(df)}. Regenerating embeddings.")
                            except Exception as e:
                                print(f"Error loading embeddings from pickle: {e}. Will regenerate embeddings.")
                        else:
                            print(f"Embeddings pickle hash mismatch. Expected: {expected_hash}, Got: {actual_hash}. Will regenerate embeddings.")
                    else:
                        print(f"Embeddings pickle file not found at: {pickle_path}. Will regenerate embeddings.")

                # If embeddings weren't loaded, generate them
                if not embeddings_loaded:
                    # Navigate to embeddings section
                    self.go_to_section(self.section2)
                    self.generate_embeddings_clicked()
            else:
                self.pending_config_data = None

        except Exception as e:
            self.pending_config_data = None
            self.expected_embeddings_hash = None
            self.expected_embeddings_path = None
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
                for subset_info in config_data["subsets"]:
                    subset_name = subset_info.get("name", "Unnamed")
                    subset_expr = subset_info["expression"]

                    # Count matches for each type using the embeddings dataframe
                    type_counts = self._count_subset_matches(subset_expr)

                    # Add to subsets list with recalculated counts
                    self.subsets.append({"name": subset_name, "expression": subset_expr, "counts": type_counts})

                self.update_subset_table()

            # Step 6: Load classifier configuration
            if "classifier_configuration" in config_data:
                self.classifiers_config_text.setPlainText(config_data["classifier_configuration"])

            # Navigate to training section
            self.go_to_section(self.section4)

            _ready_msg = QMessageBox(self)
            _ready_msg.setIcon(QMessageBox.Information)
            _ready_msg.setWindowTitle("Success")
            _ready_msg.setText(
                f"Configuration loaded and workflow executed:\n"
                f"- Loaded {len(self.mgf_files)} MGF files\n"
                f"- Generated embeddings ({len(self.df_embeddings)} entries)\n"
                f"- Loaded {len(self.subsets)} subsets\n"
                f"- Loaded classifier configuration\n\n"
                f"Ready to train classifiers!"
            )
            _ready_ok = _ready_msg.addButton(QMessageBox.Ok)
            self._exec_with_autoclose(_ready_msg, _ready_ok)

            # Automatically start training if requested via -startTraining flag
            if self._auto_start_training:
                print("Auto-starting training as requested by -startTraining flag...")
                QTimer.singleShot(100, self.train_classifiers_clicked)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to complete configuration loading:\n{str(e)}")
        finally:
            self.pending_config_data = None

    # ---------------------------------------------------------------------- #
    # Utility helpers                                                          #
    # ---------------------------------------------------------------------- #

    def _exec_with_autoclose(self, msg, default_button, timeout_ms=30000):
        """Execute a QMessageBox and auto-click *default_button* after *timeout_ms* ms.

        A countdown (updated every second) is appended to the dialog text so the
        user can see how long they have to react.
        """
        interval_ms = 1000
        remaining = [timeout_ms // interval_ms]
        original_text = msg.text()

        def _tick():
            remaining[0] -= 1
            if remaining[0] <= 0:
                countdown_timer.stop()
                default_button.click()
            else:
                msg.setText(f"{original_text}\n\n(Auto-closing with default option in {remaining[0]}s)")

        countdown_timer = QTimer(msg)
        countdown_timer.setInterval(interval_ms)
        countdown_timer.timeout.connect(_tick)
        msg.setText(f"{original_text}\n\n(Auto-closing with default option in {remaining[0]}s)")
        countdown_timer.start()
        msg.exec_()

    def show_about(self):
        """Show about dialog with GUI description and version."""
        version = self.get_version()
        about_text = f"""<h2>AnnoMe Classification</h2>
        <p><b>Version:</b> {version}</p>
        
        <p>This application provides a complete workflow for classifying compounds using mass spectrometry data:</p>
        
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
                # Evaluate the user's config in an explicit namespace
                namespace = {}
                exec(classifiers_config_text, namespace)

                if "classifiers_to_compare" not in namespace:
                    QMessageBox.warning(
                        self,
                        "Error",
                        "Classifiers configuration must define a 'classifiers_to_compare' variable.\n\nclassifiers_to_compare should be a dict where each key is a classifier set name and each value is a dict of sklearn classifiers.\n\nExample:\nclassifiers_to_compare = {\n    'set1': {\n        'RandomForest': RandomForestClassifier(),\n        'SVM': SVC()\n    },\n    'set2': {\n        'KNN': KNeighborsClassifier()\n    }\n}",
                    )
                    return

                self.classifiers_config = namespace["classifiers_to_compare"]
                if not isinstance(self.classifiers_config, dict):
                    QMessageBox.warning(self, "Error", "classifiers_to_compare must be a dictionary")
                    return

                # Validate that each value is a dict with required keys
                for set_name, classifier_set in self.classifiers_config.items():
                    if not isinstance(classifier_set, dict):
                        QMessageBox.warning(self, "Error", f"Each value in classifiers_to_compare must be a dictionary.\\nThe value for '{set_name}' is not a dictionary.")
                        return

                    # Check for new format with 'classifiers' key
                    if "classifiers" in classifier_set:
                        if not isinstance(classifier_set["classifiers"], dict):
                            QMessageBox.warning(self, "Error", f"The 'classifiers' entry in set '{set_name}' must be a dictionary of sklearn classifiers.")
                            return
                        if "min_prediction_threshold" not in classifier_set:
                            QMessageBox.warning(self, "Error", f"The set '{set_name}' must include a 'min_prediction_threshold' value.")
                            return
                        if not isinstance(classifier_set["min_prediction_threshold"], (int, float)):
                            QMessageBox.warning(self, "Error", f"The 'min_prediction_threshold' in set '{set_name}' must be a number.")
                            return
                        # Validate cross_validation if present
                        if "cross_validation" in classifier_set:
                            cv = classifier_set["cross_validation"]
                            if cv is not None and not hasattr(cv, "split"):
                                QMessageBox.warning(self, "Error", f"The 'cross_validation' in set '{set_name}' must be None or an sklearn cross-validation object with a 'split' method.")
                                return
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Invalid classifiers configuration:\n{str(e)}")
                return
        else:
            # Use default (None will trigger default in train_and_classify)
            self.classifiers_config = None

        output_dir = self.output_dir_input.text()

        # Check if output directory exists and is not empty
        if os.path.exists(output_dir):
            # Check if directory has any contents
            dir_contents = os.listdir(output_dir)
            if dir_contents:
                # Directory is not empty - ask user what to do
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Output Directory Not Empty")
                msg.setText(f"The output directory already contains files:\n\n{output_dir}\n\nFound {len(dir_contents)} item(s).")
                msg.setInformativeText("What would you like to do?\n\nNote: Deletion cannot be undone!")

                # Add custom buttons
                suffix_btn = msg.addButton("Use Suffix", QMessageBox.ActionRole)
                delete_btn = msg.addButton("Delete All", QMessageBox.DestructiveRole)
                cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)

                # Make suffix button the default
                msg.setDefaultButton(suffix_btn)

                self._exec_with_autoclose(msg, suffix_btn)
                clicked_btn = msg.clickedButton()

                if clicked_btn == delete_btn:
                    # User wants to delete - remove all contents
                    try:
                        import shutil

                        for item in dir_contents:
                            item_path = os.path.join(output_dir, item)
                            if os.path.isfile(item_path) or os.path.islink(item_path):
                                os.unlink(item_path)
                            elif os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                        print(f"Cleared output directory: {output_dir}")
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Failed to delete directory contents:\n{str(e)}")
                        return

                elif clicked_btn == suffix_btn:
                    # User wants to use a suffix - find an available directory name
                    import datetime

                    base_dir = output_dir.rstrip(os.sep).rstrip("/").rstrip("\\")
                    counter = 1
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                    # Try timestamp first
                    new_output_dir = f"{base_dir}_{timestamp}"
                    if os.path.exists(new_output_dir) and os.listdir(new_output_dir):
                        # Timestamp exists and not empty, try counter
                        while True:
                            new_output_dir = f"{base_dir}_{counter}"
                            if not os.path.exists(new_output_dir) or not os.listdir(new_output_dir):
                                break
                            counter += 1

                    output_dir = new_output_dir
                    self.output_dir_input.setText(output_dir)
                    print(f"Using new output directory with suffix: {output_dir}")

                else:
                    # User cancelled - ask them to select a different directory
                    QMessageBox.information(self, "Select Different Directory", "Please select a different output directory or clear the existing one manually before training.")
                    return

        os.makedirs(output_dir, exist_ok=True)

        # Clear previous results
        self.subset_results.clear()
        self.current_subset_index = 0

        # Train classifiers for each subset separately
        self.train_next_subset(output_dir)

    def train_next_subset(self, output_dir):
        """Train classifiers for all subsets and classifier sets (creates n×m combinations)."""
        if self.current_subset_index > 0:
            # Training already completed (this method should only run once)
            return

        # Mark as started
        self.current_subset_index = 1

        # Create subset functions for all subsets
        subsets_dict = {}
        self.subset_filter_functions = {}  # Store for later use

        for subset_data in self.subsets:
            subset_name = subset_data["name"]
            subset_expr = subset_data["expression"]

            # Create a closure to capture the current subset_expr (compile once for performance)
            def make_subset_func(expr):
                compiled = compile(expr, "<string>", "eval")
                eval_globals = {"abs": abs, "float": float, "int": int, "str": str}

                def subset_func(row):
                    meta = {k: v for k, v in row.items() if k not in ["peaks", "embeddings"]}
                    try:
                        res = eval(compiled, {**eval_globals, "meta": meta})
                        return res
                    except Exception as ex:
                        print(f"Error evaluating subset expression for row {meta}: {ex}")
                        return False

                return subset_func

            subset_func = make_subset_func(subset_expr)
            subsets_dict[subset_name] = subset_func
            self.subset_filter_functions[subset_name] = subset_func

        # Create progress dialog
        self.progress_dialog = QProgressDialog(f"Training classifiers", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)

        # Set busy cursor during training
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Start worker thread with all subsets (parallel via joblib inside)
        n_jobs = self.n_jobs_spinbox.value()
        self.training_worker = TrainingWorker(
            self.df_embeddings,
            subsets_dict,
            output_dir,
            classifiers_to_compare=self.classifiers_config,
            n_jobs=n_jobs,
        )
        self.training_worker.progress.connect(self.on_training_progress)
        self.training_worker.finished.connect(self.on_training_finished)
        self.training_worker.error.connect(self.on_training_error)
        self.training_worker.start()

        self.train_btn.setEnabled(False)

    def on_training_progress(self, message):
        """Update progress dialog with message."""
        self.progress_dialog.setLabelText(message)

    def on_training_finished(self, results):
        """Handle training completion from parallel jobs."""
        QApplication.restoreOverrideCursor()
        output_dir = self.output_dir_input.text()

        # --- populate per-job log tabs and write log files ----------------
        self.training_log_tabs.clear()
        self.training_log_tabs.setVisible(True)

        for result in results:
            display_key = result["display_key"]
            subset_name = result["subset_name"]
            log_text = result.get("log", "")

            # -- GUI tab --
            log_widget = QTextEdit()
            log_widget.setReadOnly(True)
            log_widget.setLineWrapMode(QTextEdit.NoWrap)
            font = log_widget.font()
            font.setFamily("Courier New")
            log_widget.setFont(font)
            log_widget.setPlainText(log_text)
            self.training_log_tabs.addTab(log_widget, display_key)

            # -- log file --
            combo_output_dir = os.path.join(
                output_dir,
                display_key.replace(" // ", "_").replace(" ", "_"),
            )
            os.makedirs(combo_output_dir, exist_ok=True)
            log_file_path = os.path.join(combo_output_dir, "training.log")
            try:
                with open(log_file_path, "w", encoding="utf-8") as lf:
                    lf.write(log_text)
            except Exception as e:
                print(f"Warning: could not write log file {log_file_path}: {e}")

            # -- store results --
            filter_fn = self.subset_filter_functions.get(subset_name)

            self.subset_results[display_key] = {
                "df_train": result["df_train"],
                "df_validation": result["df_validation"],
                "df_inference": result["df_inference"],
                "df_metrics": result["df_metrics"],
                "trained_classifiers": result["trained_classifiers"],
                "long_table": result["long_table"],
                "pivot_table": result["pivot_table"],
                "filter_fn": filter_fn,
            }

            # Export filtered MGF files for this combination
            self.export_filtered_mgf_files(display_key, combo_output_dir)

        self.progress_dialog.close()

        num_combinations = len(results)
        self.training_status.setText(f"Status: Training completed for {num_combinations} combination(s), output written to {output_dir}.")
        self.train_btn.setEnabled(True)
        self.populate_subset_results_list()
        self.populate_spectrum_file_list()

        # Generate master Excel file consolidating all validation results
        self.generate_master_excel(output_dir)

        QMessageBox.information(
            self,
            "Success",
            f"Classifiers trained for {num_combinations} combination(s)\nOutput written to {output_dir}.\nMaster summary file generated.",
        )

    def on_training_error(self, error_msg):
        """Handle training error."""
        QApplication.restoreOverrideCursor()
        self.progress_dialog.close()
        self.training_status.setText("Status: Error occurred")
        self.train_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", error_msg)

    def generate_master_excel(self, output_dir):
        """Generate master Excel file consolidating results from all subsets."""
        try:
            import polars as pl

            print(f"\nGenerating master overview of all datasets")

            # Initialize an empty list to store DataFrames
            all_results = []

            # Iterate all .xlsx files
            for xlsx_file in os.listdir(output_dir):
                if xlsx_file.endswith("_data.xlsx"):
                    file_path = os.path.join(output_dir, xlsx_file)
                    try:
                        # Try to read the 'overall' sheet
                        df = pl.read_excel(file_path, sheet_name="overall")
                        # Add a new column with the dataset name (derived from filename)
                        subset_name = xlsx_file.replace("_data.xlsx", "")
                        df = df.with_columns(pl.lit(subset_name).alias("subset"))
                        # Append the DataFrame to the list
                        all_results.append(df)
                    except Exception as e:
                        print(f"  Could not load 'overall' sheet from {xlsx_file}: {e}")
                        # Sheet 'overall' might not exist in this file, or it might not be a valid Excel file
                        pass

            if len(all_results) == 0:
                print("No datasets with 'overall' sheet found in the output directory.")
                return

            # Concatenate all DataFrames into a single DataFrame
            all_results = pl.concat(all_results, how="vertical")

            # Process the data according to the user's requirements
            all_results = all_results.with_columns(pl.when(pl.col("annotated_as_times:relevant") != 0).then(pl.lit("relevant")).otherwise(pl.lit("other")).alias("annotated_as"))

            all_results = all_results.rename({"row_count": "n_features"})

            # Calculate percent_features grouped by source, subset, and type
            all_results = all_results.with_columns((100.0 * pl.col("n_features") / pl.col("n_features").sum().over(["source", "subset"])).round(1).alias("percent_features"))

            # Clean up source column
            all_results = all_results.with_columns(pl.col("source").str.replace(" - gt ", " - ", literal=True))

            # Extract gt_type from source
            all_results = all_results.with_columns(
                pl.col("type").str.extract(r"(.*) - (other|relevant)", 1).alias("source_cleaned"),
                pl.col("type").str.extract(r"(.*) - (other|relevant)", 2).alias("gt_type"),
            )

            # Add TN for other/other, FP for other/relevant, FN for relevant/other, and TP for relevant/relevant in columns gt_type/annotated_as
            all_results = all_results.with_columns(
                pl.when((pl.col("gt_type") == "other") & (pl.col("annotated_as") == "other"))
                .then(pl.lit("TN"))
                .when((pl.col("gt_type") == "other") & (pl.col("annotated_as") == "relevant"))
                .then(pl.lit("FP"))
                .when((pl.col("gt_type") == "relevant") & (pl.col("annotated_as") == "other"))
                .then(pl.lit("FN"))
                .when((pl.col("gt_type") == "relevant") & (pl.col("annotated_as") == "relevant"))
                .then(pl.lit("TP"))
                .otherwise(pl.lit(""))
                .alias("pred_type"),
            )

            # Order the DataFrame
            sort_columns = ["subset", "type", "source"]
            if "gt_type" in all_results.columns:
                sort_columns.append("gt_type")
            if "pred_type" in all_results.columns:
                sort_columns.append("pred_type")
            if "annotated_as" in all_results.columns:
                sort_columns.append("annotated_as")
            all_results = all_results.sort(sort_columns)

            # Reorder the columns
            base_columns = ["subset", "type", "source"]
            if "gt_type" in all_results.columns:
                base_columns.append("gt_type")
            if "pred_type" in all_results.columns:
                base_columns.append("pred_type")
            base_columns.extend(["annotated_as", "n_features", "percent_features"])

            # Only select columns that exist
            available_columns = all_results.columns
            final_columns = [col for col in base_columns if col in available_columns]
            all_results = all_results.select(final_columns)

            # Export to master Excel file
            output_excel_file = os.path.join(output_dir, "training_summary.xlsx")
            all_results.write_excel(output_excel_file, worksheet="all_results")

            print(f"Exported master summary to {output_excel_file}")

        except Exception as e:
            print(f"Error generating master Excel file: {e}")
            # Don't fail the entire training process if summary generation fails
            import traceback

            traceback.print_exc()

    def export_filtered_mgf_files(self, combined_key, subset_output_dir):
        """Export filtered MGF files based on classification results.

        For each input MGF file in the combination:
        - Creates two output files with suffix: '_relevant.mgf' and '_other.mgf'
        - Saves to the combination output directory

        Args:
            combined_key: Key in format "classifier_set // subset" or just "subset"
            subset_output_dir: Output directory for this combination
        """
        from .Filters import export_mgf_file

        if combined_key not in self.subset_results:
            print(f"Warning: No results found for combination '{combined_key}'")
            return

        results = self.subset_results[combined_key]
        long_table = results.get("long_table")

        if long_table is None or long_table.empty:
            print(f"Warning: No long_table data available for combination '{combined_key}'")
            return

        if "classification:relevant" not in long_table.columns:
            print(f"Warning: 'classification:relevant' column not found in long_table for combination '{combined_key}'")
            return

        print(f"\nExporting filtered MGF files for combination '{combined_key}'...")
        print(f"Long table shape: {long_table.shape}")
        print(f"Values of column classification:relevant: {long_table['classification:relevant'].value_counts().to_dict()}")

        # Get all unique source files that were used in this subset
        source_files = long_table["source"].unique()
        print(f"Source files to process: {len(source_files)}")

        for source_file in source_files:
            if source_file not in self.mgf_files:
                print(f"Warning: Source file '{source_file}' not found in loaded MGF files")
                continue

            mgf_data = self.mgf_files[source_file]
            original_entries = mgf_data["entries"]
            print(f"\nProcessing {source_file}: {len(original_entries)} total entries")

            # Get all rows from long_table for this source file
            source_data = long_table[long_table["source"] == source_file].copy()
            print(f"  Found {len(source_data)} entries in long_table for this file")

            # Build a classification lookup by AnnoMe_internal_ID
            classification_lookup = {}
            if "AnnoMe_internal_ID" in source_data.columns:
                for _, row in source_data.iterrows():
                    internal_id = row["AnnoMe_internal_ID"]
                    classification = row.get("classification:relevant", "")
                    classification_lookup[internal_id] = classification
                print(f"  Built lookup using AnnoMe_internal_ID: {len(classification_lookup)} entries")
            else:
                print(f"  Warning: AnnoMe_internal_ID not found in long_table")

            # Separate entries by classification
            relevant_entries = []
            other_entries = []
            no_classification_entries = []

            for entry in original_entries:
                # Check if entry has AnnoMe_internal_ID
                internal_id = entry.get("AnnoMe_internal_ID", "")
                classification = classification_lookup.get(internal_id, "")

                # print(f"   - processing internal_id {internal_id}: classification = '{classification}'")

                if classification == "relevant":
                    relevant_entries.append(entry)
                elif classification == "other":
                    other_entries.append(entry)
                else:
                    no_classification_entries.append(entry)

            print(f"  Classified: {len(relevant_entries)} relevant, {len(other_entries)} other, {len(no_classification_entries)} unclassified")

            # Generate output filenames with suffix
            base_filename = os.path.basename(source_file)
            name_without_ext, ext = os.path.splitext(base_filename)

            # Always export both files, even if empty
            relevant_filename = f"{name_without_ext}_relevant{ext}"
            other_filename = f"{name_without_ext}_other{ext}"
            no_classification_filename = f"{name_without_ext}_unclassified{ext}"

            relevant_output_path = os.path.join(subset_output_dir, relevant_filename)
            other_output_path = os.path.join(subset_output_dir, other_filename)
            no_classification_output_path = os.path.join(subset_output_dir, no_classification_filename)

            # Export relevant entries
            try:
                export_mgf_file(relevant_entries, relevant_output_path)
                print(f"  ✓ Exported {len(relevant_entries)} relevant entries to {relevant_filename}")
            except Exception as e:
                print(f"  ✗ Error exporting relevant entries: {e}")
                traceback.print_exc()

            # Export other entries
            try:
                export_mgf_file(other_entries, other_output_path)
                print(f"  ✓ Exported {len(other_entries)} other entries to {other_filename}")
            except Exception as e:
                print(f"  ✗ Error exporting other entries: {e}")
                traceback.print_exc()

    def populate_subset_results_list(self):
        """Populate the spectrum subset list (step 6) and refresh the results table (step 5)."""
        # Populate section 6 subset list
        self.spectrum_subset_list.clear()
        for combined_key in sorted(self.subset_results.keys()):
            self.spectrum_subset_list.addItem(combined_key)

        # Refresh the results table in step 5 to show all combinations
        self.refresh_all_results_table()

    def on_results_table_selection_changed(self):
        """Update the histogram to reflect only the selected table rows.

        If no rows are selected the full dataset is shown.
        """
        if not hasattr(self, "_results_combined_summary") or self._results_combined_summary is None:
            return

        selected_rows = list({idx.row() for idx in self.results_table.selectedIndexes()})

        if not selected_rows:
            # Nothing selected – show everything
            self.update_results_histogram(self._results_combined_summary)
            return

        # Build a set of (combination, source) keys from the selected table rows
        selected_keys = set()
        for row in selected_rows:
            classifier_item = self.results_table.item(row, 0)
            subset_item = self.results_table.item(row, 1)
            source_item = self.results_table.item(row, 2)
            if classifier_item and subset_item and source_item:
                combo = f"{classifier_item.text()} // {subset_item.text()}"
                source = source_item.text()
                selected_keys.add((combo, source))

        if not selected_keys:
            self.update_results_histogram(self._results_combined_summary)
            return

        filtered = self._results_combined_summary[self._results_combined_summary.apply(lambda r: (r["combination"], r["source"]) in selected_keys, axis=1)].reset_index(drop=True)

        self.update_results_histogram(filtered if not filtered.empty else self._results_combined_summary)

    # Section 5 methods
    def refresh_all_results_table(self):
        """Refresh the results table to show all combinations at once."""
        if not self.subset_results:
            self.results_table.setRowCount(0)
            self._results_combined_summary = None
            self.update_results_histogram(None)
            return

        # Get file types for display
        file_types = {}
        for filename, data in self.mgf_files.items():
            file_types[filename] = data["type"]

        # Collect all rows from all combinations
        all_rows = []

        for combined_key in sorted(self.subset_results.keys()):
            results = self.subset_results[combined_key]
            long_table = results.get("long_table")

            if long_table is None or long_table.empty:
                continue

            # Use the long table which has the "classification:relevant" column
            df = long_table.copy()

            # Ensure the classification:relevant column exists
            if "classification:relevant" not in df.columns:
                continue

            # Aggregate by source and type, counting relevant vs not relevant
            # Create a binary relevant flag: relevant if "classification:relevant" == "relevant", otherwise not relevant
            df["is_relevant"] = df["classification:relevant"] == "relevant"

            # Group by source and type, then count relevant and not relevant
            summary = df.groupby(["source", "type"]).agg(total=("is_relevant", "count"), relevant_count=("is_relevant", "sum")).reset_index()

            # Calculate not relevant count
            summary["not_relevant_count"] = summary["total"] - summary["relevant_count"]

            # Remove rows where total is 0
            summary = summary[summary["total"] > 0]

            # Add combination name to each row
            summary["combination"] = combined_key
            all_rows.append(summary)

        if not all_rows:
            self.results_table.setRowCount(0)
            self._results_combined_summary = None
            self.update_results_histogram(None)
            return

        # Combine all summaries
        combined_summary = pd.concat(all_rows, ignore_index=True)

        # Disable sorting during population to avoid re-sorting scrambling rows
        self.results_table.setSortingEnabled(True)

        # Set up the table with 8 columns (classifier and subset as separate columns)
        self.results_table.setRowCount(len(combined_summary))
        self.results_table.setColumnCount(9)
        self.results_table.setHorizontalHeaderLabels(
            ["Classifier", "Subset", "MGF File", "Type", "Total (n)", "Prediction: Other (Count)", "Prediction: Other (%)", "Prediction: Relevant (Count)", "Prediction: Relevant (%)"]
        )

        # Define match/mismatch colors (once, outside loop)
        COLOR_MATCH = QColor(144, 238, 144)  # Light green
        COLOR_MISMATCH = QColor(255, 182, 193)  # Light red

        # Populate the table
        for row_idx, row in combined_summary.iterrows():
            combination = row["combination"]
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

            # Split combination into classifier and subset
            parts = combination.split(" // ")
            classifier = parts[0] if len(parts) >= 1 else combination
            subset = parts[1] if len(parts) >= 2 else ""

            # Classifier column (first column)
            self.results_table.setItem(row_idx, 0, QTableWidgetItem(classifier))

            # Subset column (second column)
            self.results_table.setItem(row_idx, 1, QTableWidgetItem(subset))

            # Source column
            self.results_table.setItem(row_idx, 2, QTableWidgetItem(source))

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
            self.results_table.setItem(row_idx, 3, type_item)

            # Total count (no background color)
            self.results_table.setItem(row_idx, 4, QTableWidgetItem(str(int(total))))

            # Not relevant count
            item = QTableWidgetItem(str(int(not_relevant)))
            if is_inference:
                item.setBackground(QBrush(COLOR_INFERENCE))
            elif expected_type == "other":
                item.setBackground(QBrush(COLOR_MATCH))
            else:
                item.setBackground(QBrush(COLOR_MISMATCH))
            self.results_table.setItem(row_idx, 5, item)

            # Not relevant %
            item = QTableWidgetItem(f"{not_relevant_pct:.1f}%")
            if is_inference:
                item.setBackground(QBrush(COLOR_INFERENCE))
            elif expected_type == "other":
                item.setBackground(QBrush(COLOR_MATCH))
            else:
                item.setBackground(QBrush(COLOR_MISMATCH))
            self.results_table.setItem(row_idx, 6, item)

            # Relevant count
            item = QTableWidgetItem(str(int(relevant)))
            if is_inference:
                item.setBackground(QBrush(COLOR_INFERENCE))
            elif expected_type == "relevant":
                item.setBackground(QBrush(COLOR_MATCH))
            else:
                item.setBackground(QBrush(COLOR_MISMATCH))
            self.results_table.setItem(row_idx, 7, item)

            # Relevant %
            item = QTableWidgetItem(f"{relevant_pct:.1f}%")
            if is_inference:
                item.setBackground(QBrush(COLOR_INFERENCE))
            elif expected_type == "relevant":
                item.setBackground(QBrush(COLOR_MATCH))
            else:
                item.setBackground(QBrush(COLOR_MISMATCH))
            self.results_table.setItem(row_idx, 8, item)

        # Re-enable sorting and auto-resize columns
        self.results_table.setSortingEnabled(True)
        self.results_table.resizeColumnsToContents()

        # Store full summary so the selection handler can filter it
        self._results_combined_summary = combined_summary

        # Redraw histogram
        self.update_results_histogram(combined_summary)

    def update_results_histogram(self, combined_summary):
        """Redraw the classification results histogram.

        Three groups of subplots are drawn in a single vertical column:
          1. One subplot per classifier-set // subset combination
          2. One subplot per source file
          3. One subplot per dataset type

        Each subplot shows 100 % stacked vertical bars (Other / Relevant).
        The canvas height is computed from the number of subplots and the
        scroll area provides vertical navigation.
        """
        if not self._results_histogram_available:
            return

        self.results_figure.clear()

        if combined_summary is None or combined_summary.empty:
            self.results_figure.set_size_inches(7, 3)
            ax = self.results_figure.add_subplot(111)
            ax.text(0.5, 0.5, "No results to display", ha="center", va="center", fontsize=12)
            ax.set_axis_off()
            dpi = self.results_figure.dpi
            self.results_canvas.resize(int(7 * dpi), int(3 * dpi))
            self.results_canvas.draw()
            return

        # ------------------------------------------------------------------ #
        # Colour scheme – matches the table type colours                       #
        # ------------------------------------------------------------------ #
        TYPE_META = {
            "train - relevant": {"bg": "#C5D89D", "bar_rel": "#5A9E4A", "bar_oth": "#A7C87A"},
            "train - other": {"bg": "#D25353", "bar_rel": "#7B1F1F", "bar_oth": "#C06060"},
            "validation - relevant": {"bg": "#9CAB84", "bar_rel": "#4A7A3A", "bar_oth": "#7A9A64"},
            "validation - other": {"bg": "#9E3B3B", "bar_rel": "#5A1010", "bar_oth": "#8A4040"},
            "inference": {"bg": "#456882", "bar_rel": "#7BAED0", "bar_oth": "#A0C4DC"},
        }
        FALLBACK = {"bg": "#AAAAAA", "bar_rel": "#555555", "bar_oth": "#888888"}

        # ---- Determine the three groups ----------------------------------- #
        combos = sorted(combined_summary["combination"].unique())
        files = sorted(combined_summary["source"].unique())
        ordered_types = [t for t in TYPE_META if t in combined_summary["type"].values]
        ordered_types += sorted(t for t in combined_summary["type"].unique() if t not in TYPE_META)

        n_combos = len(combos)
        n_files = len(files)
        n_types = len(ordered_types)
        n_total = n_combos + n_files + n_types

        if n_total == 0:
            self.results_canvas.draw()
            return

        # ---- Size the figure for vertical scrollability ------------------- #
        BAR_HEIGHT_IN = 0.35  # height per individual bar
        MIN_SUBPLOT_HEIGHT_IN = 1.0  # minimum height even for a single bar
        TITLE_PAD_IN = 0.4  # extra space for subplot title + x-label
        SUBPLOT_WIDTH_IN = 8.0  # slightly wider for long y-labels
        dpi = self.results_figure.dpi

        # Pre-compute number of bars per subplot so heights can vary
        bar_counts = []
        # Group 1: per combination
        for combo in combos:
            n = len(combined_summary[combined_summary["combination"] == combo])
            bar_counts.append(n)
        # Group 2: per source file
        for fname in files:
            n = len(combined_summary[combined_summary["source"] == fname])
            bar_counts.append(n)
        # Group 3: per type
        for type_name in ordered_types:
            n = len(combined_summary[combined_summary["type"] == type_name])
            bar_counts.append(n)

        subplot_heights = [max(MIN_SUBPLOT_HEIGHT_IN, n * BAR_HEIGHT_IN + TITLE_PAD_IN) for n in bar_counts]
        fig_h = max(2.0, sum(subplot_heights))
        self.results_figure.set_size_inches(SUBPLOT_WIDTH_IN, fig_h)

        # ---- Create one subplot per row using gridspec -------------------- #
        gs = self.results_figure.add_gridspec(
            n_total,
            1,
            height_ratios=subplot_heights,
        )
        axes = [self.results_figure.add_subplot(gs[i]) for i in range(n_total)]

        # ---- Helper: draw 100% stacked bars for a DataFrame slice --------- #
        # ---- Match / mismatch / inference colours (mirror the table) ------- #
        C_MATCH = "#90EE90"  # light green  – correct prediction
        C_MISMATCH = "#FFB6C1"  # light pink   – wrong prediction
        C_INFERENCE = "#7BAED0"  # steel blue   – inference (no ground truth)
        C_MATCH_TXT = "#2d6e2d"  # dark green for annotation text on green
        C_MISMATCH_TXT = "#8b0000"  # dark red   for annotation text on pink
        C_INFERENCE_TXT = "#1a3a50"  # dark blue  for annotation text on blue

        def _bar_colours(file_type):
            """Return (colour_other, colour_relevant, text_other, text_relevant)."""
            if "inference" in file_type:
                return C_INFERENCE, C_INFERENCE, C_INFERENCE_TXT, C_INFERENCE_TXT
            elif "relevant" in file_type:
                # correct = relevant bar; wrong = other bar
                return C_MISMATCH, C_MATCH, C_MISMATCH_TXT, C_MATCH_TXT
            else:
                # correct = other bar; wrong = relevant bar
                return C_MATCH, C_MISMATCH, C_MATCH_TXT, C_MISMATCH_TXT

        def _draw_stacked(ax, sub, x_labels, title, group_bg=None):
            for yi, (_, row) in enumerate(sub.iterrows()):
                total = row["total"]
                if total == 0:
                    continue
                rel_pct = row["relevant_count"] / total * 100
                oth_pct = row["not_relevant_count"] / total * 100
                c_oth, c_rel, ct_oth, ct_rel = _bar_colours(row["type"])

                # Horizontal stacked bars: Other from 0, Relevant starts at oth_pct
                ax.barh(yi, oth_pct, color=c_oth, alpha=0.92)
                ax.barh(yi, rel_pct, left=oth_pct, color=c_rel, alpha=0.92)
                if rel_pct > 6:
                    ax.text(oth_pct + rel_pct / 2, yi, f"{rel_pct:.0f}%", ha="center", va="center", fontsize=7, color=ct_rel, fontweight="bold")
                if oth_pct > 6:
                    ax.text(oth_pct / 2, yi, f"{oth_pct:.0f}%", ha="center", va="center", fontsize=7, color=ct_oth, fontweight="bold")

            ax.set_yticks(range(len(x_labels)))
            ax.set_yticklabels(x_labels, fontsize=7)
            ax.set_xlim(0, 100)
            ax.margins(x=0)
            ax.set_xlabel("%", fontsize=8)
            ax.tick_params(axis="x", labelsize=7)
            if group_bg:
                ax.set_facecolor(group_bg + "22")
            ax.set_title(title, fontsize=8, fontweight="bold", pad=3, loc="left")

        # ---- Group 1: one subplot per combination ------------------------- #
        for idx, combo in enumerate(combos):
            sub = combined_summary[combined_summary["combination"] == combo].reset_index(drop=True)
            _draw_stacked(axes[idx], sub, x_labels=sub["source"].tolist(), title=f"[Combination]  {combo}")

        # ---- Group 2: one subplot per source file ------------------------- #
        for idx, fname in enumerate(files):
            sub = combined_summary[combined_summary["source"] == fname].reset_index(drop=True)
            ftype = sub["type"].iloc[0] if not sub.empty else ""
            meta = TYPE_META.get(ftype, FALLBACK)
            _draw_stacked(axes[n_combos + idx], sub, x_labels=sub["combination"].tolist(), title=f"[File]  {fname}  ({ftype})")

        # ---- Group 3: one subplot per dataset type ------------------------ #
        for idx, type_name in enumerate(ordered_types):
            sub = combined_summary[combined_summary["type"] == type_name].reset_index(drop=True)
            meta = TYPE_META.get(type_name, FALLBACK)
            if len(sub["combination"].unique()) == 1:
                x_labels = sub["source"].tolist()
            else:
                x_labels = [f"{c}\n{s}" for c, s in zip(sub["combination"], sub["source"])]
            _draw_stacked(axes[n_combos + n_files + idx], sub, x_labels=x_labels, title=f"[Type]  {type_name}")

        # Resize canvas to match the computed figure dimensions so the scroll
        # area can provide vertical navigation.
        canvas_w = int(SUBPLOT_WIDTH_IN * dpi)
        canvas_h = int(fig_h * dpi)
        self.results_canvas.resize(canvas_w, canvas_h)

        # ---- Finalize ----------------------------------------------------- #
        # Call tight_layout BEFORE canvas.resize() so it operates on the figure
        # size we set with set_size_inches (the Qt resize event would otherwise
        # reset it first, making tight_layout compute against a stale size).
        # rect=[left, bottom, right, top] in figure-fraction coordinates.
        # Generous left margin so y-tick labels (filenames) are never clipped.
        self.results_figure.tight_layout()
        self.results_canvas.draw()

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

    # Section 6 methods
    def on_subset_selected_for_spectra(self):
        """Handle subset selection for spectrum inspection."""
        # Update the MGF file list to show files from the selected subset
        self.populate_spectrum_file_list()
        # Clear the spectrum table since subset changed
        self.spectrum_table.setRowCount(0)

    def populate_spectrum_file_list(self):
        """Populate the spectrum file list based on selected subset."""
        self.spectrum_file_list.clear()

        # Get selected subset from list
        selected_items = self.spectrum_subset_list.selectedItems()
        if not selected_items:
            # No subset selected, don't show any files
            return

        selected_subset = selected_items[0].text()

        # Get files that are in the selected subset's dataframes
        subset_files = set()
        if selected_subset in self.subset_results:
            results = self.subset_results[selected_subset]
            for df_key in ["df_train", "df_validation", "df_inference"]:
                df = results.get(df_key)
                if df is not None and not df.empty and "source" in df.columns:
                    subset_files.update(df["source"].unique())

        # Add files that are in this subset
        for filename in sorted(self.mgf_files.keys()):
            if filename in subset_files:
                self.spectrum_file_list.addItem(filename)

    def _get_entry_index_from_internal_id(self, internal_id, data):
        """Extract the 0-based entry index from an AnnoMe_internal_ID string.

        The ID format is 'filename___curId' where curId is the 0-based
        sequential index assigned during embedding generation, matching
        the order of entries returned by parse_mgf_file.

        Falls back to scanning data["entries"] for a matching ID if the
        format is unexpected.
        """
        if internal_id and "___" in str(internal_id):
            try:
                return int(str(internal_id).rsplit("___", 1)[1])
            except (ValueError, IndexError):
                pass
        # Fallback: linear search through entries
        for idx, entry in enumerate(data["entries"]):
            if entry.get("AnnoMe_internal_ID", "") == internal_id:
                return idx
        return -1

    def on_spectrum_selected(self):
        """Update the embedded spectrum viewer when a spectrum is selected."""
        current_row = self.spectrum_table.currentRow()
        if current_row < 0 or not self.spectrum_file_list.selectedItems() or not self.spectrum_subset_list.selectedItems():
            return

        # Get selected subset and its results
        selected_subset = self.spectrum_subset_list.selectedItems()[0].text()
        if selected_subset not in self.subset_results:
            return

        results = self.subset_results[selected_subset]
        long_table = results.get("long_table")
        if long_table is None or long_table.empty:
            return

        id_item = self.spectrum_table.item(current_row, 0)
        if not id_item:
            return

        # Use the AnnoMe_internal_ID (stored in UserRole) to find the correct entry in long_table
        internal_id = id_item.data(Qt.UserRole)
        filename = self.spectrum_file_list.selectedItems()[0].text()

        # Find the spectrum row in long_table by AnnoMe_internal_ID
        matching_rows = long_table[long_table["AnnoMe_internal_ID"] == internal_id]
        if matching_rows.empty:
            return

        row = matching_rows.iloc[0]

        # Extract metadata (exclude spectrum data columns)
        meta_data = {k: v for k, v in row.items() if k not in ["$$SpectrumData", "peaks", "embeddings"]}

        # Get classification from table
        classification_item = self.spectrum_table.item(current_row, 7)
        classification = classification_item.text() if classification_item else ""

        # Update classification label with color coding
        if classification == "relevant":
            self.spectrum_classification_label.setText(f"Classification: {classification}")
            self.spectrum_classification_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #90EE90; color: #000000;")
        elif classification == "other":
            self.spectrum_classification_label.setText(f"Classification: {classification}")
            self.spectrum_classification_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #FFB6C1; color: #000000;")
        else:
            self.spectrum_classification_label.setText("Classification: Not classified")
            self.spectrum_classification_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #f0f0f0; color: #666666;")

        # Build detailed prediction info to add to metadata
        spectrum_name = meta_data.get("name", "")

        # Update metadata display with classification info included
        meta_html = "<table border='1' cellpadding='5' style='width:100%'>"

        # Add all metadata
        for key, value in sorted(meta_data.items()):
            meta_html += f"<tr><td><b>{key}</b></td><td>{value}</td></tr>"
        meta_html += "</table>"
        self.spectrum_meta_text.setHtml(meta_html)

        # Update spectrum plot
        try:
            spectrum_data = row.get("cleaned_spectra", None)

            self.spectrum_ax.clear()

            if spectrum_data is not None:
                # Parse m/z and intensity values as floats
                if isinstance(spectrum_data, Spectrum):
                    mz_values = spectrum_data.peaks.mz
                    intensity_values = spectrum_data.peaks.intensities
                elif isinstance(spectrum_data, (list, tuple)) and len(spectrum_data) == 2:
                    # Convert string values to floats
                    mz_values = [float(x) for x in spectrum_data[0]]
                    intensity_values = [float(x) for x in spectrum_data[1]]
                elif hasattr(spectrum_data, "shape") and len(spectrum_data.shape) == 2:
                    # numpy array - ensure floats
                    mz_values = spectrum_data[0, :].astype(float)
                    intensity_values = spectrum_data[1, :].astype(float)
                else:
                    # Fallback: assume it's a list of [mz, intensity] pairs
                    mz_values = [float(peak[0]) for peak in spectrum_data]
                    intensity_values = [float(peak[1]) for peak in spectrum_data]

                self.spectrum_ax.vlines(mz_values, 0, intensity_values, colors="blue", linewidth=1.5)
                self.spectrum_ax.set_xlabel("m/z", fontsize=10)
                self.spectrum_ax.set_ylabel("Intensity", fontsize=10)
                self.spectrum_ax.set_title(f"MS/MS Spectrum - {spectrum_name}", fontsize=11)
                self.spectrum_ax.grid(True, alpha=0.3)
            else:
                self.spectrum_ax.text(0.5, 0.5, "No spectrum data available", ha="center", va="center", fontsize=12)
                self.spectrum_ax.set_xlim(0, 1)
                self.spectrum_ax.set_ylim(0, 1)

            self.spectrum_figure.tight_layout()
            self.spectrum_canvas.draw()
        except Exception as e:
            self.spectrum_ax.clear()
            self.spectrum_ax.text(0.5, 0.5, f"Error: {str(e)}", ha="center", va="center", fontsize=10, color="red")
            self.spectrum_ax.set_xlim(0, 1)
            self.spectrum_ax.set_ylim(0, 1)
            self.spectrum_canvas.draw()

    def on_spectrum_file_selected(self):
        """Handle spectrum file selection."""
        # Check that both subset and file are selected
        subset_items = self.spectrum_subset_list.selectedItems()
        file_items = self.spectrum_file_list.selectedItems()

        if not subset_items or not file_items:
            self.spectrum_table.setRowCount(0)
            return

        selected_subset = subset_items[0].text()
        filename = file_items[0].text()

        if selected_subset not in self.subset_results:
            self.spectrum_table.setRowCount(0)
            return

        # Disable sorting during population to avoid issues
        self.spectrum_table.setSortingEnabled(False)
        self.spectrum_table.setRowCount(0)

        # Get the subset filter function and results
        results = self.subset_results[selected_subset]
        subset_filter = results.get("filter_fn")

        if subset_filter is None:
            self.spectrum_table.setRowCount(0)
            return

        # Get data directly from the long_table which contains all information
        long_table = results.get("long_table")
        if long_table is None or long_table.empty:
            self.spectrum_table.setRowCount(0)
            return

        # Filter by source and subset filter
        df_filtered = long_table[long_table["source"] == filename] if "source" in long_table.columns else long_table

        # Apply the subset filter function
        if subset_filter is not None:
            df_filtered = df_filtered[df_filtered.apply(subset_filter, axis=1)]

        # Populate table directly from the filtered long_table
        for row_idx, (_, row) in enumerate(df_filtered.iterrows()):
            self.spectrum_table.insertRow(row_idx)

            # Store AnnoMe_internal_ID in the ID column for later retrieval
            internal_id = row.get("AnnoMe_internal_ID", "")
            id_item = QTableWidgetItem()
            id_item.setData(Qt.DisplayRole, row_idx + 1)  # Display row number
            id_item.setData(Qt.UserRole, internal_id)  # Store internal ID for lookup
            self.spectrum_table.setItem(row_idx, 0, id_item)

            # m/z (numeric sorting)
            mz_item = QTableWidgetItem()
            mz = row.get("precursor_mz", "")
            try:
                mz_float = float(mz) if mz else 0.0
                mz_item.setData(Qt.DisplayRole, mz_float)
            except (ValueError, TypeError):
                mz_item.setData(Qt.DisplayRole, str(mz))
            self.spectrum_table.setItem(row_idx, 1, mz_item)

            # RT (numeric sorting)
            rt_item = QTableWidgetItem()
            rt = row.get("RTINSECONDS", "")
            try:
                rt_float = float(rt) if rt else 0.0
                rt_item.setData(Qt.DisplayRole, rt_float)
            except (ValueError, TypeError):
                rt_item.setData(Qt.DisplayRole, str(rt))
            self.spectrum_table.setItem(row_idx, 2, rt_item)

            # CE (numeric sorting)
            ce_item = QTableWidgetItem()
            ce = row.get("CE", "")
            try:
                ce_float = float(ce) if ce else 0.0
                ce_item.setData(Qt.DisplayRole, ce_float)
            except (ValueError, TypeError):
                ce_item.setData(Qt.DisplayRole, str(ce))
            self.spectrum_table.setItem(row_idx, 3, ce_item)

            # Source (text)
            source = row.get("source", "")
            self.spectrum_table.setItem(row_idx, 4, QTableWidgetItem(str(source)))

            # Fragmentation method (text)
            frag_method = row.get("fragmentation_method", "")
            self.spectrum_table.setItem(row_idx, 5, QTableWidgetItem(str(frag_method)))

            # Ion mode (text)
            ion_mode = row.get("ionMode", row.get("ionmode", ""))
            self.spectrum_table.setItem(row_idx, 6, QTableWidgetItem(str(ion_mode)))

            # Classification (text with color coding)
            classification = row.get("classification:relevant", "")
            if not classification:
                classification = "other"
            class_item = QTableWidgetItem(str(classification))
            if classification == "relevant":
                class_item.setBackground(QBrush(QColor(144, 238, 144)))  # Light green
            elif classification == "other":
                class_item.setBackground(QBrush(QColor(255, 182, 193)))  # Light pink
            self.spectrum_table.setItem(row_idx, 7, class_item)

            # Color the entire row based on classification
            row_color = None
            if classification == "relevant":
                row_color = QColor(240, 255, 240)  # Very light green
            elif classification == "other":
                row_color = QColor(255, 240, 240)  # Very light pink

            if row_color:
                for col in range(8):
                    item = self.spectrum_table.item(row_idx, col)
                    if item:
                        item.setBackground(QBrush(row_color))

        # Re-enable sorting
        self.spectrum_table.setSortingEnabled(True)
        self.spectrum_table.resizeColumnsToContents()

    def view_spectrum_details(self):
        """View details of selected spectrum."""
        if not self.spectrum_file_list.selectedItems() or not self.spectrum_subset_list.selectedItems():
            QMessageBox.warning(self, "Warning", "Please select a file and subset")
            return

        current_row = self.spectrum_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Warning", "Please select a spectrum")
            return

        # Get selected subset and its results
        selected_subset = self.spectrum_subset_list.selectedItems()[0].text()
        if selected_subset not in self.subset_results:
            QMessageBox.warning(self, "Warning", "Subset results not available")
            return

        results = self.subset_results[selected_subset]
        long_table = results.get("long_table")
        if long_table is None or long_table.empty:
            QMessageBox.warning(self, "Warning", "No data available in long_table")
            return

        # Get spectrum index from the ID column (first column) using AnnoMe_internal_ID
        id_item = self.spectrum_table.item(current_row, 0)
        if not id_item:
            return

        internal_id = id_item.data(Qt.UserRole)

        # Find the spectrum row in long_table by AnnoMe_internal_ID
        matching_rows = long_table[long_table["AnnoMe_internal_ID"] == internal_id]
        if matching_rows.empty:
            QMessageBox.warning(self, "Warning", "Spectrum not found in data")
            return

        row = matching_rows.iloc[0]

        # Extract metadata directly from row (keys are not nested in params)
        meta_data = {k: v for k, v in row.items() if k not in ["$$SpectrumData", "peaks", "embeddings"]}

        # Get spectrum data - stored in $$SpectrumData (case insensitive check)
        spectrum_data = row.get("cleaned_spectra", None)

        # Get classification result from table
        classification_item = self.spectrum_table.item(current_row, 7)
        classification = classification_item.text() if classification_item else ""

        dialog = SpectrumViewer(spectrum_data, meta_data, self)
        dialog.exec_()

    def export_individual_spectrum(self):
        """Export individual spectrum to Excel."""
        if not self.spectrum_file_list.selectedItems() or not self.spectrum_subset_list.selectedItems():
            QMessageBox.warning(self, "Warning", "Please select a file and subset")
            return

        current_row = self.spectrum_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Warning", "Please select a spectrum")
            return

        # Get selected subset and its results
        selected_subset = self.spectrum_subset_list.selectedItems()[0].text()
        if selected_subset not in self.subset_results:
            QMessageBox.warning(self, "Warning", "Subset results not available")
            return

        results = self.subset_results[selected_subset]
        long_table = results.get("long_table")
        if long_table is None or long_table.empty:
            QMessageBox.warning(self, "Warning", "No data available in long_table")
            return

        # Get spectrum index from the ID column using AnnoMe_internal_ID
        id_item = self.spectrum_table.item(current_row, 0)
        if not id_item:
            return

        internal_id = id_item.data(Qt.UserRole)

        # Find the spectrum row in long_table by AnnoMe_internal_ID
        matching_rows = long_table[long_table["AnnoMe_internal_ID"] == internal_id]
        if matching_rows.empty:
            QMessageBox.warning(self, "Warning", "Spectrum not found in data")
            return

        row = matching_rows.iloc[0]

        # Extract metadata directly from row
        meta_data = {k: v for k, v in row.items() if k not in ["$$SpectrumData", "peaks", "embeddings"]}

        # Add classification from table
        classification_item = self.spectrum_table.item(current_row, 7)
        if classification_item:
            meta_data["classification"] = classification_item.text()

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
        """Navigate to a specific section (tab)."""
        # Find the index of the section widget in the tab widget
        if section == self.section1:
            self.tab_widget.setCurrentIndex(0)
        elif section == self.section2:
            self.tab_widget.setCurrentIndex(1)
        elif section == self.section3:
            self.tab_widget.setCurrentIndex(2)
        elif section == self.section4:
            self.tab_widget.setCurrentIndex(3)
        elif section == self.section5:
            self.tab_widget.setCurrentIndex(4)
        elif section == self.section6:
            self.tab_widget.setCurrentIndex(5)


def main():
    parser = argparse.ArgumentParser(
        prog="AnnoMe-Classification",
        description="AnnoMe Classification GUI",
        add_help=True,
    )
    parser.add_argument(
        "-loadConfig",
        metavar="PATH",
        default=None,
        help="Path to a JSON configuration file that will be loaded automatically on startup.",
    )
    parser.add_argument(
        "-startTraining",
        action="store_true",
        default=False,
        help="Automatically start classifier training after the configuration has been loaded. Requires -loadConfig to be specified as well.",
    )

    # argparse uses sys.argv[1:] by default; Qt also reads sys.argv so pass only
    # non-Qt arguments.  The simplest approach is to parse known args only.
    args, qt_args = parser.parse_known_args()

    if args.startTraining and args.loadConfig is None:
        parser.error("-startTraining requires -loadConfig <PATH> to be specified.")

    # Build argument list for QApplication (program name + remaining unknown args)
    qt_argv = [sys.argv[0]] + qt_args

    app = QApplication(qt_argv)
    window = ClassificationGUI(
        config_path=args.loadConfig,
        auto_start_training=args.startTraining,
    )
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
