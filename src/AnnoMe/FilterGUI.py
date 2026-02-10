import sys
import os
import re
from collections import defaultdict, OrderedDict
import io
import json
import tomllib
from colorama import Fore, Style
import polars as pl
import time
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
    QTabWidget,
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
import math

import rdkit.Chem
from rdkit.Chem import Descriptors
import natsort
import tempfile
from pprint import pprint
import traceback
import csv

from . import Filters


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
        if checked:
            self.toggle_button.setText(text.replace("▶", "▼"))
        else:
            self.toggle_button.setText(text.replace("▼", "▶"))
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


class StructureViewerWindow(QMainWindow):
    """A separate window to display matched and non-matched structures."""

    def __init__(self, filter_name, matched_smiles, non_matched_smiles, parent=None):
        super().__init__(parent)
        self.filter_name = filter_name
        self.matched_smiles = matched_smiles
        self.non_matched_smiles = non_matched_smiles
        self.temp_dir = tempfile.TemporaryDirectory()
        self.zoom_level = 100  # Default zoom level

        self.setWindowTitle(f"Structure Viewer - {filter_name}")
        self.setGeometry(150, 150, 1200, 800)

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(25)
        self.zoom_slider.setMaximum(400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickPosition(QSlider.TicksBelow)
        self.zoom_slider.setTickInterval(25)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)  # Connect to slot
        zoom_layout.addWidget(self.zoom_slider)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(50)
        zoom_layout.addWidget(self.zoom_label)

        zoom_reset_btn = QPushButton("Reset Zoom")
        zoom_reset_btn.clicked.connect(self.reset_zoom)
        zoom_layout.addWidget(zoom_reset_btn)

        zoom_layout.addStretch()
        layout.addLayout(zoom_layout)

        # Create splitter for matched and non-matched
        splitter = QSplitter(Qt.Horizontal)

        # Matched structures
        matched_widget = QWidget()
        matched_layout = QVBoxLayout()

        matched_header = QLabel(f"Matched Structures ({len(self.matched_smiles)} unique SMILES)")
        matched_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        matched_layout.addWidget(matched_header)

        self.matched_scroll = QScrollArea()
        self.matched_scroll.setWidgetResizable(True)
        self.matched_container = QLabel()
        self.matched_container.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.matched_scroll.setWidget(self.matched_container)
        matched_layout.addWidget(self.matched_scroll)

        matched_widget.setLayout(matched_layout)
        splitter.addWidget(matched_widget)

        # Non-matched structures
        non_matched_widget = QWidget()
        non_matched_layout = QVBoxLayout()

        non_matched_header = QLabel(f"Non-Matched Structures ({len(self.non_matched_smiles)} unique SMILES)")
        non_matched_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        non_matched_layout.addWidget(non_matched_header)

        self.non_matched_scroll = QScrollArea()
        self.non_matched_scroll.setWidgetResizable(True)
        self.non_matched_container = QLabel()
        self.non_matched_container.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.non_matched_scroll.setWidget(self.non_matched_container)
        non_matched_layout.addWidget(self.non_matched_scroll)

        non_matched_widget.setLayout(non_matched_layout)
        splitter.addWidget(non_matched_widget)

        layout.addWidget(splitter)
        central_widget.setLayout(layout)

        # Store original pixmaps
        self.matched_pixmap = None
        self.non_matched_pixmap = None

        # Render structures
        self.render_structures()

    def render_structures(self):
        """Render the structures as images."""
        # Render matched structures
        if self.matched_smiles:
            try:
                print(f"Rendering {len(self.matched_smiles)} matched structures...")

                # Generate image using custom rendering to avoid RDKit PNG issues
                matched_path = os.path.join(self.temp_dir.name, f"{self.filter_name}_matched.png")
                self.render_molecules_to_file(self.matched_smiles[:500], matched_path, 5)

                # Load with QPixmap
                self.matched_pixmap = QPixmap(matched_path)

                if self.matched_pixmap.isNull():
                    raise ValueError(f"Failed to load pixmap from {matched_path}")

                print(f"Pixmap loaded: {self.matched_pixmap.width()}x{self.matched_pixmap.height()}")
                self.update_matched_display()

            except Exception as e:
                error_msg = f"Error displaying matched structures:\n{str(e)}\n\nType: {type(e).__name__}"
                print(error_msg)
                import traceback

                traceback.print_exc()
                self.matched_container.setText(error_msg)
        else:
            self.matched_container.setText("No matched structures")

        # Render non-matched structures
        if self.non_matched_smiles:
            try:
                print(f"Rendering {len(self.non_matched_smiles)} non-matched structures...")

                # Generate image using custom rendering to avoid RDKit PNG issues
                non_matched_path = os.path.join(self.temp_dir.name, f"{self.filter_name}_non_matched.png")
                self.render_molecules_to_file(self.non_matched_smiles[:500], non_matched_path, 5)

                # Load with QPixmap
                self.non_matched_pixmap = QPixmap(non_matched_path)

                if self.non_matched_pixmap.isNull():
                    raise ValueError(f"Failed to load pixmap from {non_matched_path}")

                print(f"Pixmap loaded: {self.non_matched_pixmap.width()}x{self.non_matched_pixmap.height()}")
                self.update_non_matched_display()

            except Exception as e:
                error_msg = f"Error displaying non-matched structures:\n{str(e)}\n\nType: {type(e).__name__}"
                print(error_msg)
                import traceback

                traceback.print_exc()
                self.non_matched_container.setText(error_msg)
        else:
            self.non_matched_container.setText("No non-matched structures")

    def render_molecules_to_file(self, smiles_list, output_path, mols_per_row=5):
        """
        Render molecules to a PNG file using SVG intermediate to avoid PNG header issues.

        Args:
            smiles_list: List of SMILES strings
            output_path: Path to save the PNG file
            mols_per_row: Number of molecules per row
        """
        from rdkit.Chem import Draw
        from rdkit import Chem

        try:
            from cairosvg import svg2png

            has_cairosvg = True
        except ImportError:
            has_cairosvg = False
            print("Warning: cairosvg not available, will use alternative method")

        # Convert SMILES to molecules
        mols = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    mols.append(mol)
            except Exception:
                pass

        if not mols:
            # Create empty image with message
            img = PILImage.new("RGB", (800, 200), color="white")
            draw = ImageDraw.Draw(img)
            draw.text((10, 90), "No valid molecules to display", fill="black")
            img.save(output_path, format="PNG")
            return

        print(f"Rendering {len(mols)} valid molecules...")

        # Calculate grid dimensions
        n_rows = math.ceil(len(mols) / mols_per_row)

        # Try SVG route if cairosvg is available
        if has_cairosvg:
            try:
                svg_data = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=(400, 400), maxMols=len(mols), useSVG=True)

                # Convert SVG to PNG using cairosvg
                png_data = svg2png(bytestring=svg_data)

                # Save PNG data
                with open(output_path, "wb") as f:
                    f.write(png_data)

                print(f"Successfully rendered using SVG->PNG conversion")
                return

            except Exception as e:
                print(f"SVG conversion failed: {e}, trying alternative method")

        # Fallback: Render individual molecules and combine manually
        print("Using fallback rendering method...")
        mol_size = (400, 400)

        # Create individual molecule images
        mol_images = []
        for mol in mols:
            try:
                # Generate SVG for single molecule
                svg = Draw.MolToImage(mol, size=mol_size, kekulize=True, wedgeBonds=True, fitImage=True)
                mol_images.append(svg)
            except Exception as e:
                print(f"Error rendering molecule: {e}")
                # Create blank image
                blank = PILImage.new("RGB", mol_size, color="white")
                mol_images.append(blank)

        # Create grid image
        grid_width = mols_per_row * mol_size[0]
        grid_height = n_rows * mol_size[1]
        grid_img = PILImage.new("RGB", (grid_width, grid_height), color="white")

        # Paste molecules into grid
        for idx, mol_img in enumerate(mol_images):
            row = idx // mols_per_row
            col = idx % mols_per_row
            x = col * mol_size[0]
            y = row * mol_size[1]
            grid_img.paste(mol_img, (x, y))

        # Save combined image
        grid_img.save(output_path, format="PNG")
        print(f"Successfully rendered using fallback method")

    def on_zoom_changed(self, value):
        """Handle zoom slider changes."""
        self.zoom_level = value
        self.zoom_label.setText(f"{value}%")
        self.update_matched_display()
        self.update_non_matched_display()

    def reset_zoom(self):
        """Reset zoom to 100%."""
        self.zoom_slider.setValue(100)

    def update_matched_display(self):
        """Update the matched structures display with current zoom."""
        if self.matched_pixmap and not self.matched_pixmap.isNull():
            try:
                scaled_width = int(self.matched_pixmap.width() * self.zoom_level / 100)
                scaled_height = int(self.matched_pixmap.height() * self.zoom_level / 100)
                scaled_pixmap = self.matched_pixmap.scaled(scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.matched_container.setPixmap(scaled_pixmap)
                self.matched_container.adjustSize()
            except Exception as e:
                print(f"Error updating matched display: {e}")
                import traceback

                traceback.print_exc()

    def update_non_matched_display(self):
        """Update the non-matched structures display with current zoom."""
        if self.non_matched_pixmap and not self.non_matched_pixmap.isNull():
            try:
                scaled_width = int(self.non_matched_pixmap.width() * self.zoom_level / 100)
                scaled_height = int(self.non_matched_pixmap.height() * self.zoom_level / 100)
                scaled_pixmap = self.non_matched_pixmap.scaled(scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.non_matched_container.setPixmap(scaled_pixmap)
                self.non_matched_container.adjustSize()
            except Exception as e:
                print(f"Error updating non-matched display: {e}")
                import traceback

                traceback.print_exc()

    def closeEvent(self, event):
        """Clean up temporary directory on close."""
        # Remove ourselves from parent's tracking list to allow GC
        if self.parent() and hasattr(self.parent(), "structure_windows"):
            try:
                self.parent().structure_windows.remove(self)
            except ValueError:
                pass
        self.temp_dir.cleanup()
        event.accept()


class DownloadThread(QThread):
    """Thread for downloading resources in the background."""

    progress_update = pyqtSignal(int)
    max_update = pyqtSignal(int)
    description_update = pyqtSignal(str)
    download_complete = pyqtSignal()
    download_error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._is_cancelled = False

    def cancel(self):
        """Cancel the download operation."""
        self._is_cancelled = True

    def run(self):
        """Run the download in a separate thread."""

        def update_progress(value):
            self.progress_update.emit(int(value))
            return True

        def set_max_progress(max_value):
            self.max_update.emit(int(max_value))

        def set_description(description):
            self.description_update.emit(description)

        try:
            # Download MSMS libraries
            set_description("Downloading MS/MS libraries...")
            Filters.download_MSMS_libraries(status_bar_update_func=update_progress, status_bar_max_func=set_max_progress, status_bar_description_func=set_description)

            # Download MS2DeepScore model
            set_description("Downloading MS2DeepScore model...")
            Filters.download_MS2DeepScore_model(status_bar_update_func=update_progress, status_bar_max_func=set_max_progress, status_bar_description_func=set_description)

            self.download_complete.emit()

        except Exception as e:
            print(str(e))
            self.download_error.emit(str(e))


class MGFFilterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mgf_files = {}  # {file_path: {fields, smiles_field, filters}}
        self.smarts_filters = {}  # {filter_name: smarts_string}
        # OPTIMIZED: Store only matching SMILES sets per filter (not entire blocks)
        # Format: {filter_name: set(matching_smiles)}
        self.filter_matched_smiles = {}  # Memory-efficient: only SMILES strings
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sections = []  # List to keep track of all sections
        self.structure_windows = []  # Keep track of open structure viewer windows
        # Track meta-field selections per file: {file_path: {meta_field: {selected_values}}}
        self.meta_field_selections = {}
        # Polars DataFrame for all MGF data
        self.df_data = None
        # Debounce timer for meta-value changes
        self._meta_values_debounce_timer = None
        # Download thread reference (prevent GC during download)
        self._download_thread = None

        # Define predefined SMARTS filters
        self.predefined_filters = self.get_predefined_filters()

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

    def get_predefined_filters(self):
        """Get dictionary of predefined SMARTS filters."""
        return OrderedDict(
            [
                (
                    "Prenyl Flavonoid",
                    "<<[O,o]~[C,c]~1~[C,c]~[C,c](~[O,o]~[C,c]2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~1~2)~[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]3 OR [C,c]~1~[C,c]~[C,c](~[O,o]~[C,c]2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~1~2)~[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]3>> AND CC=C(C)CCC=C(C)C",
                ),
                (
                    "Flavone",
                    "<<[O,o]~[C,c]~1~[C,c]~[C,c](~[O,o]~[C,c]2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~1~2)~[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]3 OR [C,c]~1~[C,c]~[C,c](~[O,o]~[C,c]2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~1~2)~[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]3>>",
                ),
                (
                    "Isoflavone",
                    "<<[O,o]~[C,c]~1~[C,c]~2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~2~[O,o]~[C,c]~[C,c]~1[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~3 OR [C,c]~1~[C,c]~2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~2~[O,o]~[C,c]~[C,c]~1[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~3>>",
                ),
                (
                    "Flavone or Isoflavone",
                    "<<[O,o]~[C,c]~1~[C,c]~[C,c](~[O,o]~[C,c]2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~1~2)~[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]3 OR [C,c]~1~[C,c]~[C,c](~[O,o]~[C,c]2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~1~2)~[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]3 OR [O,o]~[C,c]~1~[C,c]~2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~2~[O,o]~[C,c]~[C,c]~1[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~3 OR [C,c]~1~[C,c]~2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~2~[O,o]~[C,c]~[C,c]~1[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~3>>",
                ),
                (
                    "Prenyl Flavone or Isoflavone",
                    "<<[O,o]~[C,c]~1~[C,c]~[C,c](~[O,o]~[C,c]2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~1~2)~[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]3 OR [C,c]~1~[C,c]~[C,c](~[O,o]~[C,c]2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~1~2)~[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]3 OR [O,o]~[C,c]~1~[C,c]~2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~2~[O,o]~[C,c]~[C,c]~1[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~3 OR [C,c]~1~[C,c]~2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~2~[O,o]~[C,c]~[C,c]~1[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~3>> AND CC=C(C)CCC=C(C)C",
                ),
                ("Chalcone (saturated)", "[O,o]=[C,c](-[CH2]-[CH2]-[C,c]@1@[C,c]@[C,c]@[C,c]@[C,c]@[C,c]@1)-[C,c]@2@[C,c]@[C,c]@[C,c]@[C,c]@[C,c]@2"),
                ("Chalcone (unsaturated)", "[O,o]=[C,c](-[CH]=[CH]-[C,c]@1@[C,c]@[C,c]@[C,c]@[C,c]@[C,c]@1)-[C,c]@2@[C,c]@[C,c]@[C,c]@[C,c]@[C,c]@2"),
                ("Chromone", "O=C@1@C@C@O@C@2@C@C@C@C@C@12"),
                ("Geranyl", "CC=C(C)CCC=C(C)C"),
                ("Farnesyl", "[CH2][CH]=C([CH3])[CH2]([CH2][CH]=C([CH3])[CH2]([CH2][CH]=C([CH3])[CH3]))"),
            ]
        )

    @staticmethod
    def concat_dataframes_with_alignment(dataframes):
        """Concatenate multiple Polars DataFrames vertically (stacking rows).

        IMPORTANT: Each row remains completely separate and unique. Rows are NOT merged or combined.
        This function simply STACKS all rows from all DataFrames on top of each other.

        The 'diagonal' strategy handles the case where different DataFrames have different columns:
        - Columns with the same name are aligned
        - Missing columns in any DataFrame are filled with null values
        - All rows from all DataFrames are preserved as-is (no merging/joining)

        Example:
            df1: [A, B] with 2 rows
            df2: [B, C] with 3 rows
            Result: [A, B, C] with 5 rows total (df1 rows have C=null, df2 rows have A=null)

        Args:
            dataframes: List of Polars DataFrames to stack vertically

        Returns:
            Single concatenated Polars DataFrame with all rows from all inputs
        """
        if not dataframes:
            return None

        if len(dataframes) == 1:
            return dataframes[0]

        # Vertical concatenation with automatic column alignment
        # how="diagonal": stacks rows vertically, aligns columns by name, fills missing with null
        # This does NOT combine/merge rows - each row remains unique and separate
        return pl.concat(dataframes, how="diagonal")

    # ---- SMILES validity helpers (used in many places) ----

    _SMILES_NULL_STRINGS = ["", "n/a", "na", "none", "null"]

    @staticmethod
    def _is_valid_smiles_expr(col_name):
        """Return a Polars expression that is True when `col_name` contains a valid, non-empty SMILES string."""
        return pl.col(col_name).is_not_null() & (pl.col(col_name).cast(pl.Utf8).str.len_bytes() > 0) & ~pl.col(col_name).cast(pl.Utf8).str.to_lowercase().is_in(MGFFilterGUI._SMILES_NULL_STRINGS)

    @staticmethod
    def _is_valid_smiles_value(s):
        """Return True when a Python string represents a valid, non-empty SMILES."""
        return bool(s) and str(s).strip() != "" and str(s).strip().lower() not in MGFFilterGUI._SMILES_NULL_STRINGS

    def get_blocks_from_dataframe(self, filter_expr=None):
        """Reconstruct blocks from DataFrame.

        Args:
            filter_expr: Optional Polars expression to filter rows. If None, returns all rows.

        Returns:
            List of block dictionaries
        """
        if self.df_data is None:
            return []

        # Apply filter if provided
        if filter_expr is not None:
            df = self.df_data.filter(filter_expr)
        else:
            df = self.df_data

        # Convert DataFrame to list of dictionaries, excluding internal columns
        blocks = []
        internal_cols = [col for col in df.columns if col.startswith("__AnnoMe_")]
        data_cols = [col for col in df.columns if col not in internal_cols]

        for row in df.select(data_cols).iter_rows(named=True):
            # Remove None values
            block = {k: v for k, v in row.items() if v is not None}
            blocks.append(block)

        return blocks

    def rebuild_dataframe_from_mgf_files(self):
        """Rebuild the Polars DataFrame from all loaded MGF files.

        This method is now primarily used when adding new SMARTS filters to existing data.
        For initial loading, DataFrames are created during file loading.
        """
        if not self.mgf_files:
            self.df_data = None
            return

        # If DataFrame doesn't exist, create it from blocks (fallback)
        if self.df_data is None:
            # Collect all unique meta keys from all files
            all_meta_keys = set()
            for file_data in self.mgf_files.values():
                for block in file_data["blocks"]:
                    # Exclude special keys
                    for key in block.keys():
                        if key not in ["$$spectrumdata", "$$spectrumData", "peaks"]:
                            all_meta_keys.add(key)

            # Build list of records (one per spectrum)
            records = []
            for file_path, file_data in self.mgf_files.items():
                source_name = os.path.basename(file_path)

                for block in file_data["blocks"]:
                    # Start with all meta keys set to None
                    record = {key: None for key in all_meta_keys}

                    # Fill in values that exist in this block
                    for key, value in block.items():
                        if key in all_meta_keys:
                            record[key] = value

                    # Add internal fields
                    record["__AnnoMe_source"] = source_name
                    record["__AnnoMe_source_path"] = file_path
                    record["__AnnoMe_meta_filter"] = True  # Initially all pass meta filter
                    record["__AnnoMe_smiles_required"] = file_data.get("smiles_required", False)

                    # REMOVED: No longer store SMARTS filter results in DataFrame columns
                    # Filter results are now stored as sets of matching SMILES in filter_matched_smiles

                    records.append(record)

            # Create Polars DataFrame
            if records:
                self.df_data = pl.DataFrame(records)
                print(f"Created DataFrame with {len(self.df_data)} rows and {len(self.df_data.columns)} columns")
            else:
                self.df_data = None
        else:
            # REMOVED: No longer add SMARTS filter columns to DataFrame
            # Filter results are stored separately in filter_matched_smiles

            # Update smiles_required column using Polars-native replace (avoids slow map_elements)
            smiles_req_paths = [path for path, data in self.mgf_files.items() if data.get("smiles_required", False)]
            self.df_data = self.df_data.with_columns([pl.col("__AnnoMe_source_path").is_in(smiles_req_paths).alias("__AnnoMe_smiles_required")])

    def compute_all_file_statistics(self):
        """Compute statistics for all files in one efficient pass using group_by aggregation.

        Returns:
            dict: {source_name: {total, filtered, no_smiles}}
        """
        if self.df_data is None or self.df_data.is_empty():
            return {}

        # Compute basic statistics with group_by (one pass through data)
        stats_df = self.df_data.group_by("__AnnoMe_source").agg(
            [
                pl.count().alias("total"),
                pl.col("__AnnoMe_meta_filter").sum().cast(pl.Int64).alias("filtered"),
                pl.col("__AnnoMe_has_valid_smiles").is_not_null().sum().cast(pl.Int64).alias("has_smiles"),
            ]
        )

        # Convert to dict for fast lookup
        stats_dict = {}
        for row in stats_df.iter_rows(named=True):
            source = row["__AnnoMe_source"]
            stats_dict[source] = {
                "total": row["total"],
                "filtered": row["filtered"],
                "no_smiles": row["total"] - row["has_smiles"],  # Calculate missing SMILES
            }

        return stats_dict

    def update_meta_filter_in_dataframe(self):
        """Update the __AnnoMe_meta_filter column in the DataFrame based on meta-field selections and SMILES requirements."""
        if self.df_data is None:
            return

        # Start with all rows passing the filter
        filter_mask = pl.lit(True)

        # Apply meta-field filters for each file
        for file_path, field_selections in self.meta_field_selections.items():
            if not field_selections:
                continue

            source_name = os.path.basename(file_path)

            for field_name, selected_values in field_selections.items():
                if not selected_values:
                    # No values selected = filter out all rows from this file with this field
                    file_filter = ~(pl.col("__AnnoMe_source") == source_name)
                else:
                    # Keep only rows from this file that match selected values
                    file_filter = (pl.col("__AnnoMe_source") != source_name) | (pl.col(field_name).cast(pl.Utf8).is_in(list(selected_values)))

                filter_mask = filter_mask & file_filter

        # Apply SMILES requirement filter
        for file_path, file_data in self.mgf_files.items():
            if file_data.get("smiles_required", False):
                source_name = os.path.basename(file_path)
                smiles_field = file_data.get("smiles_field")

                if smiles_field and smiles_field in self.df_data.columns:
                    # For this file, SMILES must be present and non-empty
                    smiles_filter = (pl.col("__AnnoMe_source") != source_name) | self._is_valid_smiles_expr(smiles_field)
                    filter_mask = filter_mask & smiles_filter

        # Update the DataFrame with the new filter column
        self.df_data = self.df_data.with_columns([filter_mask.alias("__AnnoMe_meta_filter")])

        filtered_count = self.df_data.filter(pl.col("__AnnoMe_meta_filter")).select(pl.count()).item()
        total_count = len(self.df_data)
        print(f"Updated meta filter: {filtered_count} / {total_count} rows pass filter")

    def init_ui(self):
        version = self.get_version()
        self.setWindowTitle(f"AnnoMe MGF SMARTS Filter Wizard v{version}")
        self.setGeometry(100, 100, 1400, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Create tab widget with tabs on the left
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.West)

        # Section 1: Load MGF Files
        section1_widget = QWidget()
        section1_scroll = QScrollArea()
        section1_scroll.setWidgetResizable(True)
        section1_scroll.setWidget(section1_widget)
        self.init_section1(section1_widget)
        self.tab_widget.addTab(section1_scroll, "1. Load MGF Files")

        # Section 2: Canonicalization
        section2_widget = QWidget()
        section2_scroll = QScrollArea()
        section2_scroll.setWidgetResizable(True)
        section2_scroll.setWidget(section2_widget)
        self.init_section2(section2_widget)
        self.tab_widget.addTab(section2_scroll, "2. SMILES Canonicalization")

        # Section 3: SMARTS Filters
        section3_widget = QWidget()
        section3_scroll = QScrollArea()
        section3_scroll.setWidgetResizable(True)
        section3_scroll.setWidget(section3_widget)
        self.init_section3(section3_widget)
        self.tab_widget.addTab(section3_scroll, "3. Define SMARTS Filters")

        # Section 4: Export Results
        section4_widget = QWidget()
        section4_scroll = QScrollArea()
        section4_scroll.setWidgetResizable(True)
        section4_scroll.setWidget(section4_widget)
        self.init_section4(section4_widget)
        self.tab_widget.addTab(section4_scroll, "4. Export Filtered Results")

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tab_widget)
        main_widget.setLayout(main_layout)

    def init_section1(self, parent_widget=None):
        """Initialize the MGF file loading section."""
        if parent_widget is None:
            parent_widget = QWidget()
        content = QWidget()
        main_layout = QHBoxLayout()

        # Help text on the left
        help_group = QGroupBox("Help")
        help_layout = QVBoxLayout()
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
            <h3>Load MGF Files</h3>
            <p>This section allows you to load one or more MGF files for filtering.</p>
            <ol>
                <li><b>Load MGF File:</b> Click to browse and load MGF files (can select multiple)</li>
                <li><b>SMILES Required:</b> Toggle to filter out spectra without SMILES information</li>
                <li><b>Select SMILES Field:</b> Choose which field contains the SMILES strings</li>
                <li><b>Filter by Meta-Field:</b> Optionally filter spectra by specific field values</li>
                <li><b>Statistics:</b> View how many spectra are loaded, filtered, and removed</li>
            </ol>
            <p><b>Tip:</b> You can select multiple files in the table (Ctrl+Click or Shift+Click) to apply settings to all at once.</p>
        """)
        help_layout.addWidget(help_text)
        help_group.setLayout(help_layout)
        help_group.setMaximumWidth(350)
        main_layout.addWidget(help_group, 1)  # 25% width

        # Controls on the right
        controls = QWidget()
        layout = QVBoxLayout()

        # Load button and Download libraries button
        load_layout = QHBoxLayout()
        load_btn = QPushButton("Load MGF File(s)")
        load_btn.clicked.connect(self.load_mgf_file)
        load_layout.addWidget(load_btn)

        download_btn = QPushButton("Download default libraries")
        download_btn.clicked.connect(self.download_default_libraries)
        load_layout.addWidget(download_btn)

        load_layout.addStretch()
        layout.addLayout(load_layout)

        # File table
        layout.addWidget(QLabel("Loaded MGF Files (select one or more):"))
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(7)
        self.file_table.setHorizontalHeaderLabels(["File Name", "Total Entries", "No SMILES", "SMILES Field", "Filtered", "Removed", "Meta-Filters"])
        self.file_table.horizontalHeader().setStretchLastSection(True)
        self.file_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.file_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_table.itemSelectionChanged.connect(self.on_file_selection_changed)

        # Adjust column widths
        header = self.file_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # File Name
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Total
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # No SMILES
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # SMILES Field
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Filtered
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Removed
        header.setSectionResizeMode(6, QHeaderView.Stretch)  # Meta-Filters

        layout.addWidget(self.file_table)

        # File details
        details_group = QGroupBox("File Details (applies to selected files)")
        details_layout = QVBoxLayout()

        # SMILES field selection with SMILES Required checkbox
        smiles_layout = QHBoxLayout()
        smiles_layout.addWidget(QLabel("SMILES Field:"))
        self.smiles_field_combo = QComboBox()
        self.smiles_field_combo.currentTextChanged.connect(self.on_smiles_field_changed)
        smiles_layout.addWidget(self.smiles_field_combo)
        apply_smiles_btn = QPushButton("Apply to Selected")
        apply_smiles_btn.clicked.connect(self.apply_smiles_field_to_selected)
        smiles_layout.addWidget(apply_smiles_btn)
        self.smiles_required_checkbox = QCheckBox("SMILES Required")
        self.smiles_required_checkbox.setChecked(False)
        self.smiles_required_checkbox.stateChanged.connect(self.on_smiles_required_changed)
        smiles_layout.addWidget(self.smiles_required_checkbox)
        details_layout.addLayout(smiles_layout)

        # Meta-field filtering
        details_layout.addWidget(QLabel("Filter by Meta-Field:"))
        self.meta_field_combo = QComboBox()
        self.meta_field_combo.currentTextChanged.connect(self.on_meta_field_changed)
        details_layout.addWidget(self.meta_field_combo)

        self.meta_values_list = QListWidget()
        self.meta_values_list.setSelectionMode(QListWidget.MultiSelection)
        self.meta_values_list.itemSelectionChanged.connect(self.on_meta_values_changed)
        details_layout.addWidget(self.meta_values_list)

        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

        controls.setLayout(layout)
        main_layout.addWidget(controls, 3)  # 75% width

        content.setLayout(main_layout)
        parent_widget.setLayout(QVBoxLayout())
        parent_widget.layout().addWidget(content)

    def init_section2(self, parent_widget=None):
        """Initialize the canonicalization section."""
        if parent_widget is None:
            parent_widget = QWidget()
        content = QWidget()
        main_layout = QHBoxLayout()

        # Help text on the left
        help_group = QGroupBox("Help")
        help_layout = QVBoxLayout()
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
            <h3>SMILES Canonicalization</h3>
            <p>Canonicalization converts all SMILES strings to a standard format.</p>
            <p><b>Why canonicalize?</b></p>
            <ul>
                <li>Different SMILES can represent the same molecule</li>
                <li>Canonical form ensures consistent comparisons</li>
                <li>Helps identify duplicates</li>
            </ul>
            <p><b>Note:</b> This process may take some time for large datasets. A progress bar will show the status.</p>
            <p><b>Tip:</b> This step is optional but recommended for accurate filtering.</p>
        """)
        help_layout.addWidget(help_text)
        help_group.setLayout(help_layout)
        help_group.setMaximumWidth(350)
        main_layout.addWidget(help_group, 1)  # 25% width

        # Controls on the right
        controls = QWidget()
        layout = QVBoxLayout()

        # Canonicalization
        self.canon_checkbox = QCheckBox("Calculate Canonical SMILES")
        self.canon_checkbox.setChecked(True)
        layout.addWidget(self.canon_checkbox)

        canon_btn = QPushButton("Apply Canonicalization")
        canon_btn.clicked.connect(self.apply_canonicalization)
        layout.addWidget(canon_btn)

        self.canon_status_label = QLabel("Status: Not applied")
        layout.addWidget(self.canon_status_label)

        layout.addStretch()

        controls.setLayout(layout)
        main_layout.addWidget(controls, 3)  # 75% width

        content.setLayout(main_layout)
        parent_widget.setLayout(QVBoxLayout())
        parent_widget.layout().addWidget(content)

    def init_section3(self, parent_widget=None):
        """Initialize the SMARTS filter section."""
        if parent_widget is None:
            parent_widget = QWidget()
        content = QWidget()
        main_layout = QHBoxLayout()

        # Help text on the left
        help_group = QGroupBox("Help")
        help_layout = QVBoxLayout()
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
            <h3>Define SMARTS Filters</h3>
            <p>SMARTS (SMiles ARbitrary Target Specification) is a pattern language for chemical structures.</p>
            <p><b>How to use:</b></p>
            <ol>
                <li>Enter a filter name</li>
                <li>Enter SMARTS pattern(s) using AND/OR logic</li>
                <li>Use <b>&lt;&lt; ... &gt;&gt;</b> for OR groups, <b>AND</b> to separate groups</li>
                <li>Click "Add Filter" or press Enter</li>
                <li>View statistics in the table</li>
                <li>Click "View Structures" to see matched/non-matched compounds</li>
                <li>Edit existing filters by double-clicking them</li>
                <li>Use "Predefined Filters" button for common patterns</li>
                <li>Use "Load Filters from JSON" to import multiple filters at once</li>
            </ol>
            <p><b>Example 1:</b> Simple AND:<br>
            <code>c1ccccc1 AND [OH]</code></p>
            <p><b>Example 2:</b> OR group (flavone OR isoflavone):<br>
            <code>&lt;&lt;flavone_smarts OR isoflavone_smarts&gt;&gt;</code></p>
            <p><b>Example 3:</b> Combined (flavone OR isoflavone) AND prenyl:<br>
            <code>&lt;&lt;flavone_smarts OR isoflavone_smarts&gt;&gt; AND CC=C(C)CCC=C(C)C</code></p>
            <p><b>JSON Format:</b> {"filter_name": "smarts_pattern", ...}</p>
            <p><b>Note:</b> Use AND (case-insensitive) to separate patterns that must all match, and &lt;&lt; pattern1 OR pattern2 &gt;&gt; for alternatives.</p>
            <p><b>Tip:</b> Use tools like <a href="https://smarts.plus">smarts.plus</a> to build and test SMARTS patterns.</p>
        """)
        help_layout.addWidget(help_text)
        help_group.setLayout(help_layout)
        help_group.setMaximumWidth(350)
        main_layout.addWidget(help_group, 1)  # 25% width

        # Controls on the right
        controls = QWidget()
        layout = QVBoxLayout()

        # Filter Name input - first line
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Filter Name:"))
        self.filter_name_input = QLineEdit()
        name_layout.addWidget(self.filter_name_input)
        layout.addLayout(name_layout)

        # SMARTS input - second line
        smarts_layout = QHBoxLayout()
        smarts_layout.addWidget(QLabel("SMARTS (AND/OR logic):"))
        self.smarts_input = QLineEdit()
        self.smarts_input.returnPressed.connect(self.add_smarts_filter)
        smarts_layout.addWidget(self.smarts_input)
        layout.addLayout(smarts_layout)

        # Buttons - third line
        buttons_layout = QHBoxLayout()
        add_btn = QPushButton("Add/Update Filter")
        add_btn.clicked.connect(self.add_smarts_filter)
        buttons_layout.addWidget(add_btn)

        predefined_btn = QPushButton("Predefined Filters")
        predefined_btn.clicked.connect(self.show_predefined_filters_menu)
        buttons_layout.addWidget(predefined_btn)

        load_json_btn = QPushButton("Load Filters from JSON")
        load_json_btn.clicked.connect(self.load_filters_from_json)
        buttons_layout.addWidget(load_json_btn)

        save_json_btn = QPushButton("Save Defined Filters")
        save_json_btn.clicked.connect(self.save_filters_to_json)
        buttons_layout.addWidget(save_json_btn)

        buttons_layout.addStretch()

        delete_all_btn = QPushButton("Delete All Filters")
        delete_all_btn.clicked.connect(self.delete_all_filters)
        buttons_layout.addWidget(delete_all_btn)

        layout.addLayout(buttons_layout)

        # Filter table label (no control buttons here anymore)
        layout.addWidget(QLabel("Defined Filters (right-click for options):"))

        # Filter table
        self.filter_table = QTableWidget()
        self.filter_table.setColumnCount(4)
        self.filter_table.setHorizontalHeaderLabels(["Filter Name", "Total SMILES", "Matched", "Non-Matched"])
        self.filter_table.horizontalHeader().setStretchLastSection(False)
        self.filter_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.filter_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.filter_table.itemDoubleClicked.connect(self.edit_selected_filter)
        self.filter_table.itemSelectionChanged.connect(self.on_filter_selection_changed)
        self.filter_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.filter_table.customContextMenuRequested.connect(self.show_filter_table_context_menu)
        self.filter_table.setSortingEnabled(True)

        # Adjust column widths
        header = self.filter_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # Filter Name
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Total
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Matched
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Non-Matched

        layout.addWidget(self.filter_table)

        # SMARTS Visualization Section
        layout.addWidget(QLabel("SMARTS Pattern Visualization:"))
        self.smarts_viz_scroll = QScrollArea()
        self.smarts_viz_scroll.setWidgetResizable(True)
        self.smarts_viz_scroll.setMinimumHeight(300)
        self.smarts_viz_container = QLabel()
        self.smarts_viz_container.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.smarts_viz_container.setText("Select a filter to view its SMARTS patterns")
        self.smarts_viz_scroll.setWidget(self.smarts_viz_container)
        layout.addWidget(self.smarts_viz_scroll)

        # Store pixmap for visualization
        self.smarts_viz_pixmap = None

        controls.setLayout(layout)
        main_layout.addWidget(controls, 3)  # 75% width

        content.setLayout(main_layout)
        parent_widget.setLayout(QVBoxLayout())
        parent_widget.layout().addWidget(content)

    def show_predefined_filters_menu(self):
        """Show a context menu with predefined SMARTS filters."""
        menu = QMenu(self)

        for filter_name, smarts_pattern in self.predefined_filters.items():
            action = menu.addAction(filter_name)
            action.triggered.connect(lambda checked, fn=filter_name, sp=smarts_pattern: self.load_predefined_filter(fn, sp))

        # Show menu at button position
        sender = self.sender()
        menu.exec_(sender.mapToGlobal(sender.rect().bottomLeft()))

    def show_filter_table_context_menu(self, position):
        """Show context menu for filter table."""
        # Check if a row is selected
        selected_rows = self.filter_table.selectedItems()
        if not selected_rows:
            return

        menu = QMenu(self)

        edit_action = menu.addAction("Edit Selected")
        edit_action.triggered.connect(self.edit_selected_filter)

        view_action = menu.addAction("View Structures")
        view_action.triggered.connect(self.view_structures)

        menu.addSeparator()

        delete_action = menu.addAction("Delete Selected")
        delete_action.triggered.connect(self.delete_selected_filter)

        # Show menu at cursor position
        menu.exec_(self.filter_table.viewport().mapToGlobal(position))

    def load_predefined_filter(self, filter_name, smarts_pattern):
        """Load a predefined filter into the input fields."""
        self.filter_name_input.setText(filter_name)
        self.smarts_input.setText(smarts_pattern)
        QMessageBox.information(self, "Filter Loaded", f"Predefined filter '{filter_name}' has been loaded.\n\nYou can modify the name or SMARTS pattern before adding it.")

    def init_section4(self, parent_widget=None):
        """Initialize the export section."""
        if parent_widget is None:
            parent_widget = QWidget()
        content = QWidget()
        main_layout = QHBoxLayout()

        # Help text on the left
        help_group = QGroupBox("Help")
        help_layout = QVBoxLayout()
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
            <h3>Export Filtered Results</h3>
            <p>Export your filtered spectra to MGF files.</p>
            <p><b>Output files:</b></p>
            <ul>
                <li><code>&lt;base_name&gt;_&lt;filter_name&gt;_matched.mgf</code> - Spectra matching the filter</li>
                <li><code>&lt;base_name&gt;_&lt;filter_name&gt;_noMatch.mgf</code> - Spectra not matching the filter</li>
            </ul>
            <p><b>Steps:</b></p>
            <ol>
                <li>Select which filters to export (or export all)</li>
                <li>Choose base output file name</li>
                <li>Click "Export Filtered MGF Files"</li>
            </ol>
            <p><b>Note:</b> All loaded MGF files will be combined in the export. Each filter will create two files with the filter name as a suffix.</p>
        """)
        help_layout.addWidget(help_text)
        help_group.setLayout(help_layout)
        help_group.setMaximumWidth(350)
        main_layout.addWidget(help_group, 1)  # 25% width

        # Controls on the right
        controls = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Select filters to export:"))

        # Checkbox for "Export All"
        export_all_layout = QHBoxLayout()
        self.export_all_checkbox = QCheckBox("Export All Filters")
        self.export_all_checkbox.setChecked(True)
        self.export_all_checkbox.stateChanged.connect(self.on_export_all_changed)
        export_all_layout.addWidget(self.export_all_checkbox)
        export_all_layout.addStretch()
        layout.addLayout(export_all_layout)

        # Checkbox for "Export Overview Excel"
        export_excel_layout = QHBoxLayout()
        self.export_excel_checkbox = QCheckBox("Export Overview Excel (with SMILES images and match statistics)")
        self.export_excel_checkbox.setChecked(True)
        self.export_excel_checkbox.setToolTip(
            "Generate an Excel file with:\n- Unique canonicalized SMILES codes\n- Structure images\n- Source MGF files\n- Spectrum counts per SMILES\n- Substructure match results for each filter"
        )
        export_excel_layout.addWidget(self.export_excel_checkbox)
        export_excel_layout.addStretch()
        layout.addLayout(export_excel_layout)

        # List of filters with checkboxes
        self.export_filter_list = QListWidget()
        self.export_filter_list.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(self.export_filter_list)

        export_layout = QHBoxLayout()
        export_layout.addWidget(QLabel("Base Output File:"))
        self.output_file_input = QLineEdit()
        export_layout.addWidget(self.output_file_input)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_output_file)
        export_layout.addWidget(browse_btn)
        layout.addLayout(export_layout)

        export_btn = QPushButton("Export Filtered MGF Files")
        export_btn.clicked.connect(self.export_results)
        layout.addWidget(export_btn)

        layout.addStretch()

        controls.setLayout(layout)
        main_layout.addWidget(controls, 3)  # 75% width

        content.setLayout(main_layout)
        parent_widget.setLayout(QVBoxLayout())
        parent_widget.layout().addWidget(content)

    @staticmethod
    def _load_single_mgf_file(file_path):
        """Worker function to load a single MGF file and return as Polars DataFrame."""
        try:
            start_time = time.time()
            # Parse MGF file and get Polars DataFrame directly
            df = Filters.parse_mgf_file(file_path, check_required_keys=False, return_as_polars_table=True)

            if df is None or df.is_empty():
                print(f"{Fore.YELLOW}Warning: No valid spectra found in {file_path}{Style.RESET_ALL}")
                return {
                    "file_path": file_path,
                    "fields": [],
                    "smiles_field": None,
                    "df": None,
                    "success": True,
                    "error": None,
                }

            # Get fields from DataFrame columns (excluding internal columns)
            fields = [col for col in df.columns if not col.startswith("__AnnoMe_") and col not in ["$$spectrumdata", "$$spectrumData", "peaks"]]

            # Auto-detect SMILES field (case-insensitive)
            smiles_field = None
            for field in fields:
                if field.lower() == "smiles":
                    smiles_field = field
                    break

            # Add internal tracking columns
            source_name = os.path.basename(file_path)
            df = df.with_columns([pl.lit(source_name).alias("__AnnoMe_source"), pl.lit(file_path).alias("__AnnoMe_source_path")])

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"{Fore.GREEN}Successfully loaded {file_path} with {len(df)} spectra in {elapsed_time:.2f} seconds{Style.RESET_ALL}")

            return {
                "file_path": file_path,
                "fields": fields,
                "smiles_field": smiles_field,
                "df": df,
                "success": True,
                "error": None,
            }
        except Exception as e:
            return {"file_path": file_path, "success": False, "error": str(e)}

    def load_mgf_file(self):
        """Load one or more MGF files."""
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select MGF File(s)", "", "MGF Files (*.mgf);;All Files (*)")

        if not file_paths:
            return

        progress = QProgressDialog("Loading MGF files...", "Cancel", 0, len(file_paths), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QApplication.processEvents()

        loaded_count = 0
        error_count = 0

        # Load files sequentially
        print(f"Loading {len(file_paths)} MGF file(s)...")

        dataframes_to_concat = []

        try:
            # Process files sequentially
            for idx, file_path in enumerate(file_paths):
                if progress.wasCanceled():
                    break

                progress.setValue(idx)
                progress.setLabelText(f"Loading {os.path.basename(file_path)}...")
                QApplication.processEvents()

                # Load file directly
                result = self._load_single_mgf_file(file_path)

                if result["success"]:
                    fields = result["fields"]
                    smiles_field = result["smiles_field"]
                    df = result["df"]

                    if smiles_field:
                        print(f"   - Auto-detected SMILES field: {smiles_field}")

                    self.mgf_files[file_path] = {
                        "fields": fields,
                        "smiles_field": smiles_field,
                        "meta_filters": {},
                        "canonicalized": False,
                        "smiles_required": self.smiles_required_checkbox.isChecked(),
                    }

                    if df is not None:
                        dataframes_to_concat.append(df)

                    loaded_count += 1
                else:
                    error_count += 1
                    print(f"Error loading {result['file_path']}: {result['error']}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during loading: {str(e)}")
            return

        progress.setValue(len(file_paths))

        # Combine all DataFrames and add internal columns
        if loaded_count > 0 and dataframes_to_concat:
            # Stack all DataFrames vertically (each row remains unique and separate)
            # Diagonal concatenation handles different metadata columns across files:
            # - Aligns columns by name
            # - Fills missing columns with null
            # - Does NOT merge/combine rows - simply stacks them
            self.df_data = self.concat_dataframes_with_alignment(dataframes_to_concat)

            print(f"Stacked {len(dataframes_to_concat)} DataFrames vertically")
            print(f"Total rows: {len(self.df_data)}, Total columns: {len(self.df_data.columns)}")
            print(f"Metadata columns: {len([col for col in self.df_data.columns if not col.startswith('__AnnoMe_')])}")

            # Optimize DataFrame dtypes to reduce memory usage
            # This converts text columns to Int/Float/Categorical where appropriate
            self.df_data = Filters.optimize_dataframe_dtypes(self.df_data)

            # Precompute SMILES validity for all files (do this ONCE, not per filter)
            # This dramatically reduces memory usage by avoiding repeated string operations
            smiles_validity_columns = []
            for file_path, file_data in self.mgf_files.items():
                smiles_field = file_data.get("smiles_field")
                if smiles_field and smiles_field in self.df_data.columns:
                    source_name = os.path.basename(file_path)
                    # For this source, check if SMILES field is valid
                    validity_expr = (
                        pl.when(pl.col("__AnnoMe_source") == source_name)
                        .then(
                            pl.when(self._is_valid_smiles_expr(smiles_field)).then(True).otherwise(None)  # Use None for invalid, so we can aggregate
                        )
                        .otherwise(None)
                    )
                    smiles_validity_columns.append(validity_expr)

            # Combine all validity checks into one column (coalesce takes first non-null)
            if smiles_validity_columns:
                has_valid_smiles = smiles_validity_columns[0]
                for expr in smiles_validity_columns[1:]:
                    has_valid_smiles = has_valid_smiles.fill_null(expr)
            else:
                has_valid_smiles = pl.lit(None)

            # Add internal tracking columns
            smiles_req_paths = [path for path, data in self.mgf_files.items() if data.get("smiles_required", False)]
            self.df_data = self.df_data.with_columns(
                [
                    pl.lit(True).alias("__AnnoMe_meta_filter"),
                    pl.col("__AnnoMe_source_path").is_in(smiles_req_paths).alias("__AnnoMe_smiles_required"),
                    has_valid_smiles.alias("__AnnoMe_has_valid_smiles"),
                ]
            )

            # Compute all statistics in ONE pass (much more efficient)
            print("Computing file statistics...")
            cached_stats = self.compute_all_file_statistics()
            print(f"Statistics computed for {len(cached_stats)} file(s)")

            # Add only newly loaded files to table
            for file_path in file_paths:
                if file_path in self.mgf_files:
                    # Check if this file is already in the table
                    file_name = os.path.basename(file_path)
                    already_in_table = False
                    for row in range(self.file_table.rowCount()):
                        if self.file_table.item(row, 0).text() == file_name:
                            already_in_table = True
                            break

                    if not already_in_table:
                        # Add new row with cached stats
                        self.add_file_to_table(file_path, cached_stats=cached_stats)
                    else:
                        # Update existing row with cached stats
                        print(f"Updating statistics for {file_name} in file table")
                        self.update_file_in_table(file_path, cached_stats=cached_stats)

            # Select the first newly loaded file
            if loaded_count > 0:
                self.file_table.selectRow(self.file_table.rowCount() - 1)

        msg = f"Successfully loaded {loaded_count} file(s)"
        if error_count > 0:
            msg += f"\n{error_count} file(s) failed to load"

        QMessageBox.information(self, "Load Complete", msg)

    def download_default_libraries(self):
        """Download default MS/MS libraries and models."""
        # Show information dialog
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Download Resources")
        msg.setText("This will download necessary MS/MS libraries and models for AnnoMe.")
        msg.setInformativeText(
            "Please note that some of these resources are rather large and in total are approximately 11 gigabytes in size.\n\n"
            "Make sure you have sufficient disk space and a stable internet connection before proceeding.\n\n"
            "Furthermore, please be aware that downloading these resources may take a considerable amount of time depending on your internet speed.\n\n"
            "Please do not proceed if you are on a metered connection.\n\n"
            "Do you wish to continue?"
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)

        if msg.exec_() != QMessageBox.Yes:
            return

        # Create progress dialog
        progress = QProgressDialog("Preparing download...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Downloading Resources")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        # Create and configure download thread (stored as instance attr to prevent GC)
        self._download_thread = DownloadThread()

        # Connect thread signals to progress dialog updates
        self._download_thread.progress_update.connect(progress.setValue)
        self._download_thread.max_update.connect(progress.setMaximum)
        self._download_thread.description_update.connect(progress.setLabelText)

        # Handle download completion
        def on_download_complete():
            progress.setValue(progress.maximum())
            progress.canceled.disconnect(on_cancel)  # Disconnect cancel handler
            progress.close()
            self._download_thread = None
            QMessageBox.information(self, "Download Complete", "All resources have been downloaded successfully!")

        # Handle download error
        def on_download_error(error_msg):
            progress.close()
            self._download_thread = None
            QMessageBox.critical(self, "Download Error", f"An error occurred during download:\n{error_msg}")

        # Handle cancellation
        def on_cancel():
            self._download_thread.cancel()
            self._download_thread.terminate()  # Forcefully kill the thread
            self._download_thread.wait()  # Wait for thread cleanup
            self._download_thread = None
            QMessageBox.warning(self, "Download Cancelled", "Download was cancelled by user.")

        self._download_thread.download_complete.connect(on_download_complete)
        self._download_thread.download_error.connect(on_download_error)
        progress.canceled.connect(on_cancel)

        # Start the download thread
        self._download_thread.start()

    def on_smiles_required_changed(self, state):
        """Handle SMILES required checkbox state change."""
        smiles_required = self.smiles_required_checkbox.isChecked()

        # Update all loaded files
        for file_path, file_data in self.mgf_files.items():
            file_data["smiles_required"] = smiles_required

        # Rebuild DataFrame with updated smiles_required flags
        if self.mgf_files:
            self.rebuild_dataframe_from_mgf_files()
            self.update_meta_filter_in_dataframe()

            # Compute all statistics in ONE pass, then update table
            cached_stats = self.compute_all_file_statistics()
            for file_path in self.mgf_files.keys():
                self.update_file_in_table(file_path, cached_stats=cached_stats)

    def add_file_to_table(self, file_path, cached_stats=None):
        """Add a file to the file table.

        Args:
            file_path: Path to the MGF file
            cached_stats: Optional dict from compute_all_file_statistics() to avoid recomputing
        """
        file_data = self.mgf_files[file_path]
        source_name = os.path.basename(file_path)

        row_position = self.file_table.rowCount()
        self.file_table.insertRow(row_position)

        # File Name
        self.file_table.setItem(row_position, 0, QTableWidgetItem(source_name))

        # Get statistics (either from cache or compute on demand)
        if cached_stats and source_name in cached_stats:
            stats = cached_stats[source_name]
            total = stats["total"]
            no_smiles_count = stats["no_smiles"]
            filtered = stats["filtered"]
        else:
            # Fallback: compute individually (slower)
            if self.df_data is not None:
                total = self.df_data.filter(pl.col("__AnnoMe_source") == source_name).select(pl.count()).item()
                filtered = self.df_data.filter((pl.col("__AnnoMe_source") == source_name) & pl.col("__AnnoMe_meta_filter")).select(pl.count()).item()
                no_smiles_count = self.df_data.filter((pl.col("__AnnoMe_source") == source_name) & pl.col("__AnnoMe_has_valid_smiles").is_null()).select(pl.count()).item()
            else:
                total = 0
                filtered = 0
                no_smiles_count = 0

        removed = total - filtered

        # Populate table cells
        self.file_table.setItem(row_position, 1, QTableWidgetItem(str(total)))
        self.file_table.setItem(row_position, 2, QTableWidgetItem(str(no_smiles_count)))

        smiles_field = file_data.get("smiles_field", "")
        self.file_table.setItem(row_position, 3, QTableWidgetItem(smiles_field if smiles_field else "-"))

        self.file_table.setItem(row_position, 4, QTableWidgetItem(str(filtered)))
        self.file_table.setItem(row_position, 5, QTableWidgetItem(str(removed)))

        # Meta-Filters
        meta_filters = file_data.get("meta_filters", {})
        filter_text = ", ".join([f"{k}({len(v)})" for k, v in meta_filters.items()]) if meta_filters else "-"
        self.file_table.setItem(row_position, 6, QTableWidgetItem(filter_text))

    def update_file_in_table(self, file_path, cached_stats=None):
        """Update a file's information in the table using DataFrame.

        Args:
            file_path: Path to the MGF file
            cached_stats: Optional dict from compute_all_file_statistics() to avoid recomputing
        """
        # Find the row for this file
        file_name = os.path.basename(file_path)
        for row in range(self.file_table.rowCount()):
            if self.file_table.item(row, 0).text() == file_name:
                file_data = self.mgf_files[file_path]
                source_name = file_name

                # Get statistics (either from cache or compute on demand)
                if cached_stats and source_name in cached_stats:
                    stats = cached_stats[source_name]
                    total = stats["total"]
                    no_smiles_count = stats["no_smiles"]
                    filtered_count = stats["filtered"]
                else:
                    # Fallback: compute individually (slower)
                    if self.df_data is not None:
                        total = self.df_data.filter(pl.col("__AnnoMe_source") == source_name).select(pl.count()).item()
                        filtered_count = self.df_data.filter((pl.col("__AnnoMe_source") == source_name) & pl.col("__AnnoMe_meta_filter")).select(pl.count()).item()
                        no_smiles_count = self.df_data.filter((pl.col("__AnnoMe_source") == source_name) & pl.col("__AnnoMe_has_valid_smiles").is_null()).select(pl.count()).item()
                    else:
                        total = 0
                        filtered_count = 0
                        no_smiles_count = 0

                removed = total - filtered_count

                # Update table cells
                self.file_table.item(row, 1).setText(str(total))
                self.file_table.item(row, 2).setText(str(no_smiles_count))

                smiles_field = file_data.get("smiles_field", "")
                self.file_table.item(row, 3).setText(smiles_field if smiles_field else "-")

                self.file_table.item(row, 4).setText(str(filtered_count))
                self.file_table.item(row, 5).setText(str(removed))

                # Update Meta-Filters
                meta_filters = file_data.get("meta_filters", {})
                filter_text = ", ".join([f"{k}({len(v)})" for k, v in meta_filters.items()]) if meta_filters else "-"
                self.file_table.item(row, 6).setText(filter_text)

                break

    def get_file_path_by_name(self, file_name):
        """Get full file path from file name."""
        for path in self.mgf_files.keys():
            if os.path.basename(path) == file_name:
                return path
        return None

    def on_file_selection_changed(self):
        """Handle file selection changes in the table."""
        selected_rows = set(item.row() for item in self.file_table.selectedItems())

        if not selected_rows:
            return

        # Collect all fields from selected files
        all_fields = set()
        for row in selected_rows:
            file_name = self.file_table.item(row, 0).text()
            file_path = self.get_file_path_by_name(file_name)
            if file_path:
                all_fields.update(self.mgf_files[file_path]["fields"])

        # Update SMILES field combo with common fields
        self.smiles_field_combo.clear()
        self.smiles_field_combo.addItems(sorted(all_fields))

        # If single file selected, set its SMILES field
        if len(selected_rows) == 1:
            row = list(selected_rows)[0]
            file_name = self.file_table.item(row, 0).text()
            file_path = self.get_file_path_by_name(file_name)
            if file_path and self.mgf_files[file_path]["smiles_field"]:
                self.smiles_field_combo.setCurrentText(self.mgf_files[file_path]["smiles_field"])

        # Update meta-field combo
        self.meta_field_combo.clear()
        self.meta_field_combo.addItem("")
        self.meta_field_combo.addItems(sorted(all_fields))

    def apply_smiles_field_to_selected(self):
        """Apply the selected SMILES field to all selected files."""
        selected_rows = set(item.row() for item in self.file_table.selectedItems())
        field_name = self.smiles_field_combo.currentText()

        if not selected_rows:
            QMessageBox.warning(self, "Warning", "Please select one or more files first.")
            return

        if not field_name:
            QMessageBox.warning(self, "Warning", "Please select a SMILES field.")
            return

        applied_count = 0
        files_to_update = []

        for row in selected_rows:
            file_name = self.file_table.item(row, 0).text()
            file_path = self.get_file_path_by_name(file_name)
            if file_path:
                if field_name in self.mgf_files[file_path]["fields"]:
                    self.mgf_files[file_path]["smiles_field"] = field_name
                    files_to_update.append(file_path)
                    applied_count += 1

        if applied_count > 0:
            # Recompute SMILES validity column for updated files
            smiles_validity_columns = []
            for file_path, file_data in self.mgf_files.items():
                smiles_field = file_data.get("smiles_field")
                if smiles_field and smiles_field in self.df_data.columns:
                    source_name = os.path.basename(file_path)
                    validity_expr = pl.when(pl.col("__AnnoMe_source") == source_name).then(pl.when(self._is_valid_smiles_expr(smiles_field)).then(True).otherwise(None)).otherwise(None)
                    smiles_validity_columns.append(validity_expr)

            if smiles_validity_columns:
                has_valid_smiles = smiles_validity_columns[0]
                for expr in smiles_validity_columns[1:]:
                    has_valid_smiles = has_valid_smiles.fill_null(expr)
            else:
                has_valid_smiles = pl.lit(None)

            # Update the validity column
            self.df_data = self.df_data.with_columns([has_valid_smiles.alias("__AnnoMe_has_valid_smiles")])

            # Recompute statistics and update table
            cached_stats = self.compute_all_file_statistics()
            for file_path in files_to_update:
                self.update_file_in_table(file_path, cached_stats=cached_stats)

        QMessageBox.information(self, "Success", f"Applied SMILES field '{field_name}' to {applied_count} file(s)")

    def on_smiles_field_changed(self, field_name):
        """Handle SMILES field selection (only updates combo, doesn't apply)."""
        # This method is kept for compatibility but actual application
        # is now done via the "Apply to Selected" button
        pass

    def on_meta_field_changed(self, field_name):
        """Handle meta-field selection for filtering."""
        selected_rows = set(item.row() for item in self.file_table.selectedItems())
        if not selected_rows:
            return

        # Temporarily block signals to prevent on_meta_values_changed from firing during setup
        self.meta_values_list.blockSignals(True)
        self.meta_values_list.clear()

        if not field_name:
            self.meta_values_list.blockSignals(False)
            return

        # Initialize all_values
        all_values = set()

        # Use DataFrame to collect all unique values from selected files
        if self.df_data is not None and field_name in self.df_data.columns:
            # Get source names for selected files
            selected_sources = []
            for row in selected_rows:
                file_name = self.file_table.item(row, 0).text()
                selected_sources.append(file_name)

            # Get unique values from DataFrame for selected sources
            file_df = self.df_data.filter(pl.col("__AnnoMe_source").is_in(selected_sources))
            unique_values = file_df.select(pl.col(field_name)).unique().to_series().drop_nulls().to_list()
            all_values = set(str(v) for v in unique_values)

        # Add to list
        for value in sorted(all_values):
            self.meta_values_list.addItem(value)

        # Restore previous selections if they exist for any selected file
        # For single file selection, restore that file's selections
        # For multiple files, use intersection of selections (common selected values)
        if len(selected_rows) == 1:
            # Single file: restore its specific selections
            row = list(selected_rows)[0]
            file_name = self.file_table.item(row, 0).text()
            file_path = self.get_file_path_by_name(file_name)

            if file_path and file_path in self.meta_field_selections:
                if field_name in self.meta_field_selections[file_path]:
                    # Restore saved selections
                    saved_selections = self.meta_field_selections[file_path][field_name]
                    for i in range(self.meta_values_list.count()):
                        item = self.meta_values_list.item(i)
                        if item.text() in saved_selections:
                            item.setSelected(True)
                else:
                    # No saved selections for this field, select all by default
                    for i in range(self.meta_values_list.count()):
                        self.meta_values_list.item(i).setSelected(True)
                    # Save this initial "select all" state
                    self._save_current_selections_for_file(file_path, field_name)
            else:
                # No saved selections for this file, select all by default
                for i in range(self.meta_values_list.count()):
                    self.meta_values_list.item(i).setSelected(True)
                # Save this initial "select all" state
                self._save_current_selections_for_file(file_path, field_name)
        else:
            # Multiple files: find common selections or select all if no common history
            common_selections = None
            has_any_saved = False

            for row in selected_rows:
                file_name = self.file_table.item(row, 0).text()
                file_path = self.get_file_path_by_name(file_name)

                if file_path and file_path in self.meta_field_selections:
                    if field_name in self.meta_field_selections[file_path]:
                        has_any_saved = True
                        saved = self.meta_field_selections[file_path][field_name]
                        if common_selections is None:
                            common_selections = saved.copy()
                        else:
                            # Intersection of selections
                            common_selections &= saved

            if has_any_saved and common_selections is not None:
                # Use common selections
                for i in range(self.meta_values_list.count()):
                    item = self.meta_values_list.item(i)
                    if item.text() in common_selections:
                        item.setSelected(True)
            else:
                # No saved selections, select all by default
                for i in range(self.meta_values_list.count()):
                    self.meta_values_list.item(i).setSelected(True)
                # Save this initial "select all" state for all selected files
                for row in selected_rows:
                    file_name = self.file_table.item(row, 0).text()
                    file_path = self.get_file_path_by_name(file_name)
                    if file_path:
                        self._save_current_selections_for_file(file_path, field_name)

        # Re-enable signals and trigger the filter application
        self.meta_values_list.blockSignals(False)
        # Manually trigger the changed handler to apply filters
        self.on_meta_values_changed()

    def _save_current_selections_for_file(self, file_path, field_name):
        """Helper method to save current selections for a specific file and field."""
        selected_values = {item.text() for item in self.meta_values_list.selectedItems()}
        if file_path not in self.meta_field_selections:
            self.meta_field_selections[file_path] = {}
        self.meta_field_selections[file_path][field_name] = selected_values.copy()

    def on_meta_values_changed(self):
        """Handle meta-value selection changes - debounced to avoid redundant recomputes."""
        # Save current selections immediately (cheap)
        selected_rows = set(item.row() for item in self.file_table.selectedItems())
        if not selected_rows:
            return

        field_name = self.meta_field_combo.currentText()
        if not field_name:
            return

        selected_values = {item.text() for item in self.meta_values_list.selectedItems()}

        # Apply filter to all selected files (just save state, cheap)
        for row in selected_rows:
            file_name = self.file_table.item(row, 0).text()
            file_path = self.get_file_path_by_name(file_name)
            if not file_path:
                continue

            file_data = self.mgf_files[file_path]

            # Save the current selection state for this file and field
            if file_path not in self.meta_field_selections:
                self.meta_field_selections[file_path] = {}
            self.meta_field_selections[file_path][field_name] = selected_values.copy()

            # Update meta_filters to match selections (keep for backward compatibility)
            file_data["meta_filters"][field_name] = selected_values

        # Debounce the expensive DataFrame recompute (200ms)
        if self._meta_values_debounce_timer is not None:
            self._meta_values_debounce_timer.stop()
        self._meta_values_debounce_timer = QTimer()
        self._meta_values_debounce_timer.setSingleShot(True)
        self._meta_values_debounce_timer.timeout.connect(self._apply_meta_filter_deferred)
        self._meta_values_debounce_timer.start(200)

    def _apply_meta_filter_deferred(self):
        """Deferred meta filter application (called after debounce timer)."""
        # Update the DataFrame meta filter
        self.update_meta_filter_in_dataframe()

        # Compute all statistics in ONE pass, then update table
        cached_stats = self.compute_all_file_statistics()
        for file_path in self.mgf_files.keys():
            self.update_file_in_table(file_path, cached_stats=cached_stats)

    @staticmethod
    def _canonicalize_smiles_worker(smiles):
        """Worker function to canonicalize a single SMILES string."""
        try:
            mol = rdkit.Chem.MolFromSmiles(smiles)
            if mol:
                canonical_smiles = rdkit.Chem.MolToSmiles(mol, canonical=True)
                return (smiles, canonical_smiles, True)
            else:
                return (smiles, smiles, False)
        except Exception:
            return (smiles, smiles, False)

    def apply_canonicalization(self):
        """Apply SMILES canonicalization to all loaded files using sequential processing and DataFrame."""
        if not self.canon_checkbox.isChecked():
            QMessageBox.warning(self, "Warning", "Please check the canonicalization checkbox first.")
            return

        if not self.mgf_files:
            QMessageBox.warning(self, "Warning", "Please load at least one MGF file first.")
            return

        if self.df_data is None:
            QMessageBox.warning(self, "Warning", "No data available. Please load MGF files first.")
            return

        # Find SMILES field (use the first non-null SMILES field from any file)
        smiles_field = None
        for file_data in self.mgf_files.values():
            if file_data.get("smiles_field"):
                smiles_field = file_data["smiles_field"]
                break

        if not smiles_field or smiles_field not in self.df_data.columns:
            QMessageBox.warning(self, "Warning", "No SMILES field selected. Please select a SMILES field.")
            return

        # Get all unique SMILES from DataFrame
        all_smiles_list = self.df_data.select(pl.col(smiles_field)).unique().drop_nulls().to_series().to_list()

        # Filter out empty SMILES
        all_smiles_list = [s for s in all_smiles_list if self._is_valid_smiles_value(s)]

        if not all_smiles_list:
            QMessageBox.warning(self, "Warning", "No SMILES found in loaded files.")
            return

        print(f"Canonicalizing {len(all_smiles_list)} unique SMILES...")

        progress = QProgressDialog("Canonicalizing SMILES...", "Cancel", 0, len(all_smiles_list), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QApplication.processEvents()

        success_count = 0
        error_count = 0

        try:
            # Process SMILES sequentially
            smiles_map = {}
            for idx, smiles in enumerate(all_smiles_list):
                if idx % 100 == 0:  # Update progress every 100 items
                    progress.setValue(idx)
                    QApplication.processEvents()

                if progress.wasCanceled():
                    break

                original_smiles, canonical_smiles, success = self._canonicalize_smiles_worker(smiles)
                smiles_map[original_smiles] = canonical_smiles
                if success:
                    success_count += 1
                else:
                    error_count += 1

            # Update DataFrame with canonical SMILES using efficient map_elements with dict lookup
            # This is much more efficient than chaining when().then().otherwise() for each mapping

            # Use Polars replace method which is optimized for bulk replacements
            self.df_data = self.df_data.with_columns([pl.col(smiles_field).replace(smiles_map, default=pl.col(smiles_field)).alias(smiles_field)])

            # Mark all files with this SMILES field as canonicalized
            cached_stats = self.compute_all_file_statistics()
            for file_path, file_data in self.mgf_files.items():
                if file_data.get("smiles_field") == smiles_field:
                    file_data["canonicalized"] = True
                    self.update_file_in_table(file_path, cached_stats=cached_stats)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during canonicalization: {str(e)}")
            return

        progress.setValue(len(all_smiles_list))

        status_msg = f"Canonicalized {success_count} SMILES"
        if error_count > 0:
            status_msg += f" ({error_count} errors)"

        self.canon_status_label.setText(f"Status: {status_msg}")

        QMessageBox.information(self, "Success", status_msg)

    def add_smarts_filter(self):
        """Add or update a SMARTS filter."""
        filter_name = self.filter_name_input.text().strip()
        smarts_string = self.smarts_input.text().strip()

        if not filter_name or not smarts_string:
            QMessageBox.warning(self, "Warning", "Please provide both filter name and SMARTS string.")
            return

        # Parse AND-separated groups (case-insensitive)
        and_groups = re.split(r"\s+AND\s+", smarts_string, flags=re.IGNORECASE)
        and_groups = [s.strip() for s in and_groups if s.strip()]

        # Parse each AND group for OR patterns (marked with << >>)
        parsed_smarts = []
        for and_group in and_groups:
            # Check if this group has << >> markers
            if "<<" in and_group and ">>" in and_group:
                # Extract content within << >>
                or_match = re.search(r"<<(.+?)>>", and_group, re.DOTALL)
                if or_match:
                    or_content = or_match.group(1)
                    # Split by OR (case-insensitive)
                    or_patterns = re.split(r"\s+OR\s+", or_content, flags=re.IGNORECASE)
                    or_patterns = [p.strip() for p in or_patterns if p.strip()]
                else:
                    # If markers exist but regex doesn't match, treat as single pattern
                    or_patterns = [and_group.strip()]
            else:
                # No OR patterns, treat as single pattern
                or_patterns = [and_group.strip()]

            parsed_smarts.append(or_patterns)

        print(f"Parsed SMARTS for filter '{filter_name}':")
        pprint(parsed_smarts)

        # Validate all SMARTS patterns
        try:
            for or_group in parsed_smarts:
                for smarts in or_group:
                    mol = rdkit.Chem.MolFromSmarts(smarts)
                    if not mol:
                        raise ValueError(f"Invalid SMARTS: {smarts}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid SMARTS string:\n{str(e)}")
            return

        # Check if updating existing filter
        is_update = filter_name in self.smarts_filters

        self.smarts_filters[filter_name] = parsed_smarts

        # Update or add to export list
        if not is_update:
            item = QListWidgetItem(filter_name)
            self.export_filter_list.addItem(item)
            if self.export_all_checkbox.isChecked():
                item.setSelected(True)

        self.filter_name_input.clear()
        self.smarts_input.clear()

        # Trigger filtering
        self.apply_filter(filter_name)

        action = "Updated" if is_update else "Added"
        QMessageBox.information(self, "Success", f"{action} filter: {filter_name}")

    def add_filter_to_table(self, filter_name):
        """Add a filter to the filter table."""
        row_position = self.filter_table.rowCount()
        self.filter_table.insertRow(row_position)

        # Filter Name
        self.filter_table.setItem(row_position, 0, QTableWidgetItem(filter_name))

        # Statistics will be updated by update_filter_in_table
        self.filter_table.setItem(row_position, 1, QTableWidgetItem("0"))
        self.filter_table.setItem(row_position, 2, QTableWidgetItem("0"))
        self.filter_table.setItem(row_position, 3, QTableWidgetItem("0"))

    def update_filter_in_table(self, filter_name):
        """Update a filter's statistics in the table."""
        # Find the row for this filter
        for row in range(self.filter_table.rowCount()):
            if self.filter_table.item(row, 0).text() == filter_name:
                if filter_name in self.filter_matched_smiles:
                    # Get unique SMILES count from filtered DataFrame
                    filtered_df = self.df_data.filter(pl.col("__AnnoMe_meta_filter"))

                    # Find SMILES field
                    smiles_field = None
                    for file_data in self.mgf_files.values():
                        if file_data.get("smiles_field"):
                            smiles_field = file_data["smiles_field"]
                            break

                    if smiles_field and smiles_field in filtered_df.columns:
                        # Get all valid unique SMILES
                        all_smiles = filtered_df.select(pl.col(smiles_field)).unique().drop_nulls().to_series().to_list()
                        all_smiles = set(s for s in all_smiles if self._is_valid_smiles_value(s))

                        matched = len(self.filter_matched_smiles[filter_name])
                        total = len(all_smiles)
                        non_matched = total - matched

                        self.filter_table.item(row, 1).setText(str(total))
                        self.filter_table.item(row, 2).setText(str(matched))
                        self.filter_table.item(row, 3).setText(str(non_matched))
                return

        # If not found, add new row
        self.add_filter_to_table(filter_name)
        self.update_filter_in_table(filter_name)

    def edit_selected_filter(self):
        """Edit the selected filter."""
        selected_rows = self.filter_table.selectedItems()
        if not selected_rows:
            QMessageBox.warning(self, "Warning", "Please select a filter to edit.")
            return

        row = selected_rows[0].row()
        filter_name = self.filter_table.item(row, 0).text()

        if filter_name in self.smarts_filters:
            self.filter_name_input.setText(filter_name)

            # Reconstruct the SMARTS string from parsed format
            parsed_smarts = self.smarts_filters[filter_name]
            smarts_parts = []

            for or_group in parsed_smarts:
                if len(or_group) > 1:
                    # Has OR patterns
                    or_string = "<<" + " OR ".join(or_group) + ">>"
                    smarts_parts.append(or_string)
                else:
                    # Single pattern
                    smarts_parts.append(or_group[0])

            self.smarts_input.setText(" AND ".join(smarts_parts))

    def delete_selected_filter(self):
        """Delete the selected filter."""
        selected_rows = self.filter_table.selectedItems()
        if not selected_rows:
            QMessageBox.warning(self, "Warning", "Please select a filter to delete.")
            return

        row = selected_rows[0].row()
        filter_name = self.filter_table.item(row, 0).text()

        reply = QMessageBox.question(self, "Confirm Delete", f"Are you sure you want to delete filter '{filter_name}'?", QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            # Remove from data structures
            if filter_name in self.smarts_filters:
                del self.smarts_filters[filter_name]
            if filter_name in self.filter_matched_smiles:
                del self.filter_matched_smiles[filter_name]

            # Remove from UI
            self.filter_table.removeRow(row)

            # Remove from export list
            for i in range(self.export_filter_list.count()):
                if self.export_filter_list.item(i).text() == filter_name:
                    self.export_filter_list.takeItem(i)
                    break

    def delete_all_filters(self):
        """Delete all defined filters after confirmation."""
        if not self.smarts_filters:
            QMessageBox.information(self, "No Filters", "There are no filters to delete.")
            return

        filter_count = len(self.smarts_filters)
        reply = QMessageBox.question(self, "Confirm Delete All", f"Are you sure you want to delete all {filter_count} filter(s)?\n\nThis action cannot be undone.", QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            # Clear all data structures
            self.smarts_filters.clear()
            self.filter_matched_smiles.clear()

            # Clear filter table
            self.filter_table.setRowCount(0)

            # Clear export filter list
            self.export_filter_list.clear()

            # Clear SMARTS visualization
            self.smarts_viz_container.setText("Select a filter to view its SMARTS patterns")
            self.smarts_viz_pixmap = None

            QMessageBox.information(self, "Filters Deleted", f"All {filter_count} filter(s) have been deleted.")

    def on_filter_selection_changed(self):
        """Handle filter selection changes in the table and update SMARTS visualization."""
        selected_rows = self.filter_table.selectedItems()
        if not selected_rows:
            self.smarts_viz_container.setText("Select a filter to view its SMARTS patterns")
            self.smarts_viz_pixmap = None
            return

        row = selected_rows[0].row()
        filter_name = self.filter_table.item(row, 0).text()

        if filter_name in self.smarts_filters:
            self.render_smarts_structures(filter_name)
        else:
            self.smarts_viz_container.setText("No SMARTS patterns available for this filter")
            self.smarts_viz_pixmap = None

    def view_structures(self):
        """Open a new window to view structures for the selected filter."""
        selected_rows = self.filter_table.selectedItems()
        if not selected_rows:
            QMessageBox.warning(self, "Warning", "Please select a filter to view structures.")
            return

        row = selected_rows[0].row()
        filter_name = self.filter_table.item(row, 0).text()

        if filter_name not in self.filter_matched_smiles:
            QMessageBox.warning(self, "Warning", "Filter results not available. Please apply the filter first.")
            return

        matched_smiles = list(self.filter_matched_smiles[filter_name])

        # Compute non-matched SMILES from the filtered DataFrame
        smiles_field = None
        for file_data in self.mgf_files.values():
            if file_data.get("smiles_field"):
                smiles_field = file_data["smiles_field"]
                break

        non_matched_smiles = []
        if smiles_field and self.df_data is not None and smiles_field in self.df_data.columns:
            filtered_df = self.df_data.filter(pl.col("__AnnoMe_meta_filter"))
            all_smiles = filtered_df.select(pl.col(smiles_field)).unique().drop_nulls().to_series().to_list()
            all_valid = set(s for s in all_smiles if self._is_valid_smiles_value(s))
            non_matched_smiles = list(all_valid - self.filter_matched_smiles[filter_name])

        # Create and show structure viewer window
        viewer = StructureViewerWindow(filter_name, matched_smiles, non_matched_smiles, self)
        viewer.show()

        # Keep reference to prevent garbage collection
        self.structure_windows.append(viewer)

    def render_smarts_structures(self, filter_name):
        """Render SMARTS patterns as molecular structure images."""
        if filter_name not in self.smarts_filters:
            return

        parsed_smarts = self.smarts_filters[filter_name]

        try:
            from rdkit.Chem import Draw
            from rdkit import Chem

            # Collect all SMARTS patterns from the filter
            smarts_list = []
            labels = []

            for and_idx, or_group in enumerate(parsed_smarts):
                for or_idx, smarts_pattern in enumerate(or_group):
                    smarts_list.append(smarts_pattern)
                    if len(or_group) > 1:
                        # Multiple patterns in OR group
                        labels.append(f"AND Group {and_idx + 1}, OR {or_idx + 1}")
                    else:
                        # Single pattern in AND group
                        labels.append(f"AND Group {and_idx + 1}")

            if not smarts_list:
                self.smarts_viz_container.setText("No SMARTS patterns found")
                return

            # Convert SMARTS to molecules
            mols = []
            valid_labels = []
            for smarts, label in zip(smarts_list, labels):
                try:
                    mol = Chem.MolFromSmarts(smarts)
                    if mol:
                        mols.append(mol)
                        valid_labels.append(label)
                except Exception as e:
                    print(f"Failed to parse SMARTS '{smarts}': {e}")
                    continue

            if not mols:
                self.smarts_viz_container.setText("Could not parse any SMARTS patterns")
                return

            # Render molecules to image
            mols_per_row = min(3, len(mols))
            mol_size = (400, 400)
            n_rows = math.ceil(len(mols) / mols_per_row)

            img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=mol_size, legends=valid_labels, returnPNG=False)

            # Convert PIL image to QPixmap
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)

            qimage = QImage()
            qimage.loadFromData(img_byte_arr.read())
            pixmap = QPixmap.fromImage(qimage)

            if not pixmap.isNull():
                self.smarts_viz_pixmap = pixmap
                # Scale to fit container width while maintaining aspect ratio
                scaled_pixmap = pixmap.scaledToWidth(self.smarts_viz_scroll.viewport().width() - 20, Qt.SmoothTransformation)
                self.smarts_viz_container.setPixmap(scaled_pixmap)
            else:
                self.smarts_viz_container.setText("Failed to create visualization")

        except Exception as e:
            self.smarts_viz_container.setText(f"Error rendering SMARTS: {str(e)}")
            print(f"Error rendering SMARTS structures: {e}")

    @staticmethod
    def _check_smarts_match_with_compiled(mol, compiled_patterns):
        """Check if an RDKit mol matches pre-compiled SMARTS patterns.

        Args:
            mol: RDKit molecule object
            compiled_patterns: List of lists of compiled RDKit pattern mols.
                              Outer list = AND groups, inner list = OR alternatives.

        Returns:
            bool: True if all AND groups match (at least one OR pattern each).
        """
        # All AND groups must match (top level)
        for or_group in compiled_patterns:
            # At least one pattern in the OR group must match
            or_match = False
            for pattern_mol in or_group:
                if pattern_mol is not None and mol.HasSubstructMatch(pattern_mol):
                    or_match = True
                    break

            # If no pattern in this OR group matched, the whole filter fails
            if not or_match:
                return False

        # All AND groups matched
        return True

    @staticmethod
    def _compile_smarts_patterns(parsed_smarts):
        """Pre-compile SMARTS patterns into RDKit mol objects.

        Args:
            parsed_smarts: List of OR groups, each a list of SMARTS strings.

        Returns:
            List of lists of compiled RDKit mol objects (same structure).
        """
        compiled = []
        for or_group in parsed_smarts:
            compiled_or = []
            for smarts_pattern in or_group:
                try:
                    compiled_or.append(rdkit.Chem.MolFromSmarts(smarts_pattern))
                except Exception:
                    compiled_or.append(None)
            compiled.append(compiled_or)
        return compiled

    @staticmethod
    def _process_smiles_chunk(chunk_info):
        """Worker function to process a chunk of SMILES with all filters sequentially.

        SMARTS patterns are pre-compiled once per chunk to avoid redundant parsing.
        """
        smiles_chunk, all_filters = chunk_info

        # Pre-compile all SMARTS patterns once for this chunk (major perf win)
        compiled_filters = {}
        for filter_name, parsed_smarts in all_filters.items():
            compiled_filters[filter_name] = MGFFilterGUI._compile_smarts_patterns(parsed_smarts)

        # Results for each filter
        filter_results = {}

        # Apply each filter sequentially to this chunk
        for filter_name, compiled_patterns in compiled_filters.items():
            matched_smiles = []
            non_matched_smiles = []

            for smiles in smiles_chunk:
                try:
                    mol = rdkit.Chem.MolFromSmiles(smiles)
                    if mol and MGFFilterGUI._check_smarts_match_with_compiled(mol, compiled_patterns):
                        matched_smiles.append(smiles)
                    else:
                        non_matched_smiles.append(smiles)
                except Exception:
                    non_matched_smiles.append(smiles)

            filter_results[filter_name] = {"matched_smiles": matched_smiles, "non_matched_smiles": non_matched_smiles}

        return filter_results

    def apply_filter(self, filter_name, show_progress=True):
        """Apply a single SMARTS filter across all files sequentially.

        Args:
            filter_name: Name of the filter to apply
            show_progress: Whether to show the progress dialog (default: True)
        """
        if filter_name not in self.smarts_filters:
            return

        if not self.mgf_files:
            QMessageBox.warning(self, "Warning", "Please load at least one MGF file first.")
            return

        if self.df_data is None:
            QMessageBox.warning(self, "Warning", "No data available. Please load MGF files first.")
            return

        parsed_smarts = self.smarts_filters[filter_name]

        # Get all unique SMILES from DataFrame (only from rows that pass meta filter)
        filtered_df = self.df_data.filter(pl.col("__AnnoMe_meta_filter"))

        # Find SMILES field (use the first non-null SMILES field from any file)
        smiles_field = None
        for file_data in self.mgf_files.values():
            if file_data.get("smiles_field"):
                smiles_field = file_data["smiles_field"]
                break

        if not smiles_field or smiles_field not in filtered_df.columns:
            QMessageBox.warning(self, "Warning", "No SMILES field selected. Please select a SMILES field.")
            return

        # Get unique SMILES
        all_smiles_list = filtered_df.select(pl.col(smiles_field)).unique().drop_nulls().to_series().to_list()

        # Filter out empty SMILES
        all_smiles_list = [s for s in all_smiles_list if self._is_valid_smiles_value(s)]

        if not all_smiles_list:
            QMessageBox.warning(self, "Warning", "No valid SMILES found in loaded files.")
            return

        print(f"Filtering {len(all_smiles_list)} SMILES sequentially...")

        progress = None
        if show_progress:
            progress = QProgressDialog("Filtering SMILES...", "Cancel", 0, len(all_smiles_list), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            QApplication.processEvents()

        try:
            # Process all SMILES sequentially with the single filter
            chunk_info = (all_smiles_list, {filter_name: parsed_smarts})
            result = self._process_smiles_chunk(chunk_info)

            matched_smiles = set(result[filter_name]["matched_smiles"])
            non_matched_smiles = set(result[filter_name]["non_matched_smiles"])

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during filtering: {str(e)}")
            return

        if progress:
            progress.setValue(len(all_smiles_list))

        # OPTIMIZED: Store only the set of matching SMILES (not DataFrame columns or blocks)
        # This dramatically reduces memory usage - only stores SMILES strings, not entire blocks
        print(f"Filter '{filter_name}': {len(matched_smiles)} matched, {len(non_matched_smiles)} non-matched")
        self.filter_matched_smiles[filter_name] = matched_smiles  # Already a set

        # Update table
        self.update_filter_in_table(filter_name)

    def apply_all_filters_parallel(self):
        """Apply all defined SMARTS filters sequentially."""
        if not self.smarts_filters:
            QMessageBox.warning(self, "Warning", "No filters defined.")
            return

        if not self.mgf_files:
            QMessageBox.warning(self, "Warning", "Please load at least one MGF file first.")
            return

        if self.df_data is None:
            QMessageBox.warning(self, "Warning", "No data available. Please load MGF files first.")
            return

        # Get SMILES field
        smiles_field = None
        for file_data in self.mgf_files.values():
            if file_data.get("smiles_field"):
                smiles_field = file_data["smiles_field"]
                break

        if not smiles_field or smiles_field not in self.df_data.columns:
            QMessageBox.warning(self, "Warning", "No SMILES field selected. Please select a SMILES field.")
            return

        # Collect all unique SMILES from DataFrame directly (no dict conversion)
        filtered_df = self.df_data.filter(pl.col("__AnnoMe_meta_filter"))
        all_smiles_list = filtered_df.select(pl.col(smiles_field)).unique().drop_nulls().to_series().to_list()

        # Filter out empty SMILES
        all_smiles_list = [s for s in all_smiles_list if self._is_valid_smiles_value(s)]

        if not all_smiles_list:
            QMessageBox.warning(self, "Warning", "No SMILES found in loaded files. Please select a SMILES field.")
            return

        print(f"Processing {len(self.smarts_filters)} filter(s) on {len(all_smiles_list)} SMILES sequentially...")

        progress = QProgressDialog("Processing filters...", "Cancel", 0, len(all_smiles_list), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QApplication.processEvents()

        try:
            # Process all SMILES with all filters in a single sequential pass
            chunk_info = (all_smiles_list, self.smarts_filters)
            result = self._process_smiles_chunk(chunk_info)

            for filter_name in self.smarts_filters.keys():
                # OPTIMIZED: Store only matching SMILES set (not blocks)
                self.filter_matched_smiles[filter_name] = set(result[filter_name]["matched_smiles"])

                # Update table for this filter
                self.update_filter_in_table(filter_name)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during filtering: {str(e)}")
            return

        progress.setValue(len(all_smiles_list))
        QMessageBox.information(self, "Success", f"Successfully processed {len(self.smarts_filters)} filter(s).")

    def on_export_all_changed(self, state):
        """Handle export all checkbox state change."""
        if self.export_all_checkbox.isChecked():
            # Select all filters
            for i in range(self.export_filter_list.count()):
                self.export_filter_list.item(i).setSelected(True)
        else:
            # Deselect all filters
            self.export_filter_list.clearSelection()

    def browse_output_file(self):
        """Browse for output file."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Base Output File", "", "MGF Files (*.mgf);;All Files (*)")
        if file_path:
            self.output_file_input.setText(file_path)

    def _export_overview_excel(self, filtered_df, smiles_field, selected_filters, base_dir, base_name):
        """
        Export an overview Excel file containing unique SMILES with images, source files,
        and substructure match information for each filter.

        Args:
            filtered_df: Polars DataFrame containing filtered spectra
            smiles_field: Name of the SMILES field column
            selected_filters: List of filter names to include in overview
            base_dir: Directory where Excel file should be saved
            base_name: Base name for the Excel file
        """
        try:
            print(f"\n{Fore.CYAN}Generating overview Excel file...{Style.RESET_ALL}")

            # Create temporary directory for molecule images
            temp_dir = tempfile.TemporaryDirectory()

            # Get unique SMILES and group spectra by SMILES
            unique_smiles_df = filtered_df.select([smiles_field, "__AnnoMe_source"]).filter(pl.col(smiles_field).is_not_null()).unique()

            # Group by SMILES to get statistics
            smiles_stats = filtered_df.group_by(smiles_field).agg(
                [
                    pl.count().alias("spectrum_count"),
                    pl.col("__AnnoMe_source").unique().alias("source_files"),
                    # Collect common fields (first non-null value for each SMILES)
                    *[
                        pl.col(col).drop_nulls().first().alias(col)
                        for col in filtered_df.columns
                        if col not in [smiles_field, "$$spectrumdata", "$$spectrumData", "peaks"] and not col.startswith("__AnnoMe_")
                    ],
                ]
            )

            # Sort by SMILES for consistent output
            smiles_stats = smiles_stats.sort(smiles_field)

            # Convert to list of dictionaries for easier processing
            smiles_list = smiles_stats.to_dicts()

            print(f"  Processing {len(smiles_list)} unique SMILES...")

            # Disable image generation if there are too many SMILES (performance optimization)
            include_images = len(smiles_list) <= 10000
            if not include_images:
                print(f"  {Fore.YELLOW}Warning: More than 10,000 unique SMILES found. Image generation disabled for performance.{Style.RESET_ALL}")

            # Create table data structure
            table_data = []

            for idx, smiles_row in enumerate(smiles_list):
                smiles_code = smiles_row[smiles_field]

                if not smiles_code or not self._is_valid_smiles_value(smiles_code):
                    continue

                # Create row data dictionary
                row_data = OrderedDict()

                # Add SMILES code
                row_data["A_SMILES"] = smiles_code

                # Add structure image (only if enabled)
                if include_images:
                    try:
                        img = Filters.draw_smiles([smiles_code], max_draw=1)
                        img_path = os.path.join(temp_dir.name, f"img_{idx}.png")
                        with open(img_path, "wb") as f:
                            f.write(img.data)
                        row_data["B_Structure"] = f"$$$IMG:{img_path}"
                    except Exception as e:
                        print(f"  Warning: Could not draw structure for SMILES {smiles_code}: {e}")
                        row_data["B_Structure"] = "ERROR: could not draw structure"

                # Add spectrum count
                row_data["C_SpectrumCount"] = smiles_row["spectrum_count"]

                # Add source MGF files (comma-separated list)
                source_files = smiles_row.get("source_files", [])
                if isinstance(source_files, list):
                    row_data["D_SourceFiles"] = ", ".join(sorted(set(source_files)))
                else:
                    row_data["D_SourceFiles"] = str(source_files)

                # Add common fields (Name, Formula, CAS, etc.)
                for col_name in ["name", "NAME", "Name", "compound", "COMPOUND", "Compound"]:
                    if col_name in smiles_row and smiles_row[col_name]:
                        row_data["E_Name"] = smiles_row[col_name]
                        break

                for col_name in ["formula", "FORMULA", "Formula", "sumformula", "SUMFORMULA", "SumFormula"]:
                    if col_name in smiles_row and smiles_row[col_name]:
                        row_data["F_Formula"] = smiles_row[col_name]
                        break

                for col_name in ["cas", "CAS", "Cas"]:
                    if col_name in smiles_row and smiles_row[col_name]:
                        row_data["G_CAS"] = smiles_row[col_name]
                        break

                # Check each filter and add match status with spectrum count
                for filter_name in selected_filters:
                    if filter_name not in self.filter_matched_smiles:
                        continue

                    matched_smiles_set = self.filter_matched_smiles[filter_name]

                    if smiles_code in matched_smiles_set:
                        # Count how many spectra with this SMILES match the filter
                        match_count_df = filtered_df.filter((pl.col(smiles_field) == smiles_code))
                        match_count = len(match_count_df)

                        row_data[f"H_{filter_name}"] = f"substructure match ({match_count})"
                    else:
                        row_data[f"H_{filter_name}"] = ""

                table_data.append(row_data)

            # Write to Excel file
            excel_path = os.path.join(base_dir, f"{base_name}_overview.xlsx")
            
            # Try to export as Excel, fall back to TSV if memory issues
            try:
                Filters.list_to_excel_table(table_data, excel_path, sheet_name="Overview", img_prefix="$$$IMG:", column_width=40, row_height=40 if include_images else 10)
                
                # Cleanup temporary directory
                temp_dir.cleanup()
                
                print(f"  {Fore.GREEN}Overview Excel file exported to: {excel_path}{Style.RESET_ALL}")
                
                return excel_path
                
            except Exception as excel_error:
                print(f"  {Fore.YELLOW}Warning: Could not export Excel file: {str(excel_error)}{Style.RESET_ALL}")
                print(f"  {Fore.CYAN}Exporting as TSV file instead...{Style.RESET_ALL}")
                
                # Cleanup temporary directory since we won't use images
                temp_dir.cleanup()
                
                # Export as TSV instead
                tsv_path = os.path.join(base_dir, f"{base_name}_overview.tsv")
                
                try:
                    
                    with open(tsv_path, 'w', newline='', encoding='utf-8') as f:
                        if table_data:
                            # Get all column headers from first row (and ensure all rows have same columns)
                            headers = list(table_data[0].keys())
                            
                            writer = csv.DictWriter(f, fieldnames=headers, delimiter='\t')
                            writer.writeheader()
                            
                            # Write data rows, removing image paths
                            for row in table_data:
                                clean_row = {}
                                for key, value in row.items():
                                    # Remove image path references
                                    if isinstance(value, str) and value.startswith("$$$IMG:"):
                                        clean_row[key] = "(see SMILES)"
                                    else:
                                        clean_row[key] = value
                                writer.writerow(clean_row)
                    
                    print(f"  {Fore.GREEN}Overview TSV file exported to: {tsv_path}{Style.RESET_ALL}")
                    
                    return tsv_path
                    
                except Exception as tsv_error:
                    print(f"  {Fore.RED}Error exporting TSV file: {str(tsv_error)}{Style.RESET_ALL}")
                    return None

        except Exception as e:
            print(f"  {Fore.RED}Error generating overview Excel: {str(e)}{Style.RESET_ALL}")

            traceback.print_exc()
            return None

    def _export_single_filter_optimized(self, export_info):
        """Worker function to export a single filter - uses Polars DataFrame directly.

        OPTIMIZED: Instead of converting to dictionaries, we:
        1. Filter the DataFrame to rows that match the filter's SMILES
        2. Remove internal tracking columns
        3. Pass the filtered DataFrame directly to export_mgf_file_from_polars_table

        This is much faster as it avoids row-by-row iteration and dictionary conversion.
        """
        filter_name, matched_smiles_set, filtered_df, smiles_field, base_dir, base_name = export_info

        try:
            # Filter DataFrame to only matched SMILES using Polars operations (much faster)
            matched_df = filtered_df.filter(pl.col(smiles_field).is_in(matched_smiles_set))

            # Remove internal tracking columns
            data_cols = [col for col in matched_df.columns if not col.startswith("__AnnoMe_")]
            matched_df = matched_df.select(data_cols)

            # Export matched file with filter name as suffix
            # Sanitize filter name for filename
            safe_filter_name = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in filter_name)

            matched_path = os.path.join(base_dir, f"{base_name}_{safe_filter_name}_matched.mgf")

            # Only export if there are spectra to write
            if len(matched_df) > 0:
                # Use Polars-native export function (no dictionary conversion needed)
                Filters.export_mgf_file_from_polars_table(matched_df, matched_path)
            else:
                # Delete file if it exists but would be empty
                if os.path.exists(matched_path):
                    os.remove(matched_path)

            return {"success": True, "filter_name": filter_name, "matched_count": len(matched_df)}

        except Exception as e:
            return {"success": False, "filter_name": filter_name, "error": str(e)}

    def export_results(self):
        """Export filtered results to MGF files."""
        base_output_path = self.output_file_input.text()

        if not base_output_path:
            QMessageBox.warning(self, "Warning", "Please select a base output file.")
            return

        # Get selected filters
        selected_filters = [item.text() for item in self.export_filter_list.selectedItems()]

        if not selected_filters:
            QMessageBox.warning(self, "Warning", "Please select at least one filter to export.")
            return

        # Parse base path
        base_dir = os.path.dirname(base_output_path)
        base_name = os.path.splitext(os.path.basename(base_output_path))[0]

        # Create output directory if it doesn't exist
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)

        # Get SMILES field
        smiles_field = None
        for file_data in self.mgf_files.values():
            if file_data.get("smiles_field"):
                smiles_field = file_data["smiles_field"]
                break

        if not smiles_field or smiles_field not in self.df_data.columns:
            QMessageBox.warning(self, "Warning", "No SMILES field selected. Please select a SMILES field.")
            return

        # Get filtered DataFrame (rows that pass meta filter)
        filtered_df = self.df_data.filter(pl.col("__AnnoMe_meta_filter"))

        # Prepare export info for each filter
        export_infos = []
        skipped_filters = []

        for filter_name in selected_filters:
            if filter_name not in self.filter_matched_smiles:
                skipped_filters.append(filter_name)
                continue

            # Pass only the matched SMILES set and DataFrame reference
            matched_smiles_set = self.filter_matched_smiles[filter_name]
            export_infos.append((filter_name, matched_smiles_set, filtered_df, smiles_field, base_dir, base_name))

        if not export_infos:
            QMessageBox.warning(self, "Warning", "No valid filters to export.")
            return

        print(f"Exporting {len(export_infos)} filter(s) sequentially (on-demand reconstruction)...")

        progress = QProgressDialog("Exporting filtered results...", "Cancel", 0, len(export_infos) + 2, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QApplication.processEvents()

        exported_files = []
        all_matched_smiles = set()  # Collect all SMILES that matched any filter
        filters_with_no_matches = 0  # Track filters with zero matching spectra

        try:
            # Export sequentially - each export reconstructs blocks on-demand, keeping memory usage low
            # Each export reconstructs blocks on-demand, keeping memory usage low
            results = []
            for idx, export_info in enumerate(export_infos):
                if progress.wasCanceled():
                    break

                progress.setValue(idx)
                progress.setLabelText(f"Exporting {export_info[0]}...")
                QApplication.processEvents()

                result = self._export_single_filter_optimized(export_info)
                results.append(result)

                # Collect matched SMILES from this filter
                if result["success"]:
                    all_matched_smiles.update(export_info[1])  # export_info[1] is matched_smiles_set

            # Process results
            for idx, result in enumerate(results):
                if progress.wasCanceled():
                    break

                progress.setValue(idx + 1)
                QApplication.processEvents()

                if result["success"]:
                    exported_files.append(result["filter_name"])
                    if result["matched_count"] == 0:
                        filters_with_no_matches += 1
                else:
                    skipped_filters.append(f"{result['filter_name']} (Error: {result['error']})")

            # Export combined all-match file containing all spectra that match AT LEAST ONE filter (no duplicates)
            all_match_count = 0
            if not progress.wasCanceled() and all_matched_smiles:
                progress.setValue(len(export_infos))
                progress.setLabelText("Exporting all-match file...")
                QApplication.processEvents()

                # Use Polars filtering for much better performance
                all_match_df = filtered_df.filter(pl.col(smiles_field).is_in(all_matched_smiles))

                # Remove internal tracking columns
                data_cols = [col for col in all_match_df.columns if not col.startswith("__AnnoMe_")]
                all_match_df = all_match_df.select(data_cols)

                all_match_path = os.path.join(base_dir, f"{base_name}_allMatch.mgf")

                # Only export if there are spectra to write
                if len(all_match_df) > 0:
                    all_match_count = len(all_match_df)
                    Filters.export_mgf_file_from_polars_table(all_match_df, all_match_path)
                else:
                    # Delete file if it exists but would be empty
                    if os.path.exists(all_match_path):
                        os.remove(all_match_path)

            # Export single no-match file containing all spectra that don't match ANY filter
            no_match_count = 0
            if not progress.wasCanceled():
                progress.setValue(len(export_infos) + 1)
                progress.setLabelText("Exporting no-match file...")
                QApplication.processEvents()

                # Use Polars filtering for much better performance
                no_match_df = filtered_df.filter(~pl.col(smiles_field).is_in(all_matched_smiles) | pl.col(smiles_field).is_null())

                # Remove internal tracking columns
                data_cols = [col for col in no_match_df.columns if not col.startswith("__AnnoMe_")]
                no_match_df = no_match_df.select(data_cols)

                no_match_path = os.path.join(base_dir, f"{base_name}_noMatch.mgf")

                # Only export if there are spectra to write
                if len(no_match_df) > 0:
                    no_match_count = len(no_match_df)
                    Filters.export_mgf_file_from_polars_table(no_match_df, no_match_path)
                else:
                    # Delete file if it exists but would be empty
                    if os.path.exists(no_match_path):
                        os.remove(no_match_path)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during export: {str(e)}")
            return

        progress.setValue(len(export_infos) + 2)

        # Export overview Excel file if requested
        excel_path = None
        if self.export_excel_checkbox.isChecked():
            try:
                excel_path = self._export_overview_excel(filtered_df, smiles_field, selected_filters, base_dir, base_name)
            except Exception as e:
                print(f"{Fore.RED}Error generating overview Excel: {str(e)}{Style.RESET_ALL}")
                import traceback

                traceback.print_exc()

        # Build concise summary message
        n_filters = len(exported_files)
        k_no_matches = filters_with_no_matches
        u_matched = all_match_count
        i_no_match = no_match_count

        msg = f"Export completed successfully!\n\n"
        msg += f"  • {n_filters} filter(s) written\n"
        if k_no_matches > 0:
            msg += f"  • {k_no_matches} filter(s) with no matching spectra\n"
        msg += f"  • {u_matched} spectra matched to 1 or more filters (all-match file)\n"
        msg += f"  • {i_no_match} spectra did not match any filter (no-match file)\n"
        if excel_path:
            msg += f"  • Overview Excel file created\n"
        msg += f"\nFiles saved to: {base_dir}"

        if skipped_filters:
            msg += f"\n\nWarning: {len(skipped_filters)} filter(s) skipped due to errors:\n"
            for skipped in skipped_filters:
                msg += f"  • {skipped}\n"
            QMessageBox.warning(self, "Export Complete with Warnings", msg)
        else:
            QMessageBox.information(self, "Export Complete", msg)

    def closeEvent(self, event):
        """Clean up temporary directory on close."""
        self.temp_dir.cleanup()

        # Close all structure viewer windows
        for window in self.structure_windows:
            window.close()

        event.accept()

    def load_filters_from_json(self):
        """Load multiple filters from a JSON file and apply them automatically."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select JSON Filter File", "", "JSON Files (*.json);;All Files (*)")

        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                filters_dict = json.load(f)

            if not isinstance(filters_dict, dict):
                QMessageBox.critical(self, "Error", "Invalid JSON format. Expected a dictionary with filter names as keys and SMARTS patterns as values.")
                return

            # Load and apply each filter
            loaded_count = 0
            error_count = 0
            errors = []

            progress = QProgressDialog("Loading filters from JSON...", "Cancel", 0, len(filters_dict), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)

            for idx, (filter_name, smarts_string) in enumerate(filters_dict.items()):
                if progress.wasCanceled():
                    break

                progress.setValue(idx)
                progress.setLabelText(f"Loading filter: {filter_name}")
                QApplication.processEvents()

                try:
                    # Validate filter name and SMARTS string
                    if not filter_name or not isinstance(filter_name, str):
                        raise ValueError("Invalid filter name")

                    if not smarts_string or not isinstance(smarts_string, str):
                        raise ValueError("Invalid SMARTS pattern")

                    # Parse AND-separated groups (case-insensitive)
                    and_groups = re.split(r"\s+AND\s+", smarts_string, flags=re.IGNORECASE)
                    and_groups = [s.strip() for s in and_groups if s.strip()]

                    # Parse each AND group for OR patterns (marked with << >>)
                    parsed_smarts = []
                    for and_group in and_groups:
                        # Check if this group has << >> markers
                        if "<<" in and_group and ">>" in and_group:
                            # Extract content within << >>
                            or_match = re.search(r"<<(.+?)>>", and_group, re.DOTALL)
                            if or_match:
                                or_content = or_match.group(1)
                                # Split by OR (case-insensitive)
                                or_patterns = re.split(r"\s+OR\s+", or_content, flags=re.IGNORECASE)
                                or_patterns = [p.strip() for p in or_patterns if p.strip()]
                            else:
                                # If markers exist but regex doesn't match, treat as single pattern
                                or_patterns = [and_group.strip()]
                        else:
                            # No OR patterns, treat as single pattern
                            or_patterns = [and_group.strip()]

                        parsed_smarts.append(or_patterns)

                    # Validate all SMARTS patterns
                    for or_group in parsed_smarts:
                        for smarts in or_group:
                            mol = rdkit.Chem.MolFromSmarts(smarts)
                            if not mol:
                                raise ValueError(f"Invalid SMARTS: {smarts}")

                    # Check if updating existing filter
                    is_update = filter_name in self.smarts_filters

                    # Store the filter
                    self.smarts_filters[filter_name] = parsed_smarts

                    # Update or add to export list
                    if not is_update:
                        item = QListWidgetItem(filter_name)
                        self.export_filter_list.addItem(item)
                        if self.export_all_checkbox.isChecked():
                            item.setSelected(True)

                    # Apply the filter (without showing progress dialog)
                    self.apply_filter(filter_name, show_progress=False)

                    loaded_count += 1

                except Exception as e:
                    error_count += 1
                    errors.append(f"{filter_name}: {str(e)}")
                    print(f"Error loading filter '{filter_name}': {e}")

            progress.setValue(len(filters_dict))

            # Show results
            msg = f"Successfully loaded and applied {loaded_count} filter(s)"
            if error_count > 0:
                msg += f"\n\n{error_count} filter(s) failed to load:"
                for error in errors[:10]:  # Show first 10 errors
                    msg += f"\n  - {error}"
                if len(errors) > 10:
                    msg += f"\n  ... and {len(errors) - 10} more"

            if error_count > 0:
                QMessageBox.warning(self, "Load Complete with Errors", msg)
            else:
                QMessageBox.information(self, "Load Complete", msg)

        except json.JSONDecodeError as e:
            QMessageBox.critical(self, "Error", f"Failed to parse JSON file:\n{str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load filters from JSON:\n{str(e)}")

    def save_filters_to_json(self):
        """Save all defined filters to a JSON file."""
        if not self.smarts_filters:
            QMessageBox.warning(self, "Warning", "No filters defined to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Filters to JSON", "", "JSON Files (*.json);;All Files (*)")

        if not file_path:
            return

        try:
            # Reconstruct SMARTS strings from parsed format
            filters_dict = {}

            for filter_name, parsed_smarts in self.smarts_filters.items():
                smarts_parts = []

                for or_group in parsed_smarts:
                    if len(or_group) > 1:
                        # Has OR patterns
                        or_string = "<<" + " OR ".join(or_group) + ">>"
                        smarts_parts.append(or_string)
                    else:
                        # Single pattern
                        smarts_parts.append(or_group[0])

                filters_dict[filter_name] = " AND ".join(smarts_parts)

            # Save to JSON file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(filters_dict, f, indent=2, ensure_ascii=False)

            QMessageBox.information(self, "Success", f"Successfully saved {len(filters_dict)} filter(s) to:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save filters:\n{str(e)}")


def main():
    app = QApplication(sys.argv)
    window = MGFFilterGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
