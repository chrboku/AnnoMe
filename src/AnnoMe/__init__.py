"""
AnnoMe - A package for MSMS spectra filtering and classification.
"""

# Make GUI classes available at package level
try:
    from .FilterGUI import FilterGUI
except ImportError:
    pass

try:
    from .ClassificationGUI import ClassificationGUI
except ImportError:
    pass

__version__ = "0.1.0"
__all__ = ["FilterGUI", "ClassificationGUI"]
