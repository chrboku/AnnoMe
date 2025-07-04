# Standard library imports
import ast
import json
import logging
import os
import pathlib
import random
import time
import warnings
from contextlib import contextmanager
import itertools
import tempfile

# Data science packages
import numpy as np
import pandas as pd

# Plotting packages
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotnine as p9

# ML / Bioinformatics packages
from matchms.Pipeline import Pipeline, create_workflow
from matchms.filtering.default_pipelines import DEFAULT_FILTERS
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model

import umap
import pacmap

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

# Other utilities
from natsort import natsorted

from colorama import Fore, Style

# Set random seeds
np.random.seed(42)
random.seed(42)

## Set device for torch
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Using device for torch: {device}")

# Configure plotting and display
# bokeh.io.output_notebook()


def set_random_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)


@contextmanager
def execution_timer(title=None):
    start_time = time.time()
    if title:
        title = f"[{title}]: "
    else:
        title = ""
    try:
        yield
    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        print(f"{title} Failed: {e}")
        print(f"{title} Failed: Total execution time: {Fore.YELLOW}{total_time:.2f}{Style.RESET_ALL} seconds, finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        raise e
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        print(f"{title} Finished: Total execution time: {Fore.YELLOW}{total_time:.2f}{Style.RESET_ALL} seconds, finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)


def p9theme():
    return p9.theme_minimal(base_size=6) + p9.theme(
        axis_ticks=p9.element_blank(),
        title=p9.element_text(color="#3C3C3C"),
        panel_background=p9.element_rect(fill="#FFFFFF"),
        panel_grid_major=p9.element_line(color="#D5D5D5"),
        panel_grid_minor=p9.element_blank(),
        plot_background=p9.element_rect(fill="#FFFFFF", color="#FFFFFF", size=1),
        strip_background=p9.element_rect(size=0, fill="lightgrey"),
    )


# adapted from plotnine package
def save_as_pdf_pages(
    plots,
    filename=None,
    path=None,
    verbose=True,
    **kwargs,
):

    # as in ggplot.save()
    fig_kwargs = {"bbox_inches": "tight"}
    fig_kwargs.update(kwargs)

    # If plots is already an iterator, this is a no-op; otherwise
    # convert a list, etc. to an iterator
    plots = iter(plots)

    # filename, depends on the object
    if filename is None:
        # Take the first element from the iterator, store it, and
        # use it to generate a file name
        peek = [next(plots)]
        plots = itertools.chain(peek, plots)
        filename = peek[0]._save_filename("pdf")

    if path:
        filename = pathlib.Path(path) / filename

    if verbose:
        warnings.warn(f"Filename: {filename}", p9.exceptions.PlotnineWarning)

    with PdfPages(filename, keep_empty=False) as pdf:
        # Re-add the first element to the iterator, if it was removed
        for plot in plots:
            if isinstance(plot, p9.ggplot):
                fig = plot.draw()
                with p9._utils.context.plot_context(plot).rc_context:
                    # Save as a page in the PDF file
                    pdf.savefig(fig, **fig_kwargs)
            elif isinstance(plot, plt.Figure) or isinstance(plot, matplotlib.table.Table):
                pdf.savefig(plot)
            else:
                raise TypeError(f"Unsupported type {type(plot)}. Must be ggplot or Figure.")


def process_ce_value(value):
    """
    Processes the CE value according to specified rules:
    - None or empty string remains as is.
    - Numeric values are formatted to string with one decimal place.
    - String representation of a list of numbers is parsed, sorted numerically,
        and formatted back to a string with one decimal place per number.
    - Other strings are attempted to be converted to float and formatted,
        otherwise returned as is.
    - Other types are converted to string.
    """
    if value is None or value == "":
        return value

    if isinstance(value, (int, float, np.number)):
        try:
            # Format numeric value to string with one decimal place
            return f"{float(value):.1f}"
        except ValueError:
            return str(value)  # Fallback if conversion fails

    if isinstance(value, str):
        # Check if it looks like a list string '[...]'
        if value.startswith("[") and value.endswith("]"):
            try:
                # Safely parse the string representation of a list
                parsed_list = ast.literal_eval(value)
                if isinstance(parsed_list, list):
                    # Convert all items to float, sort numerically
                    float_list = []
                    for item in parsed_list:
                        try:
                            float_list.append(float(item))
                        except (ValueError, TypeError):
                            # If any item is not numeric, return original string
                            return value
                    float_list.sort()
                    if len(float_list) == 1:
                        return f"{float_list[0]:.1f}"
                    # Format sorted list back to string representation
                    return "[" + ",".join(f"{f:.1f}" for f in float_list) + "]"
                else:
                    # If ast.literal_eval doesn't return a list, return original string
                    return value
            except (ValueError, SyntaxError):
                # If parsing fails, proceed to check if it's a simple numeric string
                pass

        # If it's not a list string or parsing failed, try converting the string to float
        try:
            float_val = float(value)
            return f"{float_val:.1f}"
        except ValueError:
            # If it cannot be converted to float, return the original string
            return value

    # Fallback for any other types
    return str(value)


# a class used to parse chemical formulas
# e.g. the formula C6H12O6 will be parsed to a dictionary {'H':12, 'C':6, 'O':6}
# NOTE: different isotopes may be specified as [13C]C5H12O6
# fmt: off
class formulaTools:
    def __init__(self, elemDetails=None):
        if elemDetails is None:
            self.elemDetails = {}
            #              Element      Name        short Neutrons Mass    Abundance
            self.elemDetails["Al"]   = ["Aluminum", "Al", 27, 26.981541, 1.00]
            self.elemDetails["Sb"]   = ["Antimony", "Sb", 121, 120.903824, 0.573]
            self.elemDetails["Ar"]   = ["Argon", "Ar", 40, 39.962383, 0.996]
            self.elemDetails["As"]   = ["Arsenic", "As", 75, 74.921596, 1.00]
            self.elemDetails["Ba"]   = ["Barium", "Ba", 138, 137.905236, 0.717]
            self.elemDetails["Be"]   = ["Beryllium", "Be", 9, 9.012183, 1.00]
            self.elemDetails["Bi"]   = ["Bismuth", "Bi", 209, 208.980388, 1.00]
            self.elemDetails["B"]    = ["Boron", "B", 11, 11.009305, 0.802]
            self.elemDetails["Br"]   = ["Bromine", "Br", 79, 78.918336, 0.5069]
            self.elemDetails["Cd"]   = ["Cadmium", "Cd", 114, 113.903361, 0.2873]
            self.elemDetails["Ca"]   = ["Calcium", "Ca", 40, 39.962591, 0.9695]
            self.elemDetails["44Ca"] = ["Calcium","Ca",44,43.955485,0.0208,]  # 3.992894
            self.elemDetails["C"]    = ["Carbon", "C", 12, 12.0, 0.9893]
            self.elemDetails["12C"]  = ["Carbon", "C", 12, 12.0, 0.9893]
            self.elemDetails["13C"]  = ["Carbon","C",13,13.00335483507,0.0107,]  # 1.00335
            self.elemDetails["Ce"]   = ["Cerium", "Ce", 140, 139.905442, 0.8848]
            self.elemDetails["Cs"]   = ["Cesium", "Cs", 133, 132.905433, 1.00]
            self.elemDetails["Cl"]   = ["Chlorine", "Cl", 35, 34.968853, 0.7577]
            self.elemDetails["35Cl"] = ["Chlorine", "Cl", 35, 34.968853, 0.7577]
            self.elemDetails["37Cl"] = ["Chlorine","Cl",37,36.965903,0.2423,]  # 1.997077
            self.elemDetails["Cr"]   = ["Chromium", "Cr", 52, 51.94051, 0.8379]
            self.elemDetails["50Cr"] = ["Chromium","Cr",50,49.946046,0.0435,]  # -1.994464
            self.elemDetails["53Cr"] = ["Chromium","Cr",53,52.940651,0.095,]  # 1.000141
            self.elemDetails["54Cr"] = ["Chromium","Cr",54,53.938882,0.0236,]  # 1.998372
            self.elemDetails["Co"]   = ["Cobalt", "Co", 59, 58.933198, 1.00]
            self.elemDetails["Cu"]   = ["Copper", "Cu", 63, 62.929599, 0.6917]
            self.elemDetails["65Cu"] = ["Copper","Cu",65,64.927792,0.3083,]  # 1.998193
            self.elemDetails["Dy"]   = ["Dysprosium", "Dy", 164, 163.929183, 0.282]
            self.elemDetails["Er"]   = ["Erbium", "Er", 166, 165.930305, 0.336]
            self.elemDetails["Eu"]   = ["Europium", "Eu", 153, 152.921243, 0.522]
            self.elemDetails["F"]    = ["Fluorine", "F", 19, 18.998403, 1.00]
            self.elemDetails["Gd"]   = ["Gadolinium", "Gd", 158, 157.924111, 0.2484]
            self.elemDetails["Ga"]   = ["Gallium", "Ga", 69, 68.925581, 0.601]
            self.elemDetails["Ge"]   = ["Germanium", "Ge", 74, 73.921179, 0.365]
            self.elemDetails["Au"]   = ["Gold", "Au", 197, 196.96656, 1.00]
            self.elemDetails["Hf"]   = ["Hafnium", "Hf", 180, 179.946561, 0.352]
            self.elemDetails["He"]   = ["Helium", "He", 4, 4.002603, 1.00]
            self.elemDetails["Ho"]   = ["Holmium", "Ho", 165, 164.930332, 1.00]
            self.elemDetails["H"]    = ["Hydrogen", "H", 1, 1.007825, 0.999]
            self.elemDetails["1H"]   = ["Hydrogen", "H", 1, 1.007825, 0.999]
            self.elemDetails["D"]    = ["Hydrogen","H",2,2.01410177812,0.001,]  
            self.elemDetails["2H"]   = ["Hydrogen","H",2,2.01410177812,0.001,]  
            self.elemDetails["In"]   = ["Indium", "In", 115, 114.903875, 0.957]
            self.elemDetails["I"]    = ["Iodine", "I", 127, 126.904477, 1.00]
            self.elemDetails["Ir"]   = ["Iridium", "Ir", 193, 192.962942, 0.627]
            self.elemDetails["Fe"]   = ["Iron", "Fe", 56, 55.934939, 0.9172]
            self.elemDetails["56Fe"] = ["Iron", "Fe", 56, 55.934939, 0.9172]
            self.elemDetails["54Fe"] = ["Iron", "Fe", 54, 53.939612, 0.058]  # -1.995327
            self.elemDetails["57Fe"] = ["Iron", "Fe", 57, 56.935396, 0.022]  # 1.000457
            self.elemDetails["Kr"]   = ["Krypton", "Kr", 84, 83.911506, 0.57]
            self.elemDetails["La"]   = ["Lanthanum", "La", 139, 138.906355, 0.9991]
            self.elemDetails["Pb"]   = ["Lead", "Pb", 208, 207.976641, 0.524]
            self.elemDetails["Li"]   = ["Lithium", "Li", 7, 7.016005, 0.9258]
            self.elemDetails["Lu"]   = ["Lutetium", "Lu", 175, 174.940785, 0.974]
            self.elemDetails["Mg"]   = ["Magnesium", "Mg", 24, 23.985045, 0.789]
            self.elemDetails["25Mg"] = ["Magnesium","Mg",25,24.985839,0.10,]  # 1.000794
            self.elemDetails["26Mg"] = ["Magnesium","Mg",26,25.982595,0.111,]  # 1.99755
            self.elemDetails["Mn"]   = ["Manganese", "Mn", 55, 54.938046, 1.00]
            self.elemDetails["Hg"]   = ["Mercury", "Hg", 202, 201.970632, 0.2965]
            self.elemDetails["Mo"]   = ["Molybdenum", "Mo", 98, 97.905405, 0.2413]
            self.elemDetails["Nd"]   = ["Neodymium", "Nd", 142, 141.907731, 0.2713]
            self.elemDetails["Ne"]   = ["Neon", "Ne", 20, 19.992439, 0.906]
            self.elemDetails["Ni"]   = ["Nickel", "Ni", 58, 57.935347, 0.6827]
            self.elemDetails["Nb"]   = ["Niobium", "Nb", 93, 92.906378, 1.00]
            self.elemDetails["N"]    = ["Nitrogen", "N", 14, 14.003074, 0.9963]
            self.elemDetails["14N"]  = ["Nitrogen", "N", 14, 14.003074, 0.9963]
            self.elemDetails["15N"]  = ["Nitrogen", "N", 15, 15.0001088982, 0.00364]
            self.elemDetails["Os"]   = ["Osmium", "Os", 192, 191.961487, 0.41]
            self.elemDetails["O"]    = ["Oxygen", "O", 16, 15.994915, 0.9976]
            self.elemDetails["Pd"]   = ["Palladium", "Pd", 106, 105.903475, 0.2733]
            self.elemDetails["P"]    = ["Phosphorus", "P", 31, 30.973763, 1.00]
            self.elemDetails["Pt"]   = ["Platinum", "Pt", 195, 194.964785, 0.338]
            self.elemDetails["K"]    = ["Potassium", "K", 39, 38.963708, 0.932]
            self.elemDetails["41K"]  = ["Potassium","K",41,40.961825,0.0673,]  # 1.998117
            self.elemDetails["Pr"]   = ["Praseodymium", "Pr", 141, 140.907657, 1.00]
            self.elemDetails["Re"]   = ["Rhenium", "Re", 187, 186.955765, 0.626]
            self.elemDetails["Rh"]   = ["Rhodium", "Rh", 103, 102.905503, 1.00]
            self.elemDetails["Rb"]   = ["Rubidium", "Rb", 85, 84.9118, 0.7217]
            self.elemDetails["Ru"]   = ["Ruthenium", "Ru", 102, 101.904348, 0.316]
            self.elemDetails["Sm"]   = ["Samarium", "Sm", 152, 151.919741, 0.267]
            self.elemDetails["Sc"]   = ["Scandium", "Sc", 45, 44.955914, 1.00]
            self.elemDetails["Se"]   = ["Selenium", "Se", 80, 79.916521, 0.496]
            self.elemDetails["Si"]   = ["Silicon", "Si", 28, 27.976928, 0.9223]
            self.elemDetails["Ag"]   = ["Silver", "Ag", 107, 106.905095, 0.5184]
            self.elemDetails["Na"]   = ["Sodium", "Na", 23, 22.98977, 1.00]
            self.elemDetails["Sr"]   = ["Strontium", "Sr", 88, 87.905625, 0.8258]
            self.elemDetails["S"]    = ["Sulfur", "S", 32, 31.972072, 0.9502]
            self.elemDetails["34S"]  = ["Sulfur", "S", 34, 33.967868, 0.0421]  # 1.995796
            self.elemDetails["Ta"]   = ["Tantalum", "Ta", 181, 180.948014, 0.9999]
            self.elemDetails["Te"]   = ["Tellurium", "Te", 130, 129.906229, 0.338]
            self.elemDetails["Tb"]   = ["Terbium", "Tb", 159, 158.92535, 1.00]
            self.elemDetails["Tl"]   = ["Thallium", "Tl", 205, 204.97441, 0.7048]
            self.elemDetails["Th"]   = ["Thorium", "Th", 232, 232.038054, 1.00]
            self.elemDetails["Tm"]   = ["Thulium", "Tm", 169, 168.934225, 1.00]
            self.elemDetails["Sn"]   = ["Tin", "Sn", 120, 119.902199, 0.324]
            self.elemDetails["Ti"]   = ["Titanium", "Ti", 48, 47.947947, 0.738]
            self.elemDetails["W"]    = ["Tungsten", "W", 184, 183.950953, 0.3067]
            self.elemDetails["U"]    = ["Uranium", "U", 238, 238.050786, 0.9927]
            self.elemDetails["V"]    = ["Vanadium", "V", 51, 50.943963, 0.9975]
            self.elemDetails["Xe"]   = ["Xenon", "Xe", 132, 131.904148, 0.269]
            self.elemDetails["Yb"]   = ["Ytterbium", "Yb", 174, 173.938873, 0.318]
            self.elemDetails["Y"]    = ["Yttrium", "Y", 89, 88.905856, 1.00]
            self.elemDetails["Zn"]   = ["Zinc", "Zn", 64, 63.929145, 0.486]
            self.elemDetails["66Zn"] = ["Zinc", "Zn", 66, 65.926035, 0.279]  # 1.99689
            self.elemDetails["67Zn"] = ["Zinc", "Zn", 67, 66.927129, 0.041]  # 2.997984
            self.elemDetails["68Zn"] = ["Zinc", "Zn", 68, 67.924846, 0.188]  # 3.995701
            self.elemDetails["Zr"]   = ["Zirconium", "Zr", 90, 89.904708, 0.5145]

        else:
            self.elemDetails = elemDetails
    # fmt: on

    # INTERNAL METHOD used for parsing
    # parses a number
    def _parseNumber(self, formula, pos):
        if pos >= len(formula):
            return -1, 1

        if formula[pos].isdigit():
            num = ""
            while formula[pos].isdigit() and pos < len(formula):
                num = num + formula[pos]
                pos = pos + 1
            return pos, int(num)
        else:
            return pos, 1

    # INTERNAL METHOD used for parsing
    # parses an element
    def _parseStruct(self, formula, pos):
        elemDict = {}

        if formula[pos] == "(":
            pos = pos + 1
            while formula[pos] != ")":
                pos, elem = self._parseStruct(formula, pos)
                for kE in elem.keys():
                    if kE in elemDict.keys():
                        elemDict[kE] = elemDict[kE] + elem[kE]
                    else:
                        elemDict[kE] = elem[kE]
            pos, numb = self._parseNumber(formula, pos + 1)
            for kE in elemDict.keys():
                elemDict[kE] = elemDict[kE] * numb
            return pos, elemDict
        elif formula[pos] == "[":
            pos = pos + 1

            num = ""
            while formula[pos].isdigit() and pos < len(formula):
                num = num + formula[pos]
                pos = pos + 1
            if pos == "":
                raise Exception("Isotope description wrong")

            curElem = formula[pos]
            if (pos + 1) < len(formula) and formula[pos + 1].isalpha() and formula[pos + 1].islower():
                curElem = formula[pos : (pos + 2)]
            if curElem != "":
                pos = pos + len(curElem)
            else:
                raise Exception("Unrecognized element")

            if formula[pos] != "]":
                raise Exception("Malformed isotope: " + formula)
            pos = pos + 1

            pos, numb = self._parseNumber(formula, pos)
            elemDict[curElem] = numb
            elemDict[num + curElem] = numb

            return pos, elemDict

        else:
            curElem = formula[pos]
            if (pos + 1) < len(formula) and formula[pos + 1].isalpha() and formula[pos + 1].islower():
                curElem = formula[pos : (pos + 2)]
            if curElem != "":
                pos, numb = self._parseNumber(formula, pos + len(curElem))
                elemDict[curElem] = numb
                return pos, elemDict
            else:
                raise Exception("Unrecognized element")
        return -1

    # parses a formula into an element-dictionary
    def parseFormula(self, formula):
        return self._parseStruct("(" + formula.replace(" ", "") + ")", 0)[1]

    # method determines if a given element represent an isotope other the the main isotope of a given element
    # e.g. isIso("13C"): True; isIso("12C"): False
    def isIso(self, iso):
        return not (iso[0].isalpha())

    # returns the element for a given isotope
    def getElementFor(self, elem):
        if not (self.isIso(elem)):
            raise Exception("Element was provided. Isotope is required")
        else:
            num = ""
            pos = 0
            while elem[pos].isdigit() and pos < len(elem):
                num = num + elem[pos]
                pos = pos + 1
            if pos == "":
                raise Exception("Isotope description wrong")
            curElem = elem[pos]

            if (pos + 1) < len(elem) and elem[pos + 1].isalpha() and elem[pos + 1].islower():
                curElem = elem[pos : (pos + 2)]
            if curElem != "":
                pos = pos + len(curElem)
            else:
                raise Exception("Unrecognized element")

            return curElem, num

    # helper method: n over k
    def noverk(self, n, k):
        return reduce(lambda a, b: a * (n - b) / (b + 1), xrange(k), 1)

    # helper method: calculates the isotopic ratio
    def getIsotopologueRatio(self, c, s, p):
        return pow(p, s) * self.noverk(c, s)

    def getMassOffset(self, elems):
        fElems = {}
        fIso = {}
        ret = 0
        for elem in elems:
            if not (self.isIso(elem)):
                fElems[elem] = elems[elem]
        for elem in elems:
            if self.isIso(elem):
                curElem, iso = self.getElementFor(elem)

                if not (fIso.has_key(curElem)):
                    fIso[curElem] = []
                fIso[curElem].append((iso, elems[elem]))

        for elem in fElems:
            rem = 0
            if fIso.has_key(elem):
                for x in fIso[elem]:
                    rem = rem + x[1]
            p = self.elemDetails[elem][4]
            c = fElems[elem] - rem
            ret = ret * pow(p, c)
        for iso in fIso:
            for cIso in fIso[iso]:
                ret = ret + (self.elemDetails[str(cIso[0] + iso)][3] - self.elemDetails[iso][3]) * cIso[1]
        return ret

    def getAbundance(self, elems):
        fElems = {}
        fIso = {}
        ret = 1.0
        for elem in elems:
            if not (self.isIso(elem)):
                fElems[elem] = elems[elem]
        for elem in elems:
            if self.isIso(elem):
                curElem, iso = self.getElementFor(elem)

                if not (fIso.has_key(curElem)):
                    fIso[curElem] = []
                fIso[curElem].append((iso, elems[elem]))
        for elem in fElems:
            rem = 0
            if fIso.has_key(elem):
                for x in fIso[elem]:
                    rem = rem + x[1]
            p = self.elemDetails[elem][4]
            c = fElems[elem] - rem
            ret = ret * pow(p, c)
        for iso in fIso:
            for cIso in fIso[iso]:
                ret = ret * self.getIsotopologueRatio(fElems[iso], cIso[1], self.elemDetails[str(cIso[0]) + iso][4])
        return ret

    def getAbundanceToMonoisotopic(self, elems):
        onlyElems = {}
        for elem in elems:
            if not (self.isIso(elem)):
                onlyElems[elem] = elems[elem]

        return self.getAbundance(elems) / self.getAbundance(onlyElems)

    # calculates the molecular weight of a given elemental collection (e.g. result of parseFormula)
    def calcMolWeight(self, elems):
        mw = 0.0
        for elem in elems.keys():
            if not (self.isIso(elem)):
                mw = mw + self.elemDetails[elem][3] * elems[elem]
            else:
                curElem, iso = self.getElementFor(elem)
                mw = mw + self.elemDetails[iso + curElem][3] * elems[elem] - self.elemDetails[curElem][3] * elems[elem]

        return mw

    # returns putaive isotopes for a given mz difference
    def getPutativeIsotopes(self, mzdiff, atMZ, z=1, ppm=5.0, maxIsoCombinations=1, used=[]):
        mzdiff = mzdiff * z
        maxIsoCombinations = maxIsoCombinations - 1

        ret = []

        for elem in self.elemDetails:
            if self.isIso(elem):
                curElem, iso = self.getElementFor(elem)
                diff = self.elemDetails[iso + curElem][3] - self.elemDetails[curElem][3]

                if maxIsoCombinations == 0:
                    if abs(mzdiff - diff) < (atMZ * ppm / 1000000.0):
                        x = [y for y in used]
                        x.append(iso + curElem)

                        d = {}
                        for y in x:
                            if not (d.has_key(y)):
                                d[y] = 0
                            d[y] = d[y] + 1
                        ret.append(d)

                else:
                    # print "next level with", mzdiff-diff, "after", iso+elem, self.elemDetails[iso+elem][3],self.elemDetails[elem][3]
                    x = [y for y in used]
                    x.append(iso + curElem)
                    x = self.getPutativeIsotopes(
                        mzdiff - diff,
                        atMZ=atMZ,
                        z=1,
                        ppm=ppm,
                        maxIsoCombinations=maxIsoCombinations,
                        used=x,
                    )
                    ret.extend(x)

        if maxIsoCombinations > 0:
            x = [y for y in used]
            x = self.getPutativeIsotopes(
                mzdiff,
                atMZ=atMZ,
                z=1,
                ppm=ppm,
                maxIsoCombinations=maxIsoCombinations,
                used=x,
            )
            ret.extend(x)

        return ret

    # prints a given elemental collection in form of a sum formula
    def flatToString(self, elems, prettyPrintWithHTMLTags=False, subStart="<sub>", subEnd="</sub>"):
        if isinstance(elems, str):
            elems = self.parseFormula(elems)

        if not prettyPrintWithHTMLTags:
            subStart = ""
            subEnd = ""

        fElems = {}
        for elem in elems:
            if not (self.isIso(elem)):
                fElems[elem] = elems[elem]
        for elem in elems:
            if self.isIso(elem):
                curElem, iso = self.getElementFor(elem)

                if fElems.has_key(curElem):
                    fElems[curElem] = fElems[curElem] - elems[elem]
                    fElems["[" + iso + curElem + "]"] = elems[elem]
                else:
                    fElems["[" + iso + curElem + "]"] = elems[elem]

        return "".join([("%s%s%d%s" % (e, subStart, fElems[e], subEnd) if fElems[e] > 1 else "%s" % e) for e in sorted(fElems.keys())])

    def getIsotopes(self, minInt=0.02):
        ret = {}
        for elem in self.elemDetails:
            if self.isIso(elem):
                el = self.getElementFor(elem)
                prob = self.elemDetails[elem][4] / self.elemDetails[el[0]][4]
                if prob >= minInt:
                    ret[elem] = (
                        self.elemDetails[elem][3] - self.elemDetails[el[0]][3],
                        prob,
                    )

        return ret

    def calcDifferenceBetweenElemDicts(self, elemsFragment, elemsParent):
        loss = {}
        for elem in elemsParent:
            l = 0
            if elem in elemsFragment:
                l = elemsParent[elem] - elemsFragment[elem]
            else:
                l = elemsParent[elem]
            if l > 0:
                loss[elem] = l
        return loss

    def calcDifferenceBetweenSumFormulas(self, sfFragment, sfParent):
        return self.calcDifferenceBetweenElemDicts(self.parseFormula(sfFragment), self.parseFormula(sfParent))


# helper method that returns the mass of a given isotope. Used in the main interface of MetExtract II
def getIsotopeMass(isotope):
    fT = formulaTools()
    mass = -1
    element = ""
    for i in fT.elemDetails:
        elemDetails = fT.elemDetails[i]
        if elemDetails[1][0].isdigit():
            if isotope == elemDetails[1]:
                mass = elemDetails[3]
                element = elemDetails[0]
        elif elemDetails[1][0].isalpha():
            if isotope == "%d%s" % (elemDetails[2], elemDetails[1]):
                mass = elemDetails[3]
                element = elemDetails[0]
    return mass, element


def getElementOfIsotope(isotope):
    fT = formulaTools()
    return fT.elemDetails[isotope][1]



def chunk_mgf_file(input_mgf_path, max_blocks=30000):
    """
    Generator that yields paths to temporary MGF files, each containing up to max_blocks spectra from the input file.
    Each yielded file is deleted when closed.
    """
    def write_chunk(chunk_lines):
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mgf", mode="w", encoding="utf-8")
        temp.writelines(chunk_lines)
        temp.close()
        return temp.name

    with open(input_mgf_path, "r", encoding="utf-8") as infile:
        chunk_lines = []
        block_count = 0
        for line in infile:
            chunk_lines.append(line)
            if line.strip().upper() == "END IONS":
                block_count += 1
                if block_count >= max_blocks:
                    temp_path = write_chunk(chunk_lines)
                    yield temp_path
                    chunk_lines = []
                    block_count = 0
        # Write any remaining spectra
        if chunk_lines:
            temp_path = write_chunk(chunk_lines)
            yield temp_path



def select_randomly_n_spectra(input_mgf_path, n = None):
    if n is None or n < 0:
        return input_mgf_path, False

    # Read all spectra blocks from the MGF file
    with open(input_mgf_path, "r", encoding="utf-8") as infile:
        spectra_blocks = []
        current_block = []
        for line in infile:
            current_block.append(line)
            if line.strip().upper() == "END IONS":
                spectra_blocks.append(current_block)
                current_block = []
        # Handle any trailing block (shouldn't happen in valid MGF)
        if current_block:
            spectra_blocks.append(current_block)

    if n >= len(spectra_blocks):
        return input_mgf_path, False

    selected_blocks = random.sample(spectra_blocks, n)

    # Write selected blocks to a new temporary file
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mgf", mode="w", encoding="utf-8")
    for block in selected_blocks:
        temp.writelines(block)
    temp.close()

    return temp.name, True

def get_and_print_conf_matrix(gt, pred, labels, print_pre = "", col_correct = Fore.GREEN, col_wrong = Fore.YELLOW):
    
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(gt, pred, labels = labels)
    conf_matrix_sum = conf_matrix.sum(axis=1, keepdims=True)
    conf_matrix_percent = (conf_matrix / conf_matrix_sum * 100).round(3)
    
    for i, row_label in enumerate(labels):
        for j, col_label in enumerate(labels):
            print(f"   - {col_correct if i == j else col_wrong}[{print_pre}] {row_label}({conf_matrix_sum[i, 0]}) -> {col_label}({conf_matrix[i, j]}): {conf_matrix_percent[i, j]:.3f}%{Style.RESET_ALL}")

    return conf_matrix_percent


def generate_ms2deepscore_embeddings(model_file_name, datasets, data_to_add=None):
    with execution_timer(title="MS2DeepScore"):
        print(f"\n\nRunning {Fore.YELLOW}MS2DeepScore{Style.RESET_ALL}")
        print("#######################################################")

        with execution_timer(title="loading model"):
            # load in the ms2deepscore model
            model = load_model(model_file_name)
            ms2ds_model = MS2DeepScore(model)

        # process each dataset
        for dataset in datasets:
            ds = datasets[dataset]
            print(f"\n\nGenerating MS2DeepScore embeddings for {Fore.YELLOW}{ds['name']}{Style.RESET_ALL}")

            with execution_timer(title=f"Dataset {dataset}"):
                with execution_timer(title="Running pipeline"):
                    
                    print(f"{dataset}: Processing spectra file {Fore.YELLOW}{ds["file"]}{Style.RESET_ALL}")
                    print(f"   - Type {Fore.YELLOW}{ds["type"]}{Style.RESET_ALL}")
                    
                    allowed_types = ["train - interesting", "train - other", "validation - interesting", "validation - other", "inference"]
                    if ds["type"] not in allowed_types:
                        raise ValueError(f"Dataset type {ds['type']} is not supported. Supported types are: {str(allowed_types)}.")
                    
                    # randomly sample from the input file spectra if requested by the user
                    fi = ds["file"]
                    fi_rm = False
                    if "randomly_sample" in ds.keys():
                        print(f"   - Randomly sampling {Fore.YELLOW}{ds["randomly_sample"]}{Style.RESET_ALL} spectra from input file")
                        fi, fi_rm = select_randomly_n_spectra(fi, ds["randomly_sample"])

                    # chunk the mgf file so intermediate objects are smaller
                    for chunki, mgf_chunk_file in enumerate(chunk_mgf_file(fi)):
                        print(f"   - processing chunk {chunki}")

                        with all_logging_disabled():
                            # define the pipeline
                            pipeline = Pipeline(
                                create_workflow(
                                    query_filters=DEFAULT_FILTERS,
                                    score_computations=[[MS2DeepScore, {"model": model}]],
                                )
                            )

                            # create embeddings
                            report = pipeline.run(mgf_chunk_file)
                            
                            # save results (and concatenate if chunking was necessary)
                            if "ms2deepscore:cleaned_spectra" not in ds:
                                ds["ms2deepscore:cleaned_spectra"] = pipeline.spectra_queries
                                ds["ms2deepscore:embeddings"] = ms2ds_model.get_embedding_array(pipeline.spectra_queries)
                            else:
                                ds["ms2deepscore:cleaned_spectra"].extend(pipeline.spectra_queries)
                                ds["ms2deepscore:embeddings"] = np.concatenate((ds["ms2deepscore:embeddings"], ms2ds_model.get_embedding_array(pipeline.spectra_queries)), axis = 0)

                        # manual cleanup of temporary chunk file
                        os.remove(mgf_chunk_file)

                    # manual cleanup of temporary randomly selected file (if used)
                    if fi_rm:
                        os.remove(fi)

            print(f"   - Imported {Fore.YELLOW}{ds["ms2deepscore:embeddings"].shape[0]}{Style.RESET_ALL} spectra")

    # Generate dataframe
    df = {
        "source": [],
        "type": [],
        "ms2deepscore:cleaned_spectra": [],
        "ms2deepscore:embeddings": [],
    }
    # add each dataset
    for dataset_name, ds in datasets.items():
        df["source"].extend([dataset_name] * len(ds["ms2deepscore:embeddings"]))
        df["type"].extend([ds["type"]] * len(ds["ms2deepscore:embeddings"]))
        df["ms2deepscore:cleaned_spectra"].extend(ds["ms2deepscore:cleaned_spectra"])
        df["ms2deepscore:embeddings"].extend(ds["ms2deepscore:embeddings"])
    df = pd.DataFrame(df)

    # add extra columns from each spectra (extract common keys from the MS/MS spectra)
    if data_to_add is not None:
        # extract metadata from the spectra
        for meta_info_to_add in data_to_add:
            fields = data_to_add[meta_info_to_add]
            meta_values = []

            for rowi, spectrum in enumerate(df["ms2deepscore:cleaned_spectra"]):
                if isinstance(fields, list):
                    value = None
                    for key in fields:
                        for key in [key, key.lower(), key.upper()]:
                            try:
                                value = spectrum.metadata_dict().get(key)
                                break
                            except:
                                pass
                        if value is not None:
                            break
                    
                    if value is None:
                        value = "NA"
                    meta_values.append(value)

                else:
                    meta_values.append(spectrum.metadata_dict().get(fields))

            df[meta_info_to_add] = meta_values

        # Perform post-processing of the main dataframe
        # Apply the processing function to the 'CE' column
        df["CE"] = df["CE"].apply(process_ce_value)
    
    return df




def add_mzmine_metainfos(datasets, df):
    allMZmine = []
    for dataset_name, ds in datasets.items():
        if "mzmine_meta_table" in ds.keys() and ds["mzmine_meta_table"] is not None and ds["mzmine_meta_table"] != "":
            print(f"Processing and adding MZmine meta-information from {Fore.YELLOW}{ds['mzmine_meta_table']}{Style.RESET_ALL}")
            # Read the quantification file as a TSV
            mzmine_df = pd.read_csv(ds["mzmine_meta_table"], sep=",", low_memory=False)
            mzmine_df["source"] = dataset_name

            # Update cols_to_keep with the new names
            all_cols = mzmine_df.columns.tolist()  # Get updated column names
            cols_to_keep = [
                "source",
                "id",
                "rt_range:min",
                "rt_range:max",
                "mz",
                "mz_range:min",
                "mz_range:max",
                "height",
                "area",
                "intensity_range:min",
                "intensity_range:max",
                "fragment_scans",
                "charge",
                "alignment_scores:rate",
                "alignment_scores:aligned_features_n",
                "alignment_scores:align_extra_features",
                "alignment_scores:weighted_distance_score",
                "alignment_scores:mz_diff_ppm",
                "alignment_scores:mz_diff",
                "alignment_scores:rt_absolute_error",
                "alignment_scores:ion_mobility_absolute_error",
                "feature_group",
                "ion_identities:iin_id",
                "ion_identities:ion_identities",
                "ion_identities:list_size",
                "ion_identities:neutral_mass",
                "ion_identities:partner_row_ids",
                "ion_identities:iin_relationship",
                "ion_identities:consensus_formulas",
                "ion_identities:simple_formulas",
                "compound_db_identity:compound_db_identity",
                "compound_db_identity:compound_name",
                "compound_db_identity:compound_annotation_score",
                "compound_db_identity:mol_formula",
                "compound_db_identity:adduct",
                "compound_db_identity:smiles",
                "compound_db_identity:precursor_mz",
                "compound_db_identity:mz_diff_ppm",
                "compound_db_identity:mz_diff",
                "compound_db_identity:neutral_mass",
                "compound_db_identity:rt",
                "spectral_db_matches:spectral_db_matches",
                "spectral_db_matches:compound_name",
                "spectral_db_matches:similarity_score",
                "spectral_db_matches:n_matching_signals",
                "spectral_db_matches:explained_intensity_percent",
                "spectral_db_matches:ion_adduct",
                "spectral_db_matches:mol_formula",
                "spectral_db_matches:smiles",
                "spectral_db_matches:inchi",
                "spectral_db_matches:neutral_mass",
                "spectral_db_matches:precursor_mz",
                "spectral_db_matches:mz_diff",
                "spectral_db_matches:mz_diff_ppm",
                "spectral_db_matches:rt_absolute_error",
                "molecular_networking:net_cluster_id",
                "molecular_networking:net_community_id",
                "molecular_networking:edges",
                "molecular_networking:net_cluster_size",
                "molecular_networking:net_community_size",
            ]

            # Select only the desired columns, handling potential missing columns gracefully
            mzmine_df = mzmine_df[[col for col in cols_to_keep if col in mzmine_df.columns]]

            allMZmine.append(mzmine_df)

    if len(allMZmine) > 0:

        # Concatenate all quantification dataframes into one
        mzmine_df = pd.concat(allMZmine, ignore_index=True)
        # Add prefix "mzmine_" to each column except 'source' and 'id'
        mzmine_df = mzmine_df.rename(columns={col: f"mzmine:{col}" for col in mzmine_df.columns if col not in ["source", "id"]})

        # Perform the left merge.
        # This merges the current quant_df with the entire df based on source and name/id.
        # Columns from quant_df are added to df. Rows not matching the join keys get NaN for these new columns.
        # If quant_df columns already exist in df (other than join keys), suffixes are added.
        df["id"] = df["name"].astype(str).str.split("_", n=1).str[1]
        df = pd.merge(
            df,
            mzmine_df,  # Contains columns like 'source', 'id', etc. for the current dataset
            left_on=["source", "id"],
            right_on=["source", "id"],
            how="left",
            suffixes=(
                "",
                "",
            ),  # Suffix for overlapping columns from the right dataframe (mzmine_df)
        )
        # Remove the 'id' column added by the merge
        df.drop(columns=["id"], inplace=True)

    return df



def add_sirius_fingerprints(datasets, df):
    df["sirius_fingerprint"] = None
    for dataset_name, ds in datasets.items():
        if "fingerprintFile" in ds.keys() and ds["fingerprintFile"] is not None and ds["fingerprintFile"] != "":
            if ds["fingerprintFile"] == "::RDKIT":
                pass
                ## TODO predict fingerprint of Standards and SMILE codes using RDKIt

            else:
                print(f"reading fingerprints from {Fore.YELLOW}{ds['fingerprintFile']}{Style.RESET_ALL}")
                with open(ds["fingerprintFile"], "r") as f:
                    fingerprints_data = json.load(f)

                    # Update the sirius_fingerprint column in df
                    for feature_name, fingerprint_info in fingerprints_data.items():
                        # Check if the feature name exists in the current dataset
                        mask = (df["source"] == dataset_name) & (df["name"] == feature_name)
                        assert mask.sum() == 1, f"Expected a single row match, but got {mask.sum()} matches."
                        df.loc[mask, "sirius_fingerprint"] = str(fingerprint_info["fingerprint"])

    return df   


def add_sirius_canopus(datasets, df):
    allCanopus = []
    for dataset_name, ds in datasets.items():
        if "canopus_file" in ds.keys() and ds["canopus_file"] is not None and ds["canopus_file"] != "" and ds["canopus_file"] != "::SIRIUS":
            print(f"reading Canopus annotations from {Fore.YELLOW}{ds['canopus_file']}{Style.RESET_ALL}")
            # Read the Canopus file as a TSV
            canopus_df = pd.read_csv(ds["canopus_file"], sep="\t")
            canopus_df["source"] = dataset_name
            canopus_df = canopus_df[
                [
                    "source",
                    "mappingFeatureId",
                    "molecularFormula",
                    "adduct",
                    "ClassyFire#most specific class",
                    "ClassyFire#most specific class Probability",
                ]
            ]
            allCanopus.append(canopus_df)
    
    if len(allCanopus) > 0:

        # Concatenate all Canopus dataframes into one
        canopus_df = pd.concat(allCanopus, ignore_index=True)
        # Add prefix "canopus:" to each column except 'source' and 'mappingFeatureId'
        canopus_df = canopus_df.rename(columns={col: f"canopus:{col}" for col in canopus_df.columns if col not in ["source", "mappingFeatureId"]})

        # Remove duplicate rows from canopus_df based on all columns. This is necessary as canopus entries are written twice to the file
        before = len(canopus_df)
        canopus_df = canopus_df.drop_duplicates()
        print(f"Of {before} cannopus annotations, {before - len(canopus_df)} were removed (duplicates)\n")

        print("Converting cannopus predictions to JSON format")
        # Define the columns to be included in the JSON string, dynamically select all columns except 'source' and 'mappingFeatureId'
        json_cols = [col for col in canopus_df.columns if col not in ["source", "mappingFeatureId"]]

        # Function to convert row data to JSON string
        def row_to_json(row):
            # Convert the selected columns to a dictionary
            data_dict = row[json_cols].to_dict()
            # Convert the dictionary to a JSON string
            return json.dumps(data_dict)

        # Apply the function to each row to create the 'sirius:predictions' column
        canopus_df["canopus:predictions"] = canopus_df.apply(row_to_json, axis=1)

        # Group by 'source' and 'mappingFeatureId' and aggregate 'sirius:predictions' into a list
        canopus_df = canopus_df.groupby(["source", "mappingFeatureId"])["canopus:predictions"].apply(list).reset_index()

        # Keep only relevant columns for potential later merging or analysis
        canopus_df = canopus_df[["source", "mappingFeatureId", "canopus:predictions"]]

        # Perform the left merge.
        # This merges the current canopus_df with the entire df based on source and name/mappingFeatureId.
        # Columns from canopus_df are added to df. Rows not matching the join keys get NaN for these new columns.
        # If canopus_df columns already exist in df (other than join keys), suffixes are added.
        df = pd.merge(
            df,
            canopus_df,  # Contains columns like 'source', 'mappingFeatureId', etc. for the current dataset
            left_on=["source", "name"],
            right_on=["source", "mappingFeatureId"],
            how="left",
            suffixes=(
                "",
                "",
            ),  # Suffix for overlapping columns from the right dataframe (canopus_df)
        )
        # Remove the 'mappingFeatureId' column added by the merge
        df.drop(columns=["mappingFeatureId"], inplace=True)

    return df


def add_sirius_predictions(datasets, df):
    allSIRIUS = []
    for dataset_name, ds in datasets.items():
        if "sirius_file" in ds.keys() and ds["sirius_file"] is not None and ds["sirius_file"] != "":
            print(f"reading SIRIUS annotations from {Fore.YELLOW}{ds['sirius_file']}{Style.RESET_ALL}")
            # Read the Canopus file as a TSV
            sirius_df = pd.read_csv(ds["sirius_file"], sep="\t")
            sirius_df["source"] = dataset_name
            sirius_df = sirius_df[
                [
                    "source",
                    "mappingFeatureId",
                    "structurePerIdRank",
                    "formulaRank",
                    "ConfidenceScoreExact",
                    "ConfidenceScoreApproximate",
                    "CSI:FingerIDScore",
                    "ZodiacScore",
                    "SiriusScoreNormalized",
                    "SiriusScore",
                    "molecularFormula",
                    "adduct",
                    "precursorFormula",
                    "InChIkey2D",
                    "InChI",
                    "name",
                    "smiles",
                    "xlogp",
                    "pubchemids",
                    "links",
                ]
            ]
            allSIRIUS.append(sirius_df)

    if len(allSIRIUS) > 0:

        # Concatenate all Canopus dataframes into one
        allSIRIUS = pd.concat(allSIRIUS, ignore_index=True)
        print(f"   .. imported {Fore.YELLOW}{len(allSIRIUS)}{Style.RESET_ALL} SIRIUS annotations\n")

        print("Converting SIRIUS predictions to JSON format")
        # Define the columns to be included in the JSON string, dynamically select all columns except 'source' and 'mappingFeatureId'
        json_cols = [col for col in allSIRIUS.columns if col not in ["source", "mappingFeatureId"]]

        # Function to convert row data to JSON string
        def row_to_json(row):
            # Convert the selected columns to a dictionary
            data_dict = row[json_cols].to_dict()
            # Convert the dictionary to a JSON string
            return json.dumps(data_dict)

        # Apply the function to each row to create the 'sirius:predictions' column
        allSIRIUS["sirius:predictions"] = allSIRIUS.apply(row_to_json, axis=1)

        # Group by 'source' and 'mappingFeatureId' and aggregate 'sirius:predictions' into a list
        allSIRIUS = allSIRIUS.groupby(["source", "mappingFeatureId"])["sirius:predictions"].apply(list).reset_index()

        # Keep only relevant columns for potential later merging or analysis
        allSIRIUS = allSIRIUS[["source", "mappingFeatureId", "sirius:predictions"]]

        # Merge df and allSIRIUS on 'source' and 'name' (df) with 'source' and 'mappingFeatureId' (allSIRIUS)
        df = df.merge(
            allSIRIUS,
            left_on=["source", "name"],
            right_on=["source", "mappingFeatureId"],
            how="left",
        )
        # Optionally drop the 'mappingFeatureId' column after merge
        df = df.drop(columns=["mappingFeatureId"])

    return df


def add_mzmine_quant(datasets, df):
    allQuant = []
    for dataset_name, ds in datasets.items():
        if "quant_file" in ds.keys() and ds["quant_file"] is not None and ds["quant_file"] != "":
            print(f"reading quantification information from {Fore.YELLOW}{ds['quant_file']}{Style.RESET_ALL}")
            # Read the quantification file as a TSV
            quant_df = pd.read_csv(ds["quant_file"], sep=",", low_memory=False)
            quant_df["source"] = dataset_name
            quant_df["datafile:max_quant_sample"] = None

            # Identify columns starting with 'datafile:'
            cols_to_rename = [col for col in quant_df.columns if col.startswith("datafile:")]
            # Create the renaming dictionary
            rename_mapping = {col: col.replace("datafile:", f"datafile:{dataset_name}:") for col in cols_to_rename}
            # Rename the columns
            quant_df.rename(columns=rename_mapping, inplace=True)

            # Update cols_to_keep with the new names
            all_cols = quant_df.columns.tolist()  # Get updated column names
            cols_to_keep = [
                "source",
                "id",
                "area",
                "height",
                "datafile:max_quant_sample",
            ]
            # Add columns matching the new pattern 'datafile:...:area' or 'datafile:...:height'
            for col in all_cols:
                if col.startswith("datafile:") and (col.endswith(":area") or col.endswith(":height")):
                    # Ensure we don't add duplicates if base columns somehow match the pattern
                    if col not in cols_to_keep:
                        cols_to_keep.append(col)

            # Select only the desired columns, handling potential missing columns gracefully
            quant_df = quant_df[[col for col in cols_to_keep if col in quant_df.columns]]
            datasets[dataset_name]["samples_columns"] = [col.replace("datafile:", "").replace(":area", "").replace(".mzML", "").replace(".mzXML", "") for col in quant_df.columns if col.startswith("datafile:") and col.endswith(":area")]

            allQuant.append(quant_df)

    if len(allQuant) > 0:

        # Concatenate all quantification dataframes into one
        quant_df = pd.concat(allQuant, ignore_index=True)

        # Identify the area quantification columns present in the current quant_df
        quant_area_cols = [col for col in quant_df.columns if col.startswith("datafile:") and col.endswith(":area") and "PT24-CH-" in col]

        # Check if there are any quantification columns to process
        if quant_area_cols:
            # Find the column name with the maximum area for each row, ignoring NaNs
            # Ensure data is numeric before finding the max index
            max_area_col_name = quant_df[quant_area_cols].apply(pd.to_numeric, errors="coerce").idxmax(axis=1, skipna=True)

            # Clean the column name (remove prefix/suffix) and add it to the DataFrame
            # Handle rows where all values might be NaN (idxmax returns NaN in this case)
            quant_df["datafile:max_quant_sample"] = max_area_col_name
        else:
            # If no quant columns, add a column with missing values
            quant_df["datafile:max_quant_sample"] = pd.NA

        # Perform the left merge.
        # This merges the current quant_df with the entire df based on source and name/id.
        # Columns from quant_df are added to df. Rows not matching the join keys get NaN for these new columns.
        # If quant_df columns already exist in df (other than join keys), suffixes are added.
        df["id"] = df["name"].astype(str).str.split("_", n=1).str[1]
        df = pd.merge(
            df,
            quant_df,  # Contains columns like 'source', 'id', etc. for the current dataset
            left_on=["source", "id"],
            right_on=["source", "id"],
            how="left",
            suffixes=(
                "",
                "",
            ),  # Suffix for overlapping columns from the right dataframe (quant_df)
        )
        # Remove the 'id' column added by the merge
        df.drop(columns=["id"], inplace=True)

    return df

def remove_invalid_CEs(df):
    print(f"Removing MS/MS spectra with merged collision energies")

    # Remove certain MSMS spectra that have multiple collision energies
    def process_ce(value):
        if isinstance(value, str):
            if value.startswith("[") and value.endswith("]"):
                # Parse the string as a list
                parsed_array = list(map(float, value.strip("[]").split(",")))
                parsed_array = natsorted(parsed_array)
                # Check if the list contains multiple floats
                if len(parsed_array) == 1:
                    return str(int(parsed_array[0]))  # Remove square brackets if single float
                else:
                    return None  # Mark for removal if multiple floats
        return value  # Do nothing if it's already a float-like string

    # Process CE values
    df["CE_processed"] = df["CE"].apply(process_ce)
    
    # Filter rows with valid CE and store their indices
    df = df.reset_index()
    valid_CEs = df["CE_processed"].notna()
    # Remvoe entries with invalid CEs
    initial_row_count = len(df)
    valid_indices = valid_CEs[valid_CEs].index.tolist()
    # Remove invalid rows from the dataframe
    # Show an overview of the rows to be removed
    print("\nOverview of rows to be removed:")
    rows_to_remove_indices = df.index[~valid_CEs]
    rows_to_remove = df.loc[rows_to_remove_indices]

    print(f"Number of rows to be removed: {len(rows_to_remove)}")
    print("Unique 'CE' values in rows to be removed:")
    # Display unique values, converting potential lists/arrays to strings for consistent printing
    unique_ces_to_remove = rows_to_remove["CE"].unique()
    for ce in unique_ces_to_remove:
        print(f"   - {ce}")

    # Store the removed rows in rdf before filtering df
    rdf = df.loc[~df.index.isin(valid_indices)].reset_index(drop=True)
    df = df.iloc[valid_indices].reset_index(drop=True)

    # Remove the CE_processed column
    df["CE"] = df["CE_processed"]
    df = df.drop(columns=["CE_processed"])

    # Print the number of removed rows
    removed_row_count = initial_row_count - len(df)
    print(f"Number of removed rows: {removed_row_count}, remaining rows: {len(df)}")

    return df

def show_dataset_overview(df):
    # Overview of CE, ionMode, and adduct columns
    for column in ["CE", "ionMode", "adduct", "fragmentation_method"]:
        summary_table = pd.DataFrame()

        for dataset_name in df["source"].unique():
            subset_df = df[df["source"] == dataset_name]
            value_counts = subset_df[column].value_counts()
            value_counts = value_counts.reset_index()
            value_counts.columns = [column, f"{dataset_name}_count"]
            # value_counts[f"{dataset_name}_percentage"] = (value_counts[f"{dataset_name}_count"] / value_counts[f"{dataset_name}_count"].sum()) * 100
            # value_counts[f"{dataset_name}_percentage"] = value_counts[f"{dataset_name}_percentage"].round(3)

            if summary_table.empty:
                summary_table = value_counts
            else:
                summary_table = pd.merge(summary_table, value_counts, on=column, how="outer")
        summary_table = summary_table.fillna(0)

        print(f"\nSummary table for column: {column}")
        print(f"-------------------------------------------------")
        print(summary_table)

def generate_embedding_plots(df, output_dir, colors):
    with execution_timer(title="Generating UMAP/pacmap embeddings"):
        # output the embeddings
        print("\n\nGenerating embeddings")
        print("#######################################################")

        # UMAP
        print("Generating UMAP embeddings")
        umap_reducer = umap.UMAP(
            random_state=42,  # this or whatever your favorite number is
            n_neighbors=50,  # key parameters How global or local the distribution 30, 50
            min_dist=0.2,  # can the dots overlap if you use 5 they move out a bit. 0.1, 0.2
        )
        umap_embeddings = umap_reducer.fit(np.array(df["ms2deepscore:embeddings"].tolist()))

        # PacMAP
        print("Generating PaCMAP embeddings")
        pacmap_reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)
        pacmap_embedding = pacmap_reducer.fit_transform(np.array(df["ms2deepscore:embeddings"].tolist()), init="pca")

        df["umap_1"] = umap_embeddings.embedding_[:, 0]
        df["umap_2"] = umap_embeddings.embedding_[:, 1]
        df["pacmap_1"] = pacmap_embedding[:, 0]
        df["pacmap_2"] = pacmap_embedding[:, 1]

    with execution_timer(title="plotting embeddings"):
        # plot the embeddings using plotnine
        print("\n\nPlotting the embeddings")
        print("#######################################################")

        p_umap = (
            p9.ggplot(
                df,
                mapping=p9.aes(x="umap_1", y="umap_2", label="name", colour="source", shape="ionMode"),
            )
            + p9.geom_point(data=df[df["source"] == "MB BOKU"], alpha=0.3)
            + p9.geom_point(data=df[df["source"] != "MB BOKU"], alpha=0.7)
            + p9.facet_wrap("source")
            # + p9.geom_text(nudge_x=0.025, nudge_y=0.025, size=5, colour="slategrey")
            + p9theme()
            + p9.scale_colour_manual(values=colors)
            + p9.labs(
                title="UMAP embeddings of the spectra",
                subtitle="calculated from MS2DeepScore embeddings",
            )
        )
        out_file = os.path.join(output_dir, "umap_from_MS2DeepScoreEmbeddings.pdf")
        p_umap.save(out_file, width=16, height=12)
        print(f"UMAP plot saved as {out_file}")

        p_pacmap = (
            p9.ggplot(
                df,
                mapping=p9.aes(
                    x="pacmap_1",
                    y="pacmap_2",
                    label="name",
                    colour="source",
                    shape="ionMode",
                ),
            )
            + p9.geom_point(data=df[df["source"] == "MB BOKU"], alpha=0.3)
            + p9.geom_point(data=df[df["source"] != "MB BOKU"], alpha=0.7)
            + p9.facet_wrap("source")
            # + p9.geom_text(nudge_x=0.025, nudge_y=0.025, size=5, colour="slategrey")
            + p9theme()
            + p9.scale_colour_manual(values=colors)
            + p9.labs(
                title="pacmap embeddings of the spectra",
                subtitle="calculated from MS2DeepScore embeddings",
            )
        )
        out_file = os.path.join(output_dir, "pacmap_from_MS2DeepScoreEmbeddings.pdf")
        p_pacmap.save(out_file, width=16, height=12)
        print(f"PACMAP plot saved as {out_file}")

        ## TODO include heatmap here


def train_and_classify(df, subsets = None):
    # Test different classifiers

    classifiers = {
        "Nearest Neighbors n=3": KNeighborsClassifier(3),
        "Nearest Neighbors n=10": KNeighborsClassifier(10),
        "Linear SVM": SVC(kernel="linear", C=0.025, random_state=42),
        #"RBF SVM": SVC(gamma=2, C=1, random_state=42),
        # "Gaussian Process":         GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42),
        "Neural Net": MLPClassifier(alpha=1, max_iter=1000, random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "LDA": LinearDiscriminantAnalysis(solver="svd", store_covariance=True, n_components=1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
        # "Gradient Boosting":        GradientBoostingClassifier(random_state=42),
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
    }

    if subsets is None:
        subsets = {
            "positive"     : lambda x: x["ionMode"] == "positive",
            "negative"     : lambda x: x["ionMode"] == "negative",
        }

    # Execute classifiers with cross-validation
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.8, train_size=0.2, random_state=42)
    cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

    # Filter the dataframe and ms2_embeddings for training and inference
    df_train = df[df["type"].isin(["train - interesting", "train - other"])].reset_index()
    ms2ds_embeddings_train = np.array(df_train["ms2deepscore:embeddings"].tolist())
    df_train = df_train.drop(columns=["ms2deepscore:cleaned_spectra", "ms2deepscore:embeddings"])
    print(f"Size of training dataset: {Fore.YELLOW}{ms2ds_embeddings_train.shape[0]}{Style.RESET_ALL}")

    df_vali = df[df["type"].isin(["validation - interesting", "validation - other"])].reset_index()
    X_vali = np.array(df_vali["ms2deepscore:embeddings"].tolist())
    df_vali = df_vali.drop(columns=["ms2deepscore:cleaned_spectra", "ms2deepscore:embeddings"])
    y_vali_gt = df_vali["type"].values
    y_vali_gt = np.array(["interesting" if s.lower() == "validation - interesting" else "other" for s in y_vali_gt])
    print(f"Size of validation dataset: {Fore.YELLOW}{X_vali.shape[0]}{Style.RESET_ALL}")

    df_infe = df[df["type"].isin(["inference"])].reset_index()
    X_infe = np.array(df_infe["ms2deepscore:embeddings"].tolist())
    df_infe = df_infe.drop(columns=["ms2deepscore:cleaned_spectra", "ms2deepscore:embeddings"])
    df_infe["prediction_results"] = [[] for _ in range(len(df_infe))]
    print(f"Size of inference dataset: {Fore.YELLOW}{X_infe.shape[0]}{Style.RESET_ALL}")

    labels = ["interesting", "other"]
    with execution_timer(title="Classifiers with Cross-Validation"):
        for subset in subsets:
            with execution_timer(title=f"Subset: {subset}"):
                print(f"\n\n\n********************************************************************************")
                print(f"Subset: {subset}")

                # subset the data for training
                useInds_train = df_train[df_train.apply(subsets[subset], axis=1)].index.tolist()
                if len(df_train) == 0:
                    print(f"{Fore.RED}Skipping subset '{subset}' because df_train is empty.{Style.RESET_ALL}")
                    continue
                print(f"Number of spectra in subset (train): {len(useInds_train)}, these are {len(useInds_train) / len(df_train) * 100:.2f}% of the total spectra.")

                # Subselect the embeddings for the training and inference sets
                X_train, y_train_gt = (
                    ms2ds_embeddings_train[useInds_train, :],
                    df_train["type"].values[useInds_train],
                )
                y_train_gt = np.array(["interesting" if s.lower() == "train - interesting" else "other" for s in y_train_gt])
                
                # Show an overview of y_train_gt
                unique, counts = np.unique(y_train_gt, return_counts=True)
                print("Overview of y_train_gt (ground-truth labels):")
                for label, count in zip(unique, counts):
                    print(f"   - {label}: {count}")

                # Only continue if there are at least two different labels in y_train_gt
                if len(np.unique(y_train_gt)) < 2:
                    print(f"{Fore.RED}Skipping subset '{subset}' because only one class present in y_train_gt: {np.unique(y_train_gt)}{Style.RESET_ALL}")
                    continue

                # Perform classifiers with cross-validation
                for cname in classifiers:
                    clf = classifiers[cname]

                    with execution_timer(title=f"Classifier: {subset} / {cname}"):
                        print(f"\n--------------------------------------------------------------------------------")
                        print(f"Classifier: {cname}")

                        fold_scores = []
                        fold_durations = []
                        fold_conf_matrices = []

                        # Perform 5-fold cross-validation
                        for fold, (train_idx, test_idx) in enumerate(cv.split(X_train, y_train_gt)):
                            print("")
                            X_train_train, X_train_test = (
                                X_train[train_idx, :],
                                X_train[test_idx, :],
                            )
                            y_train_train_gt, y_train_test_gt = (
                                y_train_gt[train_idx],
                                y_train_gt[test_idx],
                            )

                            # Train the model
                            start_time = time.time()
                            clf.fit(X_train_train, y_train_train_gt)
                            duration = time.time() - start_time

                            # Test on the test set
                            y_train_test_pred = clf.predict(X_train_test)
                            score = np.mean(y_train_test_pred == y_train_test_gt)

                            conf_matrix_percent = get_and_print_conf_matrix(y_train_test_gt, y_train_test_pred, labels, "Test set")

                            fold_scores.append(score)
                            fold_durations.append(duration)
                            fold_conf_matrices.append(conf_matrix_percent)

                            print(f"   [Fold {fold + 1}] Score: {Fore.YELLOW}{score:.3f}{Style.RESET_ALL}, Duration: {Fore.YELLOW}{duration:.2f} seconds{Style.RESET_ALL}")

                            # process validation dataset
                            if X_vali.shape[0] > 0:
                                # Predict on the inference embeddings
                                y_vali_pred = clf.predict(X_vali)
                                # print confusion matrix
                                get_and_print_conf_matrix(y_vali_gt, y_vali_pred, labels, "Validation set", col_correct=Fore.MAGENTA, col_wrong=Fore.CYAN)

                            # process inference dataset
                            if X_infe.shape[0] > 0:
                                # Predict on the inference embeddings
                                y_infe_pred = clf.predict(X_infe)

                                # Count the number of 'prenylated flavonoid' and others in the inference predictions
                                y_infe_pred_interesting_count = np.sum(y_infe_pred == "interesting")
                                y_infe_pred_other_count = len(y_infe_pred) - y_infe_pred_interesting_count

                                print(f"   * [Inference] Number of 'interesting': {Fore.YELLOW}{y_infe_pred_interesting_count}{Style.RESET_ALL}")
                                print(f"   * [Inference] Number of 'other'      : {Fore.YELLOW}{y_infe_pred_other_count}{Style.RESET_ALL}")

                                # Store the results in the inference_results dictionary
                                for idx in range(df_infe.shape[0]):
                                    if y_infe_pred[idx] == "interesting":
                                        df_infe.iloc[idx]["prediction_results"].append(f"{subset} / {cname} / {fold}")

                        # Calculate average results
                        avg_score, std_score, min_score, max_score = (
                            np.mean(fold_scores),
                            np.std(fold_scores),
                            np.min(fold_scores),
                            np.max(fold_scores),
                        )
                        avg_duration, std_duration = (
                            np.mean(fold_durations),
                            np.std(fold_durations),
                        )
                        avg_conf_matrix_percent = np.mean(fold_conf_matrices, axis=0)
                        avg_conf_matrix_percent_df = pd.DataFrame(
                            avg_conf_matrix_percent,
                            index=np.unique(y_train_gt),
                            columns=np.unique(y_train_gt),
                        )

                        print(f"\nAverage score: {avg_score:.3f} ± {std_score:.3f} (min: {min_score:.3f}, max: {max_score:.3f})")
                        print(f"Average duration: {avg_duration:.2f} seconds")
                        print("Average Confusion Matrix (Percentages, rows: ground-truth, columns: predictions):")
                        print(avg_conf_matrix_percent_df)

    return df_infe


def generate_prediction_overview(df, df_inference, output_dir, min_prediction_threshold = 120):
    # Generate the prediction results
    # Aggregate the df_inference results and extract the number of times it was predicted to be a compound of interest
    # Calculate the count of predictions for each row
    df_inference["prediction_count"] = df_inference["prediction_results"].apply(len)

    # Sort the DataFrame by count in descending order
    aggregated_df = df_inference.sort_values(by="prediction_count", ascending=False).reset_index(drop=True)

    # Create the bar chart using plotnine    
    plot = (
        p9.ggplot(aggregated_df, p9.aes(x="prediction_count"))
        + p9.geom_bar()
        + p9.geom_vline(xintercept=min_prediction_threshold, linetype="dashed", color="Firebrick")
        + p9theme()
        + p9.theme(axis_text_x=p9.element_text(angle=90, hjust=1))  # Rotate x-axis labels for readability
        + p9.labs(
            title="Counts",
            subtitle="counts are the number of times a feature\nwas predicted to be a compound of interest",
            x="Unique Row",
            y="Count",
        )
    )

    # Print the plot
    out_file = os.path.join(output_dir, "interesting_predictions_classificationChart.pdf")
    plot.save(out_file, width=16, height=12)
    print(f"plot saved as {out_file}")


    # Add a new column 'prediction' to df with default value 'NA'
    df["classification:interesting"] = ""

    # Update the 'prediction' column for rows present in aggregated_df
    for _, row in aggregated_df.iterrows():
        if row["prediction_count"] > min_prediction_threshold:
            # Create a mask to identify matching rows in df
            mask = (
                (df["source"] == row["source"])
                & (df["name"] == row["name"])
                & (df["CE"] == row["CE"])
                & (df["fragmentation_method"] == row["fragmentation_method"])
                & (df["ionMode"] == row["ionMode"])
                & (df["precursor_mz"] == row["precursor_mz"])
                & (df["RTINSECONDS"] == row["RTINSECONDS"])
            )
            # Set the prediction value to "interesting" for matching rows
            df.loc[mask, "classification:interesting"] = "interesting"

    p_umap = (
        p9.ggplot(
            df,
            mapping=p9.aes(
                x="umap_1",
                y="umap_2",
                label="name",
                colour="classification:interesting",
                shape="ionMode",
            ),
        )
        + p9.geom_point(alpha=0.3)
        # + p9.geom_text(nudge_x=0.025, nudge_y=0.025, size=5, colour="slategrey")
        + p9theme()
        + p9.labs(
            title="UMAP embeddings of the spectra",
            subtitle="calculated from MS2DeepScore embeddings\nprenylated compounds are predicted based on classifier majority vote",
        )
    )

    # Print the plot
    out_file = os.path.join(output_dir, "interesting_predictions_umap.pdf")
    p_umap.save(out_file, width=16, height=12)
    print(f"Umap plot saved as {out_file}")

    # Filter rows annotated as 'prenylated flavonoid' in the prediction column
    subset_df = df[df["classification:interesting"] == "interesting"].copy()
    subset_df = df.copy()

    try:
        # Plot the feature map using plotnine
        temp = subset_df.copy()
        if "area" in temp.columns:
            temp["area_log10"] = np.log10(temp["area"] + 1e-9)
        else:
            temp["area_log10"] = 5
        feature_map_plot = (
            p9.ggplot(
                temp,
                p9.aes(x="RTINSECONDS", y="precursor_mz", size="area_log10", color="classification:interesting"),
            )
            + p9.geom_point(alpha=0.1)
            + p9theme()
            + p9.labs(
                title="Feature Map of predicted Prenylated Flavonoids",
                x="Retention Time (s)",
                y="Precursor m/z",
            )
        )

        # Print the plot
        out_file = os.path.join(output_dir, "interesting_predictions_feature_map.pdf")
        feature_map_plot.save(out_file, width=16, height=12)
        print(f"Feature map plot saved as {out_file}")

    except Exception as e:
        print(f"{Fore.RED}Error while plotting feature map: {e}{Style.RESET_ALL}")
        print("Skipping feature map plot as 'area' column is missing or not suitable for plotting.")

    # Modify the 'name' column to keep only the part after the first underscore
    subset_df["name"] = subset_df["name"].apply(lambda x: x.split("_", 1)[1] if "_" in x else x)

    # Save subset_df to an HDF5 file
    out_file_h5 = os.path.join(output_dir, "interesting_predictions_long.h5")
    subset_df.to_hdf(out_file_h5, key="df", mode="w")
    print(f"Saved table to {out_file_h5}")

    # Pivot the table
    aggClassification = lambda x: str(sum(d == "interesting" for d in x)) if sum(d == "interesting" for d in x) > 0 else ""
    pivot_table = (
        subset_df[
            [
                "source",
                "type",
                "fragmentation_method",
                "ionMode",
                "name",
                "formula",
                "smiles",
                "RTINSECONDS",
                "precursor_mz",
                #"height",
                #"area",
                "CE",
                "classification:interesting",
            ]
        ]
        .reset_index(drop=True)
        .pivot_table(
            index=[
                "source",
                "type",
                "fragmentation_method",
                "ionMode",
                "name",
                "formula",
                "smiles",
                "RTINSECONDS",
                "precursor_mz",
                #"height",
                #"area",
            ],
            columns="CE",
            aggfunc={
                "classification:interesting": aggClassification,
            },
            fill_value="",
        )
    )

    for sumCols, newCol in [
        ("classification:interesting", "annotated_as_times:interesting"),
    ]:
        # Select all columns under the 'CE' level in the MultiIndex
        ce_columns = [col for col in pivot_table.columns if col[0] == sumCols]
        # Count the number of '1' values in the selected CE columns
        count_ones = pivot_table[ce_columns].applymap(lambda x: int(x) if x != "" else 0).sum(axis=1)
        pivot_table[newCol] = count_ones.values.reshape(-1, 1)

    # Extract column names starting with 'datafile:'
    additional_columns = [col for col in subset_df.columns if (col.startswith("datafile:") and col.endswith(":area")) or col.startswith("mzmine:")]
    # Reduce interesting_df to the 'name' column and all columns from datafile_columns
    reduced_df = subset_df[["name"] + additional_columns].drop_duplicates()
    # Reindex reduced_df to match the order of 'name' in pivot_table's index
    reduced_df = reduced_df.set_index("name").reindex(pivot_table.index.get_level_values("name")).reset_index()
    assert reduced_df["name"].tolist() == pivot_table.index.get_level_values("name").tolist(), "The 'name' columns in reduced_df and pivot_table are not identical."

    # Perform a left join between pivot_table and reduced_df on the 'name' column
    for datafile_column in additional_columns:
        pivot_table[datafile_column] = reduced_df[datafile_column].values

    # Sort the pivot_table by 'annotated_as_interesting' in descending order
    pivot_table = pivot_table.sort_values(
        by=[
            "annotated_as_times:interesting",
        ],
        ascending=False,
    )

    # Save the filtered DataFrame to an Excel file
    out_file = os.path.join(output_dir, "interesting_predictions_long.xlsx")
    subset_df.reset_index().to_excel(out_file, index=True)
    print(f"Saved table to {out_file}")

    out_file = os.path.join(output_dir, "interesting_predictions.xlsx")
    pivot_table.reset_index().to_excel(out_file, index=True)
    print(f"Saved pivot_table to {out_file}")
