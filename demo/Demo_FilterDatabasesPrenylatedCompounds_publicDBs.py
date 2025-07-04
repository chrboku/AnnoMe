from AnnoMe.Filters import process_database, prep_smarts_key, download_common_MSMS_libraries, download_MS2DeepScore_model, CE_parser
from collections import OrderedDict
import pathlib
import re
from colorama import Fore, Style
import os
import shutil


# Download the common MS/MS libraries if they do not exist
libraries_path = "./demo/publicDBs/libraries"
if not os.path.exists(libraries_path):
    print(f"Common MS/MS libraries not found in {Fore.YELLOW}{libraries_path}{Style.RESET_ALL}. Downloading...")
    # Create the directory if it does not exist
    os.makedirs(libraries_path, exist_ok=True)
    try:
        download_common_MSMS_libraries(libraries_path)
    except Exception as e:
        shutil.rmtree(libraries_path)
        print(f"{Fore.RED}Error downloading common MS/MS libraries: {e}{Style.RESET_ALL}")
        print(f"{Fore.RED}Please check your internet connection or the availability of the libraries.{Style.RESET_ALL}")
        raise e
else:
    print(f"Common MS/MS libraries found in {Fore.GREEN}{libraries_path}.{Style.RESET_ALL}, skipping download and processing")

# Download the MS2DeepScore model if it does not exist
model_path = f"./demo/publicDBs/models"
if not os.path.exists(model_path):
    print(f"MS2DeepScore model not found in {Fore.YELLOW}{model_path}{Style.RESET_ALL}. Downloading...")
    # Create the directory if it does not exist
    os.makedirs(model_path, exist_ok=True)
    try:
        download_MS2DeepScore_model(model_path)
    except Exception as e:
        shutil.rmtree(model_path)
        print(f"{Fore.RED}Error downloading MS2DeepScore model: {e}{Style.RESET_ALL}")
        print(f"{Fore.RED}Please check your internet connection or the availability of the model.{Style.RESET_ALL}")
        raise e
else:
    print(f"MS2DeepScore model found in {Fore.GREEN}{model_path}.{Style.RESET_ALL}, skipping download and processing")

import sys


## Parameters
# fmt: off
flavone_smart = prep_smarts_key("O=C@1@C@C(@O@C2@C@C@C@C@C@1@2)C@3@C@C@C@C@C3")
isoflavone_smart = prep_smarts_key("O=C@1@C@2@C@C@C@C@C@2@O@C@C@1C@3@C@C@C@C@C@3")
chalcone1_smart = prep_smarts_key("[O,o]=[C,c](-[CH2]-[CH2]-[C,c]@1@[C,c]@[C,c]@[C,c]@[C,c]@[C,c]@1)-[C,c]@2@[C,c]@[C,c]@[C,c]@[C,c]@[C,c]@2", replace = False)
chalcone2_smart = prep_smarts_key("[O,o]=[C,c](-[CH]=[CH]-[C,c]@1@[C,c]@[C,c]@[C,c]@[C,c]@[C,c]@1)-[C,c]@2@[C,c]@[C,c]@[C,c]@[C,c]@[C,c]@2", replace = False)
chromone_smart = prep_smarts_key("O=C@1@C@C@O@C@2@C@C@C@C@C@12")

def CE_parser(ce_str):
    """
    Parses the collision energy string to extract the numeric value.
    Handles both absolute and relative values.
    """
    if isinstance(ce_str, str):

        # Try to match patterns like '70 eV (absolute)', '75eV', '75.0eV', '15V'
        match = re.search(r"(\d+(?:\.\d+)?)\s*(eV|V)?", ce_str, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

        # Try to match patterns like '[45.0]', '[45]'
        match = re.match(r"\[(\d+(?:\.\d+)?)\]", ce_str)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    return -1



def filter_generic(spectra): 
    instrument_regex = re.compile(".*(orbitrap|q exactive|q-exactive|qexactive|exploris).*", re.IGNORECASE)
    return [block for block in spectra if 
            "instrument" in block and instrument_regex.search(block["instrument"]) and 
            #"fragmentation_method" in block and block["fragmentation_method"].lower() in ["hcd", "cid"] and
            #"collision_energy" in block and 10 <= CE_parser(block["collision_energy"]) <= 90 and
            True]

input_data = {
    #"gnps_cleaned": {"mgf_file": f"{libraries_path}/gnps_cleaned.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
    "MassBank_RIKEN": {"mgf_file": f"{libraries_path}/MassBank_RIKEN.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
    "MassSpecGym": {"mgf_file": f"{libraries_path}/MassSpecGym.mgf", "smiles_field": "smiles", "name_field": "inchikey", "sf_field": "formula", "filter": filter_generic},
    "MONA": {"mgf_file": f"{libraries_path}/MONA.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
    "WINE-DB-ORBITRAP": {"mgf_file": f"{libraries_path}/WINE-DB-ORBITRAP.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
}

# Visualize SMARTS with https://smarts.plus/view/1cf72609-6995-4b25-8a16-42eeeb8c09df
checks = OrderedDict([
    ("StructureOfInterest", {"filter": [[flavone_smart, isoflavone_smart, chalcone1_smart, chalcone2_smart], 
                                        [prep_smarts_key(x) for x in ["CC=C(C)CCC=C(C)C", "C\\C=C(/C)\\CCC(O)C(=C)C", "CC(O)C(C)CCC=C(C)C", "CC=C(C)CCCC(C)(C)O", "CC(=CCCC1(C)OCCC=C1)C", "CC(=C)C(O)CCC1(C)OC=CCC1O", "C\\C=C(/C)\\CCCC(C)(O)CO", "CC(=CCCC(C)(O)C1CCCO1)C", "CC1(C)C(O)CCC2(C)OCCCC12", "CC1C(=CCC(O)C1(C)C)C", "CC1C(=C)CCC(O)C1(C)C", "COCC(C)(O)CCCC(=CC)C", "CC(=CCCC1(C)OC=CCC1O)C", "CC1(C)OCCC=C1", "CC1OC=CC1(C)C", "CC(CC=C(C)C)C(=C)C", "CCC(=C)C", "CC=C(C)C", "C-C(-C)-C=C", "CC(=C)C=C", "CC(=CCO)C", "CC(C)(O)C=C", "CCC(C)(C)O", "CC(O)C(=C)C", "CC1OC1(C)C", "C\\C=C(/C)\\CO", "CC(O)C(C)(C)O", "CC1(C)CCc2ccccc2O1", "CC1(C)CCCCO1"]]]}),
])
include_details = False

out_path = f"{libraries_path}/derived/"

def evAbs(x):
    match = re.match(r"(\d+(?:\.\d+)?)\s*eV\s*\(absolute\)", str(x), re.IGNORECASE)
    if match:
        return f"[{float(match.group(1))}]"
    return x
standardize_block_functions = {
    #"adduct": [lambda x: x],
    "collision_energy": [evAbs],
    "fragmentation_method": [lambda x: x.lower()],
    "instrument": [lambda x: "Orbitrap"],
    "ionmode": [lambda x: "+" if x.lower() in ["positive", "pos", "p", "+"] else "-" if x.lower() in ["negative", "neg", "n", "-"] else "ERROR"],
}
# fmt: on

## Process each database
generated_files = []
output_folder_existed = pathlib.Path(out_path).exists()
pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
for database_name in input_data.keys():
    print("\n\n##########################################################################")
    print(f"Processing database: {Fore.YELLOW}{database_name}{Style.RESET_ALL}")

    mgf_file = input_data[database_name]["mgf_file"]
    smiles_field = input_data[database_name]["smiles_field"]
    name_field = input_data[database_name]["name_field"]
    sf_field = input_data[database_name]["sf_field"]
    filter_fn = input_data[database_name]["filter"]

    found_results, spectra, gen_files = process_database(database_name, mgf_file, smiles_field, name_field, sf_field, checks, standardize_block_functions, out_path, filter_fn=filter_fn, verbose=include_details)
    generated_files.extend(gen_files)

print(f"\nGenerated files:")
for file in generated_files:
    print(f"   - {file}")

if output_folder_existed:
    print(f"\n\033[91mOutput folder {out_path} already existed. Existing files have not been deleted, exercise with caution.\033[0m")
