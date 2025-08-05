from AnnoMe.Filters import (
    process_database,
    prep_smarts_key,
    download_MSMS_libraries,
    download_MS2DeepScore_model,
)
from collections import OrderedDict
import pathlib
import re
from colorama import Fore, Style


# Download the common MS/MS libraries if they do not exist
download_MSMS_libraries()


## Parameters
# fmt: off
flavone_smart = prep_smarts_key("O=C@1@C@C(@O@C2@C@C@C@C@C@1@2)C@3@C@C@C@C@C3")
isoflavone_smart = prep_smarts_key("O=C@1@C@2@C@C@C@C@C@2@O@C@C@1C@3@C@C@C@C@C@3")

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

libraries_path = "./resources/libraries"
input_data = {
    "gnps_cleaned": {"mgf_file": f"{libraries_path}/gnps_cleaned.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
    "MassBank_RIKEN": {"mgf_file": f"{libraries_path}/MassBank_RIKEN.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
    "MassSpecGym": {"mgf_file": f"{libraries_path}/MassSpecGym.mgf", "smiles_field": "smiles", "name_field": "inchikey", "sf_field": "formula", "filter": filter_generic},
    "MONA": {"mgf_file": f"{libraries_path}/MONA.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
    "WINE-DB-ORBITRAP": {"mgf_file": f"{libraries_path}/WINE-DB-ORBITRAP.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},

    "MSnLib_20241003_enamdisc_neg_ms2": {"mgf_file": f"{libraries_path}/20241003_enamdisc_neg_ms2.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
    "MSnLib_20241003_enamdisc_pos_ms2": {"mgf_file": f"{libraries_path}/20241003_enamdisc_pos_ms2.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
    "MSnLib_20241003_enammol_neg_ms2": {"mgf_file": f"{libraries_path}/20241003_enammol_neg_ms2.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
    "MSnLib_20241003_enammol_pos_ms2": {"mgf_file": f"{libraries_path}/20241003_enammol_pos_ms2.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
    "MSnLib_20241003_mcebio_neg_ms2": {"mgf_file": f"{libraries_path}/20241003_mcebio_neg_ms2.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
    "MSnLib_20241003_mcebio_pos_ms2": {"mgf_file": f"{libraries_path}/20241003_mcebio_pos_ms2.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
    "MSnLib_20241003_mcedrug_neg_ms2": {"mgf_file": f"{libraries_path}/20241003_mcedrug_neg_ms2.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
    "MSnLib_20241003_mcedrug_pos_ms2": {"mgf_file": f"{libraries_path}/20241003_mcedrug_pos_ms2.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
    "MSnLib_20241003_mcescaf_neg_ms2": {"mgf_file": f"{libraries_path}/20241003_mcescaf_neg_ms2.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
    "MSnLib_20241003_mcescaf_pos_ms2": {"mgf_file": f"{libraries_path}/20241003_mcescaf_pos_ms2.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
    "MSnLib_20241003_nihnp_neg_ms2": {"mgf_file": f"{libraries_path}/20241003_nihnp_neg_ms2.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
    "MSnLib_20241003_nihnp_pos_ms2": {"mgf_file": f"{libraries_path}/20241003_nihnp_pos_ms2.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
    "MSnLib_20241003_otavapep_neg_ms2": {"mgf_file": f"{libraries_path}/20241003_otavapep_neg_ms2.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
    "MSnLib_20241003_otavapep_pos_ms2": {"mgf_file": f"{libraries_path}/20241003_otavapep_pos_ms2.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
    "MSnLib_20250228_mcediv_50k_sub_neg_ms2": {"mgf_file": f"{libraries_path}/20250228_mcediv_50k_sub_neg_ms2.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
    "MSnLib_20250228_mcediv_50k_sub_pos_ms2": {"mgf_file": f"{libraries_path}/20250228_mcediv_50k_sub_pos_ms2.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
    "MSnLib_20250228_targetmolnphts_np_neg_ms2": {"mgf_file": f"{libraries_path}/20250228_targetmolnphts_np_neg_ms2.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
    "MSnLib_20250228_targetmolnphts_pos_ms2": {"mgf_file": f"{libraries_path}/20250228_targetmolnphts_pos_ms2.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_generic},
}

# Visualize SMARTS with https://smarts.plus/view/1cf72609-6995-4b25-8a16-42eeeb8c09df
checks = OrderedDict([
    ("flavonoids", [[flavone_smart, isoflavone_smart]]),
])
include_details = False

out_path = f"./resources/libraries_filtered/flavonoids_compounds/"

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

    found_results, spectra, gen_files = process_database(
        database_name,
        mgf_file,
        smiles_field,
        name_field,
        sf_field,
        checks,
        standardize_block_functions,
        out_path,
        filter_fn=filter_fn,
        verbose=include_details,
    )
    generated_files.extend(gen_files)

print(f"\nGenerated files:")
for file in generated_files:
    print(f"   - {file}")

if output_folder_existed:
    print(f"\n\033[91mOutput folder {out_path} already existed. Existing files have not been deleted, exercise with caution.\033[0m")
