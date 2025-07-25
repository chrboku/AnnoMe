from AnnoMe.Filters import process_database, prep_smarts_key, CE_parser
from AnnoMe.Filters import (
    process_database,
    prep_smarts_key,
    download_MSMS_libraries,
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
chalcone1_smart = prep_smarts_key("[O,o]=[C,c](-[CH2]-[CH2]-[C,c]@1@[C,c]@[C,c]@[C,c]@[C,c]@[C,c]@1)-[C,c]@2@[C,c]@[C,c]@[C,c]@[C,c]@[C,c]@2", replace = False)
chalcone2_smart = prep_smarts_key("[O,o]=[C,c](-[CH]=[CH]-[C,c]@1@[C,c]@[C,c]@[C,c]@[C,c]@[C,c]@1)-[C,c]@2@[C,c]@[C,c]@[C,c]@[C,c]@[C,c]@2", replace = False)
chromone_smart = prep_smarts_key("O=C@1@C@C@O@C@2@C@C@C@C@C@12")


def filter_boku(spectra):
    instrument_regex = re.compile(".*(orbitrap|q-exactive).*", re.IGNORECASE)
    instruments = set()
    for block in spectra:
        instruments.add(block["instrument"])
    print(f"Found instruments: {instruments}")
    return [block for block in spectra if "instrument" in block and instrument_regex.search(block["instrument"])]

libraries_path = "./resources/libraries"
input_data = {
    "BOKU_iBAM": {"mgf_file": f"{libraries_path}/BOKU_iBAM.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_boku},
}

# Visualize SMARTS with https://smarts.plus/view/1cf72609-6995-4b25-8a16-42eeeb8c09df
checks = OrderedDict([
    ("prenyl_flavonoid_or_chalcone",      {"filter": [[flavone_smart, isoflavone_smart, chalcone1_smart, chalcone2_smart], [prep_smarts_key(x) for x in ["CC=C(C)CCC=C(C)C", "C\\C=C(/C)\\CCC(O)C(=C)C", "CC(O)C(C)CCC=C(C)C", "CC=C(C)CCCC(C)(C)O", "CC(=CCCC1(C)OCCC=C1)C", "CC(=C)C(O)CCC1(C)OC=CCC1O", "C\\C=C(/C)\\CCCC(C)(O)CO", "CC(=CCCC(C)(O)C1CCCO1)C", "CC1(C)C(O)CCC2(C)OCCCC12", "CC1C(=CCC(O)C1(C)C)C", "CC1C(=C)CCC(O)C1(C)C", "COCC(C)(O)CCCC(=CC)C", "CC(=CCCC1(C)OC=CCC1O)C", "CC1(C)OCCC=C1", "CC1OC=CC1(C)C", "CC(CC=C(C)C)C(=C)C", "CCC(=C)C", "CC=C(C)C", "C-C(-C)-C=C", "CC(=C)C=C", "CC(=CCO)C", "CC(C)(O)C=C", "CCC(C)(C)O", "CC(O)C(=C)C", "CC1OC1(C)C", "C\\C=C(/C)\\CO", "CC(O)C(C)(C)O", "CC1(C)CCc2ccccc2O1", "CC1(C)CCCCO1"]]]}),

    ("flavonoid", {"filter": [[flavone_smart]]}),
    ("isoflavonoid", {"filter": [[isoflavone_smart]]}),
    ("stilbene", {"filter": [[prep_smarts_key("c1:c:c:[C,c](:c:c:1)[CH]=[CH][C,c]2:c:c:c:c:c:2", replace=False)]]}),
])
include_details = False

out_path = f"./resources/libraries_filtered/"

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
