from AnnoMe.Filters import process_database, prep_smarts_key, CE_parser
from collections import OrderedDict
import pathlib
import re
from colorama import Fore, Style


## Parameters
# fmt: off
flavone_smart = prep_smarts_key("O=C@1@C@C(@O@C2@C@C@C@C@C@1@2)C@3@C@C@C@C@C3")
isoflavone_smart = prep_smarts_key("O=C@1@C@2@C@C@C@C@C@2@O@C@C@1C@3@C@C@C@C@C@3")
chalcone1_smart = prep_smarts_key("[O,o]=[C,c](-[CH2]-[CH2]-[C,c]@1@[C,c]@[C,c]@[C,c]@[C,c]@[C,c]@1)-[C,c]@2@[C,c]@[C,c]@[C,c]@[C,c]@[C,c]@2", replace = False)
chalcone2_smart = prep_smarts_key("[O,o]=[C,c](-[CH]=[CH]-[C,c]@1@[C,c]@[C,c]@[C,c]@[C,c]@[C,c]@1)-[C,c]@2@[C,c]@[C,c]@[C,c]@[C,c]@[C,c]@2", replace = False)
chromone_smart = prep_smarts_key("O=C@1@C@C@O@C@2@C@C@C@C@C@12")


def filter_boku(spectra):
    instrument_regex = re.compile(".*(orbitrap|q-exactive).*", re.IGNORECASE)
    return [block for block in spectra if "instrument" in block and instrument_regex.search(block["instrument"])]

def filter_riken(spectra): 
    instrument_regex = re.compile(".*(orbitrap|q exactive|q-exactive|qexactive|exploris).*", re.IGNORECASE)
    return [block for block in spectra if "instrument" in block and instrument_regex.search(block["instrument"]) and 
            "fragmentation_method" in block and block["fragmentation_method"].lower() in ["hcd", "cid"]]

def filter_mona(spectra):
    return [block for block in spectra if 
            "fragmentation_method" in block and block["fragmentation_method"].lower() in ["hcd", "cid"] and 
            "collision_energy" in block and 10 <= CE_parser(block["collision_energy"]) <= 90]

def filter_gnps_library(spectra):
    instrument_regex = re.compile(".*(orbitrap|q exactive|q-exactive|qexactive|exploris).*", re.IGNORECASE)
    return [block for block in spectra if 
            "name" in block and 
            "instrument" in block and instrument_regex.search(block["instrument"]) and 
            "collision_energy" in block and 10 <= CE_parser(block["collision_energy"]) <= 90 and 
            "fragmentation_method" in block and block["fragmentation_method"].lower() in ["hcd", "cid"]]

def filter_generic(spectra): 
    instrument_regex = re.compile(".*(orbitrap|q exactive|q-exactive|qexactive|exploris).*", re.IGNORECASE)
    return [block for block in spectra if 
            "instrument" in block and instrument_regex.search(block["instrument"]) and 
            "fragmentation_method" in block and block["fragmentation_method"].lower() in ["hcd", "cid"] and
            "collision_energy" in block and 10 <= CE_parser(block["collision_energy"]) <= 90 and 
            True]


data_folderer_path = "../../data"
input_data = {
    "BOKU_iBAM_MB": {"mgf_file": f"{data_folderer_path}/BOKU_iBAM_new.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_boku},
    #"MB_Riken":     {"mgf_file": f"{data_folderer_path}/MassBank_RIKEN.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_riken},
    "MONA":         {"mgf_file": f"{data_folderer_path}/MoNA-export-All_LC-MS-MS_Orbitrap.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_mona},
    "GNPS":         {"mgf_file": f"{data_folderer_path}/ALL_GNPS_cleaned.mgf", "smiles_field": "smiles", "name_field": "name", "sf_field": "formula", "filter": filter_gnps_library},
}

# Visualize SMARTS with https://smarts.plus/view/1cf72609-6995-4b25-8a16-42eeeb8c09df
checks = OrderedDict([
    #("chromones",                {"filter": chromone_smart}),
    #("flavone",                  {"filter": flavone_smart}),
    #("iso-flavone",              {"filter": isoflavone_smart}),
    #("chalcone",                 {"filter": chalcone1_smart}),
    #("dihydrochalcone",          {"filter": chalcone2_smart}),
    #
    #("geranyl_group",            {"filter": [prep_smarts_key("CC=C(C)CCC=C(C)C")]}),
    #("6-hydroxygeranyl",         {"filter": [prep_smarts_key("C\\C=C(/C)\\CCC(O)C(=C)C")]}),
    #("2-hydroxygeranyl",         {"filter": [prep_smarts_key("CC(O)C(C)CCC=C(C)C")]}),
    #("7-hydroxygeranyl",         {"filter": [prep_smarts_key("CC=C(C)CCCC(C)(C)O")]}),
    #("pyranogeranyl",            {"filter": [prep_smarts_key("CC(=CCCC1(C)OCCC=C1)C")]}),
    #("dihydroxypyranogeranyl",   {"filter": [prep_smarts_key("CC(=C)C(O)CCC1(C)OC=CCC1O")]}),
    #("dihydroxygeranyl",         {"filter": [prep_smarts_key("C\\C=C(/C)\\CCCC(C)(O)CO")]}),
    #("hydroxyfuranogeranyl",     {"filter": [prep_smarts_key("CC(=CCCC(C)(O)C1CCCO1)C")]}),
    #("hydroxygeranylxanthen",    {"filter": [prep_smarts_key("CC1(C)C(O)CCC2(C)OCCCC12")]}),
    #("hydroxycyclicgeranyl_1",   {"filter": [prep_smarts_key("CC1C(=CCC(O)C1(C)C)C")]}),
    #("hydroxycyclicgeranyl_2",   {"filter": [prep_smarts_key("CC1C(=C)CCC(O)C1(C)C")]}),
    #("hydroxymethoxygeranyl",    {"filter": [prep_smarts_key("COCC(C)(O)CCCC(=CC)C")]}),
    #("hydroxypyranogeranyl",     {"filter": [prep_smarts_key("CC(=CCCC1(C)OC=CCC1O)C")]}),
    #("pyranoprenyl",             {"filter": [prep_smarts_key("CC1(C)OCCC=C1")]}),
    #("furanoprenyl",             {"filter": [prep_smarts_key("CC1OC=CC1(C)C")]}),
    #("levandulyl",               {"filter": [prep_smarts_key("CC(CC=C(C)C)C(=C)C")]}),
    #("6-methylbutenyl",          {"filter": [prep_smarts_key("CCC(=C)C")]}),
    #
    #("prenyl_1",                 {"filter": [prep_smarts_key("CC=C(C)C")]}),
    ### The following prenyl group is excluded as it matches any flavone, needs hydrogen atoms to be specified
    ##("prenyl_2",                 {"filter": [prep_smarts_key("CCC(C)C")]}),
    #("prenyl_3",                 {"filter": [prep_smarts_key("C-C(-C)-C=C")]}),
    #("prenyl_4",                 {"filter": [prep_smarts_key("CC(=C)C=C")]}),
    #("prenyl_5",                 {"filter": [prep_smarts_key("CC(=CCO)C")]}),
    #("prenyl_6",                 {"filter": [prep_smarts_key("CC(C)(O)C=C")]}),
    #("prenyl_7",                 {"filter": [prep_smarts_key("CCC(C)(C)O")]}),
    #("prenyl_8",                 {"filter": [prep_smarts_key("CC(O)C(=C)C")]}),
    #("prenyl_9",                 {"filter": [prep_smarts_key("CC1OC1(C)C")]}),
    #("prenyl_10",                {"filter": [prep_smarts_key("C\\C=C(/C)\\CO")]}),
    ### The following prenyl group is excluded as it matches any flavone, needs hydrogen atoms to be specified
    ##("prenyl_11",                {"filter": [prep_smarts_key("CCC(C)CO")]}),
    #("prenyl_12",                {"filter": [prep_smarts_key("CC(O)C(C)(C)O")]}),
    #("prenyl_13",                {"filter": [prep_smarts_key("CC1(C)CCc2ccccc2O1")]}),
    #("prenyl_14",                {"filter": [prep_smarts_key("CC1(C)CCCCO1"),]}),
    #
    #("prenyl + flavone",         {"filter": [flavone_smart, [prep_smarts_key(x) for x in ["CC=C(C)CCC=C(C)C", "C\\C=C(/C)\\CCC(O)C(=C)C", "CC(O)C(C)CCC=C(C)C", "CC=C(C)CCCC(C)(C)O", "CC(=CCCC1(C)OCCC=C1)C", "CC(=C)C(O)CCC1(C)OC=CCC1O", "C\\C=C(/C)\\CCCC(C)(O)CO", "CC(=CCCC(C)(O)C1CCCO1)C", "CC1(C)C(O)CCC2(C)OCCCC12", "CC1C(=CCC(O)C1(C)C)C", "CC1C(=C)CCC(O)C1(C)C", "COCC(C)(O)CCCC(=CC)C", "CC(=CCCC1(C)OC=CCC1O)C", "CC1(C)OCCC=C1", "CC1OC=CC1(C)C", "CC(CC=C(C)C)C(=C)C", "CCC(=C)C", "CC=C(C)C", "C-C(-C)-C=C", "CC(=C)C=C", "CC(=CCO)C", "CC(C)(O)C=C", "CCC(C)(C)O", "CC(O)C(=C)C", "CC1OC1(C)C", "C\\C=C(/C)\\CO", "CC(O)C(C)(C)O", "CC1(C)CCc2ccccc2O1", "CC1(C)CCCCO1"]]]}),
    #("prenyl + iso-flavone",     {"filter": [isoflavone_smart, [prep_smarts_key(x) for x in ["CC=C(C)CCC=C(C)C", "C\\C=C(/C)\\CCC(O)C(=C)C", "CC(O)C(C)CCC=C(C)C", "CC=C(C)CCCC(C)(C)O", "CC(=CCCC1(C)OCCC=C1)C", "CC(=C)C(O)CCC1(C)OC=CCC1O", "C\\C=C(/C)\\CCCC(C)(O)CO", "CC(=CCCC(C)(O)C1CCCO1)C", "CC1(C)C(O)CCC2(C)OCCCC12", "CC1C(=CCC(O)C1(C)C)C", "CC1C(=C)CCC(O)C1(C)C", "COCC(C)(O)CCCC(=CC)C", "CC(=CCCC1(C)OC=CCC1O)C", "CC1(C)OCCC=C1", "CC1OC=CC1(C)C", "CC(CC=C(C)C)C(=C)C", "CCC(=C)C", "CC=C(C)C", "C-C(-C)-C=C", "CC(=C)C=C", "CC(=CCO)C", "CC(C)(O)C=C", "CCC(C)(C)O", "CC(O)C(=C)C", "CC1OC1(C)C", "C\\C=C(/C)\\CO", "CC(O)C(C)(C)O", "CC1(C)CCc2ccccc2O1", "CC1(C)CCCCO1"]]]}),
    #("prenyl + chalcone",        {"filter": [chalcone1_smart, [prep_smarts_key(x) for x in ["CC=C(C)CCC=C(C)C", "C\\C=C(/C)\\CCC(O)C(=C)C", "CC(O)C(C)CCC=C(C)C", "CC=C(C)CCCC(C)(C)O", "CC(=CCCC1(C)OCCC=C1)C", "CC(=C)C(O)CCC1(C)OC=CCC1O", "C\\C=C(/C)\\CCCC(C)(O)CO", "CC(=CCCC(C)(O)C1CCCO1)C", "CC1(C)C(O)CCC2(C)OCCCC12", "CC1C(=CCC(O)C1(C)C)C", "CC1C(=C)CCC(O)C1(C)C", "COCC(C)(O)CCCC(=CC)C", "CC(=CCCC1(C)OC=CCC1O)C", "CC1(C)OCCC=C1", "CC1OC=CC1(C)C", "CC(CC=C(C)C)C(=C)C", "CCC(=C)C", "CC=C(C)C", "C-C(-C)-C=C", "CC(=C)C=C", "CC(=CCO)C", "CC(C)(O)C=C", "CCC(C)(C)O", "CC(O)C(=C)C", "CC1OC1(C)C", "C\\C=C(/C)\\CO", "CC(O)C(C)(C)O", "CC1(C)CCc2ccccc2O1", "CC1(C)CCCCO1"]]]}),
    #("prenyl + dihydrochalcone", {"filter": [chalcone2_smart, [prep_smarts_key(x) for x in ["CC=C(C)CCC=C(C)C", "C\\C=C(/C)\\CCC(O)C(=C)C", "CC(O)C(C)CCC=C(C)C", "CC=C(C)CCCC(C)(C)O", "CC(=CCCC1(C)OCCC=C1)C", "CC(=C)C(O)CCC1(C)OC=CCC1O", "C\\C=C(/C)\\CCCC(C)(O)CO", "CC(=CCCC(C)(O)C1CCCO1)C", "CC1(C)C(O)CCC2(C)OCCCC12", "CC1C(=CCC(O)C1(C)C)C", "CC1C(=C)CCC(O)C1(C)C", "COCC(C)(O)CCCC(=CC)C", "CC(=CCCC1(C)OC=CCC1O)C", "CC1(C)OCCC=C1", "CC1OC=CC1(C)C", "CC(CC=C(C)C)C(=C)C", "CCC(=C)C", "CC=C(C)C", "C-C(-C)-C=C", "CC(=C)C=C", "CC(=CCO)C", "CC(C)(O)C=C", "CCC(C)(C)O", "CC(O)C(=C)C", "CC1OC1(C)C", "C\\C=C(/C)\\CO", "CC(O)C(C)(C)O", "CC1(C)CCc2ccccc2O1", "CC1(C)CCCCO1"]]]}),
    
    ("StructureOfInterest",      {"filter": [[flavone_smart, isoflavone_smart, chalcone1_smart, chalcone2_smart], [prep_smarts_key(x) for x in ["CC=C(C)CCC=C(C)C", "C\\C=C(/C)\\CCC(O)C(=C)C", "CC(O)C(C)CCC=C(C)C", "CC=C(C)CCCC(C)(C)O", "CC(=CCCC1(C)OCCC=C1)C", "CC(=C)C(O)CCC1(C)OC=CCC1O", "C\\C=C(/C)\\CCCC(C)(O)CO", "CC(=CCCC(C)(O)C1CCCO1)C", "CC1(C)C(O)CCC2(C)OCCCC12", "CC1C(=CCC(O)C1(C)C)C", "CC1C(=C)CCC(O)C1(C)C", "COCC(C)(O)CCCC(=CC)C", "CC(=CCCC1(C)OC=CCC1O)C", "CC1(C)OCCC=C1", "CC1OC=CC1(C)C", "CC(CC=C(C)C)C(=C)C", "CCC(=C)C", "CC=C(C)C", "C-C(-C)-C=C", "CC(=C)C=C", "CC(=CCO)C", "CC(C)(O)C=C", "CCC(C)(C)O", "CC(O)C(=C)C", "CC1OC1(C)C", "C\\C=C(/C)\\CO", "CC(O)C(C)(C)O", "CC1(C)CCc2ccccc2O1", "CC1(C)CCCCO1"]]]}),
])
include_details = False

out_path = f"{data_folderer_path}/derived/"

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
