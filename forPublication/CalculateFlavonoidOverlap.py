ex_1 = ["resources/libraries_filtered/flavonoids_compounds/BOKU_iBAM___table.xlsx"]
ex_2 = [
    "resources/libraries_filtered/flavonoids_compounds/MSnLib_20241003_enamdisc_neg_ms2___table.xlsx",
    "resources/libraries_filtered/flavonoids_compounds/MSnLib_20241003_enamdisc_pos_ms2___table.xlsx",
    "resources/libraries_filtered/flavonoids_compounds/MSnLib_20241003_enammol_neg_ms2___table.xlsx",
    "resources/libraries_filtered/flavonoids_compounds/MSnLib_20241003_enammol_pos_ms2___table.xlsx",
    "resources/libraries_filtered/flavonoids_compounds/MSnLib_20241003_mcebio_neg_ms2___table.xlsx",
    "resources/libraries_filtered/flavonoids_compounds/MSnLib_20241003_mcebio_pos_ms2___table.xlsx",
    "resources/libraries_filtered/flavonoids_compounds/MSnLib_20241003_mcedrug_neg_ms2___table.xlsx",
    "resources/libraries_filtered/flavonoids_compounds/MSnLib_20241003_mcedrug_pos_ms2___table.xlsx",
    "resources/libraries_filtered/flavonoids_compounds/MSnLib_20241003_mcescaf_neg_ms2___table.xlsx",
    "resources/libraries_filtered/flavonoids_compounds/MSnLib_20241003_mcescaf_pos_ms2___table.xlsx",
    "resources/libraries_filtered/flavonoids_compounds/MSnLib_20241003_nihnp_neg_ms2___table.xlsx",
    "resources/libraries_filtered/flavonoids_compounds/MSnLib_20241003_nihnp_pos_ms2___table.xlsx",
    "resources/libraries_filtered/flavonoids_compounds/MSnLib_20241003_otavapep_neg_ms2___table.xlsx",
    "resources/libraries_filtered/flavonoids_compounds/MSnLib_20241003_otavapep_pos_ms2___table.xlsx",
    "resources/libraries_filtered/flavonoids_compounds/MSnLib_20250228_mcediv_50k_sub_neg_ms2___table.xlsx",
    "resources/libraries_filtered/flavonoids_compounds/MSnLib_20250228_mcediv_50k_sub_pos_ms2___table.xlsx",
    "resources/libraries_filtered/flavonoids_compounds/MSnLib_20250228_targetmolnphts_np_neg_ms2___table.xlsx",
    "resources/libraries_filtered/flavonoids_compounds/MSnLib_20250228_targetmolnphts_pos_ms2___table.xlsx",
]


import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
import networkx as nx
from PIL import Image, ImageDraw, ImageFont
import tempfile
from AnnoMe.Filters import draw_smiles
import re
import os


def is_int(x):
    """Check if a value can be converted to an integer."""
    try:
        int(x)
        return True
    except ValueError:
        return False


def proc_df(df):
    try:
        df = df["A_uniqueSmiles"].to_list()
        all_smiles = set()
        for smiles in df:
            if isinstance(smiles, str):
                if smiles.startswith("{") and smiles.endswith("}"):
                    # Remove curly braces and split by comma
                    items = [item.strip().strip("'") for item in smiles[1:-1].split(",")]
                    if len(items) == 1:
                        all_smiles.add(items[0].strip("'"))
                    elif is_int(items[0]):
                        print(smiles)
                        for it in items[1:]:
                            if it:
                                all_smiles.add(it.strip("'"))
                else:
                    match = re.match(r"^\d+: \{(.+)\}$", smiles)
                    if match:
                        # Extract the content inside the curly braces after the integer prefix
                        items = [item.strip().strip("'") for item in match.group(1).split(",")]
                        for it in items:
                            if it:
                                all_smiles.add(it.strip("'"))
                    else:
                        raise ValueError(f"Unexpected smiles format: {smiles}")

        return all_smiles
    except Exception as e:
        print(f"Error processing DataFrame: {e}")
        raise e


# Read and extract "A_uniqueSmiles" from all files in ex_1
set1 = set()
for file in ex_1:
    df = pd.read_excel(file)
    if "A_uniqueSmiles" in df.columns and "C_flavonoids" in df.columns:
        df_selected = df[["A_uniqueSmiles"]].dropna()
        df_selected = df_selected[df["C_flavonoids"].str.contains("substructure match", na=False)]
        df_selected = proc_df(df_selected)
        set1.update(df_selected)

# Read and extract "A_uniqueSmiles" from all files in ex_2
set2 = set()
for file in ex_2:
    df = pd.read_excel(file)
    if "A_uniqueSmiles" in df.columns and "C_flavonoids" in df.columns:
        df_selected = df[["A_uniqueSmiles"]].dropna()
        df_selected = df_selected[df["C_flavonoids"].str.contains("substructure match", na=False)]
        df_selected = proc_df(df_selected)
        set2.update(df_selected)

# Show a Venn diagram of set1 and set2 as print outputs
print("Venn diagram (counts):")
print("")
print(f"Set 1: {len(set1)}")
print(f"Set 2: {len(set2)}")
print("")
print(f"  Only in set1: {len(set1 - set2)}")
print(f"  Only in set2: {len(set2 - set1)}")
print(f"  In both: {len(set1 & set2)}")
print(f"  Total unique: {len(set1 | set2)}")


# Compare structures: find overlap by sum formula
def get_sum_formula(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.rdMolDescriptors.CalcMolFormula(mol)
    return None


def process_smiles(smiles):
    """Process SMILES to remove non-standard characters and return a canonical form."""
    try:
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(smiles)
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        raise e


def convert_ipython_image_to_PIL(image):
    """Convert an IPython image to a PIL image."""
    with tempfile.NamedTemporaryFile(suffix=".png", mode="wb", delete=False, delete_on_close=False) as tmp:
        tmp.write(image.data)
        tmp.close()
    pil_img = Image.open(tmp.name)
    # TODO clean temp file
    return pil_img


def topology_from_rdkit(rdkit_molecule):
    """
    Converts an RDKit molecule to a networkx graph.

    Args:
        rdkit_molecule: An RDKit molecule object.

    Returns:
        A networkx Graph object representing the molecule's topology.
    """
    mol = Chem.MolFromSmiles(rdkit_molecule)
    mol = Chem.RemoveHs(mol)
    topology = nx.Graph()
    for atom in mol.GetAtoms():
        topology.add_node(atom.GetIdx())
        for bonded in atom.GetNeighbors():
            topology.add_edge(atom.GetIdx(), bonded.GetIdx())
    return topology


def compare_mol_sets(set1, set2):
    """
    Compare two sets of SMILES strings and return the overlap and unique elements.

    Args:
        set1: A set of SMILES strings.
        set2: Another set of SMILES strings.

    Returns:
        A tuple containing:
            - Overlap: Set of SMILES that are in both sets.
            - Only in set1: Set of SMILES that are only in set1.
            - Only in set2: Set of SMILES that are only in set2.
    """
    where = [[], [], []]
    for smi1 in set1:
        top1 = topology_from_rdkit(smi1)
        found_in_set2 = False
        for smi2 in set2:
            top2 = topology_from_rdkit(smi2)
            if nx.is_isomorphic(top1, top2):
                found_in_set2 = True
                break
        if found_in_set2:
            where[0].append(smi1)
        else:
            where[1].append(smi1)

    for smi2 in set2:
        top2 = topology_from_rdkit(smi2)
        found_in_set1 = False
        for smi1 in set1:
            top1 = topology_from_rdkit(smi1)
            if nx.is_isomorphic(top1, top2):
                found_in_set1 = True
                break
        if not found_in_set1:
            where[2].append(smi2)

    return where[0], where[1], where[2]


libraries_smiles_set = set(process_smiles(smi) for smi in set2)
std_smiles_set = set(process_smiles(smi) for smi in set1)

libraries_sf_smiles = {"all": libraries_smiles_set}
std_sf_smiles = {"all": std_smiles_set}

for smi in libraries_smiles_set:
    sf = get_sum_formula(smi)
    if sf:
        if sf not in libraries_sf_smiles:
            libraries_sf_smiles[sf] = set()
        libraries_sf_smiles[sf].add(smi)

for smi in std_smiles_set:
    sf = get_sum_formula(smi)
    if sf:
        if sf not in std_sf_smiles:
            std_sf_smiles[sf] = set()
        std_sf_smiles[sf].add(smi)

all_sumformulas = list(set(libraries_sf_smiles.keys()) | set(std_sf_smiles.keys())) + ["all"]
for sf in all_sumformulas:
    # Compare structures: find overlap by exact match of canonical SMILES
    overlap_smiles = libraries_sf_smiles.get(sf, set()) & std_sf_smiles.get(sf, set())
    only_in_libraries = libraries_sf_smiles.get(sf, set()) - std_sf_smiles.get(sf, set())
    only_in_std = std_sf_smiles.get(sf, set()) - libraries_sf_smiles.get(sf, set())

    if False:
        # Compare molecule graphs: find overlap by isomorphism
        where = compare_mol_sets(libraries_sf_smiles.get(sf, set()), std_sf_smiles.get(sf, set()))
        overlap_smiles = set(where[0])
        only_in_libraries = set(where[1])
        only_in_std = set(where[2])

    # Print overlap and draw it
    print(f"Number of SMILES with {sf}")
    images = []
    found_string = ""
    for set_type, smiles in {
        "overlap": overlap_smiles,
        "only in libraries": only_in_libraries,
        "only in standard": only_in_std,
    }.items():
        print(f"   - {set_type}: {len(smiles)}")
        if smiles:
            smiles = sorted(smiles)
            l = list(smiles)
            img = draw_smiles(l, legends=l, max_draw=1000)
            images.append((set_type, img))
            found_string += "1"
        else:
            found_string += "0"
    if images:
        setnames = [f"{set_type}" for set_type, _ in images]
        pil_images = [convert_ipython_image_to_PIL(img) for _, img in images]
        widths, heights = zip(*(im.size for im in pil_images))
        total_height = 60 + sum(heights) + 50 * len(heights)
        max_width = max(widths)
        combined_img = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))

        # Draw the sum formula at the top
        y_offset = 0
        # Create a title label with the sum formula
        title_height = 60
        title_img = Image.new("RGB", (max_width, title_height), color=(255, 255, 255))
        draw_title = ImageDraw.Draw(title_img)
        try:
            title_font = ImageFont.truetype("arial.ttf", 45)
        except Exception:
            title_font = ImageFont.load_default()
        draw_title.text((10, 10), f"{sf}", fill=(0, 0, 0), font=title_font)
        combined_img.paste(title_img, (0, y_offset))
        y_offset += title_height

        # Paste each image with its label
        for setname, im in zip(setnames, pil_images):
            # Create a label image for the setname
            label_height = 50
            label_img = Image.new("RGB", (im.size[0], label_height), color=(255, 255, 255))
            draw = ImageDraw.Draw(label_img)
            try:
                font = ImageFont.truetype("arial.ttf", 40)
            except Exception:
                font = ImageFont.load_default()
            draw.text((10, 5), setname, fill=(0, 0, 0), font=font)

            # Paste the label above the image
            combined_img.paste(label_img, (0, y_offset))
            y_offset += label_height
            combined_img.paste(im, (0, y_offset))
            y_offset += im.size[1]

        # Save the combined image
        if not os.path.exists("./forPublication/comparison_libraries_Flavonoids"):
            os.makedirs("./forPublication/comparison_libraries_Flavonoids")
        combined_img.save(f"./forPublication/comparison_libraries_Flavonoids/comparisonCanonicalSmiles_{found_string}_{sf}.png")
print(f"Saved combined image to ./forPublication/comparison_libraries/comparisonCanonicalSmiles_*.png")
