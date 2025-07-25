import os
import re

from AnnoMe.Filters import parse_mgf_file, draw_smiles
import pandas as pd
from rdkit import Chem
import numpy as np
import matplotlib.pyplot as plt

import matchms
import matchms.similarity as mssim

import tqdm
from PIL import Image
import tempfile
from PIL import ImageDraw, ImageFont

import networkx as nx
import seaborn as sns

os.makedirs("./forPublication/comparison_libraries", exist_ok=True)


# Find all mgf files with prenylated flavonoids or chalcones and get their unique smiles
libraries_pren_compounds = set()
spectra = []

# Iterate through the folder and find matching files
proc_folder = "./resources/libraries_filtered/"
for typ, suffix in {"train - relevant": "(.*)___prenyl_flavonoid_or_chalcone__MatchingSmiles\\.mgf"}.items():
    print(f"\nProcessing '{typ}' datasets in '{proc_folder}' with suffix '{suffix}'")
    for file_name in os.listdir(proc_folder):
        if not file_name.startswith("BOKU_iBAM"):
            match = re.match(suffix, file_name)
            if match:
                print("   - Checking file:", file_name)
                cmpds = parse_mgf_file(os.path.join(proc_folder, file_name))

                smiles = set()
                for cmpd in cmpds:
                    if "smiles" in cmpd and cmpd["smiles"]:
                        smiles.add(cmpd["smiles"])
                        cmpd["source"] = "MSnLib"
                        spectra.append(cmpd)
                print(f"      - Found {len(smiles)} unique smiles in {file_name}")
                libraries_pren_compounds.update(smiles)
print(f"Found {len(libraries_pren_compounds)} unique prenylated flavonoids or chalcones in MSnLib")
print("\n\n")


std_pren_compounds = set()
std_spectra = []

prenyl_flavonoids_path = "./resources/libraries_other/prenyl_flavonoids.tsv"
df = pd.read_csv(prenyl_flavonoids_path, sep="\t")
if "smiles" in df.columns:
    std_pren_compounds.update(df["smiles"].dropna().unique())
print(f"\nFound {len(std_pren_compounds)} unique prenylated flavonoids or chalcones in standard datasets")
print("\n\n")

files = [
    "./resources/libraries_other/HCD_pos__sirius.mgf",
    "./resources/libraries_other/HCD_neg__sirius.mgf",
    "./resources/libraries_filtered/BOKU_iBAM___prenyl_flavonoid_or_chalcone__MatchingSmiles.mgf",
]
for file in files:
    print(f"Processing in-house file: {file}")
    cmpds = parse_mgf_file(file)
    for cmpd in cmpds:
        cmpd["source"] = "in-house"
        spectra.append(cmpd)


if True:

    def cosine_score(a, b):
        similarity_cosine = mssim.CosineGreedy(tolerance=50).pair(a, b)
        return similarity_cosine["score"], similarity_cosine["matches"]

    all_properties = {"source": set(), "ionmode": set(), "fragmentation_method": set(), "collision_energy": set()}
    for cmpd in spectra:
        all_properties["source"].add(cmpd.get("source", "unknown"))
        all_properties["ionmode"].add(cmpd.get("ionmode", "unknown"))
        all_properties["fragmentation_method"].add(cmpd.get("fragmentation_method", "unknown"))
        all_properties["collision_energy"].add(cmpd.get("collision_energy", "unknown"))

    print("Unique properties found in spectra:")
    for prop, values in all_properties.items():
        print(f"  - {prop}: {', '.join(sorted(values))}")

    for filt_ionmode, filt_fragmentation_method, filt_collision_energy in [
        ("-", "hcd", ["[45.0]", "stepped20,45,70eV(absolute)"]),
        ("+", "hcd", ["[45.0]", "stepped20,45,70eV(absolute)"]),
        ("-", "hcd", ["[20.0]"]),
        ("+", "hcd", ["[20.0]"]),
        ("-", "hcd", ["[30.0]"]),
        ("+", "hcd", ["[30.0]"]),
        ("-", "hcd", ["[60.0]"]),
        ("+", "hcd", ["[60.0]"]),
    ]:
        print(f"\nProcessing spectra with ionmode='{filt_ionmode}', fragmentation_method='{filt_fragmentation_method}', collision_energy={filt_collision_energy}")
        # Extract spectrumData from all spectra
        spectra_data = []
        labels = []
        for cmpd in spectra:
            if (
                "$$spectrumData" in cmpd
                and cmpd["$$spectrumData"]
                and cmpd.get("ionmode", "unknown") == filt_ionmode
                and cmpd.get("fragmentation_method", "unknown") == filt_fragmentation_method
                and cmpd.get("collision_energy", "unknown") in filt_collision_energy
            ):
                # Flatten spectrumData if it's a list of tuples (mz, intensity)
                data = cmpd["$$spectrumData"]
                mzs, ints = [float(f) for f in data[0]], [float(f) for f in data[1]]
                precursor_mz = cmpd.get("precursorMz", None)
                spectra_data.append(matchms.Spectrum(np.array(mzs), np.array(ints), metadata=cmpd))
                labels.append(cmpd.get("source", "unknown"))

        n = len(spectra_data)
        cosine_matrix = np.zeros((n, n), dtype=float)
        for i in tqdm.tqdm(range(n)):
            for j in range(n):
                score, matched_peaks = cosine_score(spectra_data[i], spectra_data[j])
                cosine_matrix[i, j] = score

        # Clustered heatmap using seaborn clustermap
        sns.set(font_scale=0.7)
        # Assign a color to each unique source
        unique_sources = sorted(set(labels))
        palette = sns.color_palette("Set2", len(unique_sources))
        source_color_dict = {src: palette[i] for i, src in enumerate(unique_sources)}
        row_colors = [source_color_dict[label] for label in labels]
        col_colors = [source_color_dict[label] for label in labels]

        g = sns.clustermap(
            cosine_matrix, row_cluster=True, col_cluster=True, cmap="viridis", linewidths=0.0, figsize=(12, 10), xticklabels=labels, yticklabels=labels, row_colors=row_colors, col_colors=col_colors
        )
        # Add a legend for the source colors
        for src, color in source_color_dict.items():
            g.ax_col_dendrogram.bar(0, 0, color=color, label=src, linewidth=0)
        g.ax_col_dendrogram.legend(loc="center", ncol=1, title="Source", bbox_to_anchor=(1.1, 0.5))
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.cax.set_title("Cosine Score")
        plt.savefig(f"./forPublication/comparison_libraries/spectra_cosine_clustermap {filt_ionmode}_{filt_fragmentation_method}_{filt_collision_energy}.png", dpi=300)
        plt.close()


if True:
    # Compare structures: find overlap by sum formula
    def get_sum_formula(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.rdMolDescriptors.CalcMolFormula(mol)
        return None

    def process_smiles(smiles):
        """Process SMILES to remove non-standard characters and return a canonical form."""
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(smiles)
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol, canonical=True)

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

    libraries_smiles_set = set(process_smiles(smi) for smi in libraries_pren_compounds)
    std_smiles_set = set(process_smiles(smi) for smi in std_pren_compounds)

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
            combined_img.save(f"./forPublication/comparison_libraries/comparisonCanonicalSmiles_{found_string}_{sf}.png")
    print(f"Saved combined image to ./forPublication/comparison_libraries/comparisonCanonicalSmiles_*.png")
