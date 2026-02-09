
# Classification of MSMS spectra as Prenylated Flavonoids (relevant) or not (other)

1. Make sure that a copy of the AnnoMe repository has been downloaded. Please refer to the README of AnnoMe for Instructions. 

2. Start a command propmpt and navigate to the AnnoMe directory.

3. Verify that all public MSMS repositories have been downloaded or download them. The following command will check and download missing resources automatically: 
```{bash}
uv run annome_downloadresources
```

4. Start the substructure filtering GUI with the command
```{bash}
uv run annome_filtergui
```

5. Click the "Load MGF File(s)" button in the first step of the GUI. Then select and load the following 24 files from the folder `resources/libraries`:
- 20241003_enamdisc_neg_ms2.mgf
- 20241003_enamdisc_pos_ms2.mgf
- 20241003_enammol_neg_ms2.mgf
- 20241003_enammol_pos_ms2.mgf
- 20241003_mcebio_neg_ms2.mgf
- 20241003_mcebio_pos_ms2.mgf
- 20241003_mcedrug_neg_ms2.mgf
- 20241003_mcedrug_pos_ms2.mgf
- 20241003_mcescaf_neg_ms2.mgf
- 20241003_mcescaf_pos_ms2.mgf
- 20241003_nihnp_neg_ms2.mgf
- 20241003_nihnp_pos_ms2.mgf
- 20241003_otavapep_neg_ms2.mgf
- 20241003_otavapep_pos_ms2.mgf
- 20250228_mcediv_50k_sub_neg_ms2.mgf
- 20250228_mcediv_50k_sub_pos_ms2.mgf
- 20250228_targetmolnphts_np_neg_ms2.mgf
- 20250228_targetmolnphts_pos_ms2.mgf
- BOKU_iBAM.mgf
- MONA.mgf
- MassBank_RIKEN.mgf
- MassSpecGym.mgf
- WINE-DB-ORBITRAP.mgf
- gnps_cleaned.mgf

6. Activate the second step by clicking the tab on the left side of the window. Click the button "Apply Canonicalization". It might take a while to calculate all canonical SMILES strings for the 24 loaded files.

7. Activate the third step by clicking the respective tab on the left side of the window. This will allow filtering all loaded MS/MS spectra by substructure matches using SMARTS strings. The necessary filters can be loaded with the button "Load Filters from JSON". Select the file "demo/SMARTSFilterStrings_PrenylatedCompounds.json". Loading and applying all filters will take some time. 

8. After the filters have been loaded successfully, the table will show the number of matches and mismatches. 

9. Activate the fourth and final step to export the matching and mismatching spectra to new MGF files. Click the button "" and select the folder "resources/libraries_filtered".

10. Close the filtering GUI. 

