
# Classification of MSMS spectra as Prenylated Flavonoids (relevant) or not (other)

Note: Make sure that your PC has enough main memory (tested with at least 16GB) and approximately 20GB of free disk space available. 

1. Make sure that a copy of the AnnoMe repository has been downloaded. Please refer to the README of AnnoMe for Instructions. 

2. Start a command propmpt and navigate to the AnnoMe directory.

## Download public MSMS repositories

3. Verify that all public MSMS repositories have been downloaded or download them. The following command will check and download missing resources automatically: 
Note: Make sure to check that you are not on a metered connection. The total size of all files to be downloaded is approximately 11GB, so this step might take some time depending on the available internet connection. 
```{bash}
uv run annome_downloadresources
```

## Filter MSMS spectra for Prenylated Flavonoids

4. Start the substructure filtering GUI with the command
```{bash}
uv run annome_filtergui
```

5. Click the "Load MGF File(s)" button in the first step of the GUI. Then select and load the following 19 files (all from MSnLib and one from BOKU) from the folder `resources/libraries`:
Note: Loading the files might take a couple of minutes (approximately 10 minutes on a standard laptop).
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

6. After the files have been loaded successfully, activate the second step by clicking the tab on the left side of the window. Click the button "Apply Canonicalization". It might take a while to calculate all canonical SMILES strings for the 24 loaded files.

7. Next, activate the third step by clicking the respective tab on the left side of the window. This will allow filtering all loaded MS/MS spectra by substructure matches using SMARTS strings. The necessary filters can be loaded with the button "Load Filters from JSON". Select the file "demo/GUI_PrenylatedCompounds_SMARTSFilterStrings.json" from the AnnoMe main folder. Loading and applying all filters will take some time (approximately 1 hour on a standard laptop). After the filters have been loaded successfully, the table will show the number of matches and mismatches for each filter. 

9. Then, activate the fourth and final step to export the matching and mismatching spectra to new MGF files. Click the button "Browse" and specify the new file-name "filtered" in the folder "resources/libraries_filtered_gui". Close the new file dialog and click the button "Export Filtered MGF Files to start the process. After a couple of minutes several files will have been generated. If there are MSMS spectra for a particular filter, it will be exported and named "filtered_<filter-name>_matched.mgf". Furthermore, a summary mgf file with all MSMS spectra matching at least one filter will also be generated, as well as a file with all MSMS spectra matching none of the provided filters. 

10. Close the filtering GUI. 

## Classification task

11. Start the classification filtering GUI with the command
```{bash}
uv run annome_classificationgui
```

12. Next, load the mgf files. Click the button "" and select the file "demo/gui_PrenylatedCompounds_classificationFiles.json". 

13. After loading has finished, goto the second step of the classification via a click on the tab "2. Generate Embeddings". Then click the button "Generate Embeddings" to load the mgf files and calculate the MS2DeepScore embeddings. Note: this step can take a significant amount of time, but results are cached after the first run. 

14. The next step is to define different subsets. 

15. Click on the tab of the fourth tab. This allows to define different classification models to be used. Each meta-data subset from the previous step will be used for each model in this step, makeing a total of n x m (n.. number of subsets, m.. number of classifiers to train) training and predicting instances. Alternatively, use the button 'Load Configuration' to load a predefined set of classifiers. Then click the button 'Train and Classify' to start the training and prediction process. 

16. Next, the true/false positive/negative rates can be inspected. For this click the tab of the fifth step on the left side of the classification window. A table will show these results, aggregated by meta-data subset, classification configuration, input-file, and type class. Furthermore, results of individual metabolic features can be inspected in the sixth tab. 

17. All results of the classification and prediction tasks are also exported to mgf and table files to be found in the folder specified in step 2.