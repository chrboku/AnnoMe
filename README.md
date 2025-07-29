# AnnoMe
  
![AnnoMe Logo](logo.png)
 
This is a package for the classification of MS/MS spectra of novel compounds into either "relevant" or "other" compounds. In this respect, the classification result "relevant" refers to specific substance classes of interest to the user, while the classification result "other" refer to structures not of interest. For this, the classifiers first need to be trained on a large set of MS/MS spectra of compounds of interest (e.g., obtained from reference standards) and others (e.g., obtained from reference standards of other compounds and from large MS/MS spectra repositories).

## Limitations

Classification of MS/MS spectra into substance classes is a non-trivial task and most often the classification will not be successful, especially when the training dataset is highly imbalanced. The presented python package is not capable of resolving this issue. Therefore, the users are herewith informed to exercise caution with the generated results (i.e., the assignment of MS/MS spectra to specific compound classes of interest).

## Methodology

Using training data consisting of annotated and labeled MS/MS spectra of compounds of interest and others, different classifiers (LDA, NN, SVM, etc.) are first trained using cross-validation and different random seeds. The results are then aggregated and a majority-vote is derived which indicates the final classification of the input MS/MS spectra to either "relevant" or "other". The term "relevant" refers to compounds of interest, while the term "other" refers to compounds not of interest.

## Disclaimer

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Demonstration datasets

There are three demonstrations available in AnnoMe. These try to train classifiers for
- Flavonoid and isoflavonoid compounds
- Prenylated flavonoids and isoflavonoids and prenylated chalcones.

The ground truth datasets used for training and testing/validation are
- MSnLib spectra libraries
- an in-house database obtained at BOKU University
- a wheat ear extract obtained at BOKU University.

The inference dataset are obtained from extracts of fruits from
- Paulownia tomentosa
- Glycyrrhizza uralensis
plants, which are known to produce prenylated flavonoid compounds.

To execute the respective calculation, please execute the following steps:

- [optional step]: Install the python tool uv. Please refer to [https://github.com/astral-sh/uv#installation](https://github.com/astral-sh/uv#installation) for installation instruction. This step can be omitted if uv is already available on the system.
- Open a terminal and navigate to an empty folder into which the AnnoMe package shall be cloned.
- Clone the repo AnnoMe with the command `git clone https://github.com/chrboku/AnnoMe`.
- Navigate into the new folder with `cd AnnoMe`.
- From the base folder (i.e., **AnnoMe**, the current folder), execute the demo filter-script with `uv run ./demo/Filter_PrenylatedCompounds_publicDBs.py`. This will download the publicly available MS/MS databases and the used embeddings model. Then the scripts will filter the public databases for **prenylated flavones** and **prenylated chalcones**. The results will be available in **./resources/libraries_filtered**.
- Similar to the public database, execute the demo script for the in-house databases with `uv run ./demo/Filter_PrenylatedCompounds_BOKUDB.py`.
- From the base folder (i.e., **AnnoMe**, the current folder), execute the demo analysis-script with `uv run jupyter nbconvert --to HTML --execute ./demo/Classification_PrenylatedCompounds_publicDBs.ipynb`. Once finished, the log of the classification pipeline will be available in **./demo/Classification_PrenylatedCompounds_publicDBs.html** and the prediction results will be available in **./demo/publicDBs/output/PrenylatedCompounds_PublicDBs/**.
- Similar to the public database, execute the demo analysis-script with `uv run jupyter nbconvert --to HTML --execute ./demo/Classification_PrenylatedCompounds_BOKUDB.ipynb`. Once finished, the log of the classification pipeline will be available in **./demo/Classification_PrenylatedCompounds_BOKUDB.html** and the prediction results will be available in **./demo/publicDBs/output/PrenylatedCompounds_BOKU/**.
- Besides the HTML files of the Jupyter notebooks, a series of output files will be generated for each subset of the MS/MS spectra. These are located in **demo/output/dataset_name**, and the following files are present

| File Name | Description |
|------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| df_embeddings.pkl | Pickle file containing the meta-data and embeddings of the MS/MS spectra. |
| summary_tables.xlsx | Obtained results from all subsets on one dataset allowing for an easy comparison of the performance of the training, validation, and inference sets from the different subsets of the MS/MS spectra. |
| subset/dataset_overview.xlsx | Overview of number of MS/MS spectra for the training, validation, and inference sets used. |
| subset/inference_data.xlsx | Generated prediction results on the inference data. Sheet **long**: Prediction results for each MS/MS spectrum of the inference set; sheet **overall**: Summary of the prediction results per class and sample. |
| subset/training_data.xlsx | Generated cross-validation results on the training data. Sheet **long**: Prediction results for each MS/MS spectrum of the training cross-validation set; sheet **overall**: Summary of the prediction results per class and sample. |
| subset/validation_data.xlsx | Generated validation results on the validation data. Sheet **long**: Prediction results for each MS/MS spectrum of the validation set; sheet **overall**: Summary of the prediction results per class and sample. |

## Usage for new classifications

To use AnnoMe for the classification of a new relevant chemical class, either
- respective mgf files need to be available
- respective MS/MS spectra are filtered from public repositories.

### Filter public repositories

- Clone and rename the file **demo/Filter_PrenylatedCompounds_publicDBs.py** to a new directory and open it with an editor of your choice.

- Adapt the **input_data** variable, use all or some of the public respoitories already available there, or add new mgf files to this directory. The following fields are important and necessary:

-  **key**: name of the dataset, must be unique

- value: dict of
-  **mgf_file**: path to the mgf file to import
-  **smiles_field**: name of the field for the smiles code present in each of the MS/MS spectra of that new mgf file
-  **name_field**: name of the field for the name field of the compound present in each of the MS/MS spectra of that new mgf file
-  **sf_field**: name of the field for the chemical formula field of the compound present in each of the MS/MS spectra of that new mgf file
-  **filter**: user-provided function to filter the MS/MS spectra in the mgf file, must return an array of booleans that indicate if the i-th spectrum of the mgf file should be used or not
- Adapt the **checks** variable. It is a dictionary and each entry specifies one subset that will be filtered. For example, the subset **flavonoid** in the demo file is used to filter for MS/MS spectra which are obtained from flaovnoid compounds. The respective SMARTs filters are the values of the entry. These must be a list of smart filters or another list. Each entry in the list is concatenated via AND, while each filter in a sublist is concatenated via OR. Thereby, more complex filters may be logically concatenated. For example, the value **[prenylgroup, [flavonoid, isoflavonoid]]** will fit all prenylated flavonoids or prenylated isoflavonoids, but will not filter all prenylated compounds or non prenylated flavonoids.
- Adapt the **out_path** variable, which stores where the filtered mgf files shall be saved to
- Optionally, adapt the variable **standardize_block_function**. It is a dictionary of fields that are reformatted using a provided list of functions. Each function receives the value of the respective name-field and can modify it (e.g., standardize the collision energies, rename complicated instrument names to a single string)
- Execute the filtering script with ``uv run script_location_and_name.py``

### Execute the classifier

- Clone and rename the file **demo/Classification_PrenylatedCompounds_publicDBs.ipynb** to the same new directory as above and open it with an editor of your choice.
- Adapt the parameters In the second cell with the title **Parameters**.
-  **output_dir**: Specify where the results shall be saved to.
-  **datasets**: adapt, each entry in the list must be a dictionary specifying a source for MS/MS spectra. the following information must be provided or is optional for each such source:
-  **name**: the name of the source, should be alphanumeric and contain no special characters
-  **type**: specify the type of the source, and will define it a source is used for training and which class this source has (i.e., **train - relevant** or **train - other**), if a source is used for validation and which class it has (i.e., **validation - relevant** or **validation - other**), or if the classes of the MS/MS spectra shall be predicted (i.e., **inference**).
-  **file** specifies where the mgf file is located.
-  **fragmentation_metod** specifies the key which extracts the fragmentation method from the source
-  **colour** colour of the dataset used for printing the embeddings with UMAP or PacMAP
-  **randomly_sample** specifies the maximum number of randomly picked MS/MS spectra for this file, omit to use all.
-  **data_to_add**: this dictionary specifies fields from the MS/MS spectra that will be extracted and added as meta-information to the output. Furthermore, it allows specifying which fields in the MS/MS datasources indicate the same information. For example, the entry **("precursor_mz", ["pepmass", "precursor_mz"])** specifies that the field precursor_mz can be obtained from the fields **pepmass** or **precursor_mz**.
-  **training_subsets**: this dictionary allows to sample from the loaded sources. While the key specifies a name, which should be alphanumeric and not contain any special characters, as it will also be used for generating a folder-name, the value should be a user-provided function that select from the loaded MS/MS spectra a subset (e.g., all MS/MS spectra of a certain polarity, all obtained with a specific fragmentation setup).
-  **colors**: colors for each dataset, can be automatically obtained from the **datasets** variable
- Run the training and classification pipeline using the command ``uv run jupyter nbconvert --to HTML --execute script_location_and_name.ipynb``
