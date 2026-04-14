# AnnoMe
  
![AnnoMe Logo](logo.png)
 
This is a package for the classification of MS/MS spectra of novel compounds into either "relevant" or "other" compounds. In this respect, the classification result "relevant" refers to specific substance classes of interest to the user, while the classification result "other" refer to structures not of interest. For this, the classifiers first need to be trained on a large set of MS/MS spectra of compounds of interest (e.g., obtained from reference standards) and others (e.g., obtained from reference standards of other compounds and from large MS/MS spectra repositories).

## Limitations

Classification of MS/MS spectra into substance classes is a non-trivial task and most often the classification will not be successful, especially when the training dataset is highly imbalanced. The presented python package is not capable of resolving this issue. Therefore, the users are herewith informed to exercise caution with the generated results (i.e., the assignment of MS/MS spectra to specific compound classes of interest).

## Methodology

Using training data consisting of annotated and labeled MS/MS spectra of compounds of interest and others, different classifiers (LDA, NN, SVM, etc.) are first trained using cross-validation and different random seeds. The results are then aggregated and a majority-vote is derived which indicates the final classification of the input MS/MS spectra to either "relevant" or "other". The term "relevant" refers to compounds of interest, while the term "other" refers to compounds not of interest.

## Setup

1. Install the uv toolkit. For installation instructions please refer to [https://github.com/astral-sh/uv#installation](https://github.com/astral-sh/uv#installation).

2. Obtain a copy of AnnoMe. This can be done via `git` or a versioned release of AnnoMe. 

Git: 
- Install `git`. For installation instructions please refer to [https://github.com/git-guides/install-git](https://github.com/git-guides/install-git). 
- Open a command prompt and navigate to a new folder of your choice where you want to setup the AnnoMe package.
- In the command prompt, clone the AnnoMe repository with the following commands:
```{bash}
git clone https://github.com/chrboku/AnnoMe
cd AnnoMe
```

Versioned release of AnnoMe: 
- Download the latest release from [https://github.com/chrboku/AnnoMe/releases](https://github.com/chrboku/AnnoMe/releases).
- Unpack the archive to a folder of your choice. 

3. Download available public MS/MS resources using the command `uv run annome_downloadresources` or double-click the file `annome_downloadresources.bat` (Windows) or `annome_downloadresources.sh` (Linux, Mac).
Note: All resources are approximately 11GB in size (per 2.2026), and - depending on your internet connection - the download might take some time. It can be interrupted and restarted. Also please be aware of the download size if you are on a metered internet connection

4. Start the filtering GUI via the command `uv run annome_filtergui` or double-click the file `annome_filtergui.bat` (Windows) or `annome_filtergui.sh` (Linux, Mac). 

5. Start the classification GUI via the command `uv run annome_classificationgui` or double-click the file `annome_classificationgui.bat` (Windows) or `annome_classificationgui.sh` (Linux, Mac).


## Graphical User Interface
AnnoMe provides two graphical user interface programs for a) the filtering of MGF files for interesting structures, and b) performing classification and comparison tasks. The two tools can be executed with the commands:
```bash
uv run annome_filtergui
uv run annome_classificationgui
```

## Computing Resources
Depending on the number of MSMS spectra to be processed, more computing resources are required. The limiting step is the generation of the MS2DeepScore embeddings, which took 4 hours on a standard laptop (Intel Core Ultra 5 125U, 12 cores; 16GB main memory; SSD; Windows 11). 
The same dataset could not be processed on a less powerful MacBook Air (M4; 16GB main memory; MacOS Tahoe 26.4.1; only passive cooling).

Please note that the generation of the embeddings is the most time-consuming step. Thus it is recommended to generate these embeddings once (e.g., over night or a weekend). Once these have been calculated, the embeddings are cached on disk, which significantly reduces the duration of this step from hours to few minutes. 

## Disclaimer

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Demonstration datasets

There are two demonstrations available in AnnoMe. These try to train classifiers for
- Flavonoid and isoflavonoid compounds
- Prenylated flavonoids and isoflavonoids and prenylated chalcones.

The ground truth datasets used for training and testing/validation are
- MSnLib spectra libraries
- an in-house database obtained at BOKU University
- a wheat ear extract obtained at BOKU University.

The inference dataset are obtained from extracts of plants that are known producers of prenylated flavonoid compounds. The plants are:
- *Paulownia tomentosa* (fruits)
- *Glycyrrhizza uralensis* (roots)
plants, which are known to produce prenylated flavonoid compounds.

To execute the respective calculation, please refer to [Instructions_DemoPrenylatedFlavonoids.md](https://github.com/chrboku/AnnoMe/blob/main/Instructions_DemoPrenylatedFlavonoids.md). 
