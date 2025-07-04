# AnnoMe

This is a package for the classification of MS/MS spectra of novel compounds into either "interesting" or "other" compounds. In this respect, the classification result "interesting"  refers to specific substance classes of interest to the user, while the classification result "other" refer to structures not of interest. For this, the classifiers first need to be trained on a large set of MS/MS spectra of compounds of interest (e.g., obtained from reference standards) and others (e.g., obtained from reference standards of other compounds and from large MS/MS spectra repositories). 

## Limitations

Classification of MS/MS spectra into substance classes is a non-trivial task and most often the classification will not be successful, especially when the training dataset is highly imbalanced. The presented python package is not capable of resolving this issue. Therefore, the users are herewith informed to exercise caution with the generated results (i.e., the assignment of MS/MS spectra to specific compound classes of interest). 

## Methodology

Using training data consisting of annotated and labeled MS/MS spectra of compounds of interest and others, different classifiers (LDA, NN, SVM, etc.) are first trained using cross-validation and different random seeds. The results are then aggregated and a majority-vote is derived which indicates the final classification of the input MS/MS spectra to either "interesting" or "other". The term "interesting" refers to compounds of interest, while the term "other" refers to compounds not of interest. 

## Disclaimer

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

## Usage

* [optional step]: Install the python tool uv. Please refer to [https://github.com/astral-sh/uv#installation](https://github.com/astral-sh/uv#installation) for installation instruction. This step can be omitted if uv is already available on the system. 

* Open a terminal and navigate to an empty folder. 

* Clone the repo AnnoMe with the command **git clone https://github.com/chrboku/AnnoMe**

* Execute the demo filter-script with **uv run python ./demo/Demo_FilterDatabasesPrenylatedCompounds_publicDBs.py**. This will download the publicly available MS/MS databases and the MS2DeepScore model. Then the script will filter the public databases for **prenylated flavones** and **prenylated chalcones**. 

* Execute the demo analysis-script with **uv run jupyter nbconvert --to HTML --execute ./demo/Analysis_PrenylatedCompounds_publicDBs.ipynb**. Once finished, the log of the classification pipeline will be available in **./demo/Analysis_PrenylatedCompounds_publicDBs.html** and the prediction results will be available in **./demo/publicDBs/output/PrenylatedCompounds_publicDBs/**

## TODO: include how a new search can be started, what the parameters are, and documentation about the results