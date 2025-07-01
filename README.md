# AnnoMe

This is a package for the classification of MS/MS spectra of novel compounds into either "interesting" or "other" compounds. In this respect, the classification result "interesting"  refers to specific substance classes of interest to the user, while the classification result "other" refer to structures not of interest. For this, the classifiers first need to be trained on a large set of MS/MS spectra of compounds of interest (e.g., obtained from reference standards) and others (e.g., obtained from reference standards of other compounds and from large MS/MS spectra repositories). 

## Limitations

Classification of MS/MS spectra into substance classes is a non-trivial task and most often the classification will not be successful, especially when the training dataset is highly imbalanced. The presented python package is not capable of resolving this issue. Therefore, the users are herewith informed to excercise caution with the generated results (i.e., the assignment of MS/MS spectra to specific compound classes of interst). 

## Methodology

Using training data consisting of annotated and labeled MS/MS spectra of compounds of interest and others, different classifiers (LDA, NN, SVM, etc.) are first trained using cross-validation and different random seeds. The results are then aggregated and a majority-vote is derived which indicates the final classification of the input MS/MS spectra to either "interesting" or "other". The term "interesting" refers to compounds of interest, while the term "other" refers to compounds not of interest. 

## Disclaimer

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

