## Parameters
df_file = "./output/classification_results_PrenylatedFlavonoids/default_set_-_-_hcd_pos_stepped/prediction_data.xlsx"
output_dir = "./forPublication/GTLimits_output/PrenylatedCompounds_PublicDBs_TrainingSetLimits"



## Script
# Import necessary libraries
import pandas as pd
import numpy as np

from AnnoMe.Classification import (
    train_and_classify,
    generate_prediction_overview,
    set_random_seeds,
)

from IPython.display import display

import os
import re

import plotnine as p9

from natsort import natsorted

from joblib import Parallel, delayed
import multiprocessing
import traceback
from contextlib import redirect_stdout, redirect_stderr



## Pipeline
# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the DataFrame from the specified pickle file
df = pd.read_excel(df_file)

# Check if the columns "ms2deepscore:cleaned_spectra" and "ms2deepscore:embeddings" are present
if "ms2deepscore:cleaned_spectra" in df.columns and "ms2deepscore:embeddings" in df.columns:
    # Rename the columns
    df.rename(columns={"ms2deepscore:cleaned_spectra": "cleaned_spectra", "ms2deepscore:embeddings": "embeddings"}, inplace=True)

# Convert embeddings column into numpy array for each row
def conv_to_numpy_array(x):
    x = x.replace("[", "").replace("]", "")
    x = x.replace("\n", "")
    x = re.sub(r"\s+", " ", x)
    x = x.replace(" ", ",")
    x = x.split(",")
    x = [float(i) for i in x if i]
    return np.array(x)
df["embeddings"] = df["embeddings"].apply(conv_to_numpy_array)

# show overview of the dataframe column type
type_counts = df["type"].value_counts()
print("DataFrame type counts:")
print(type_counts)


## Function for processing single reduction and iteration thereof
def _process_subset(red_name, red_n, iteration, df, output_dir):
    
    # Reduce the DataFrame to the specified percentage of rows
    c_df = df.copy()
    c_df_other = c_df[c_df["type"] != "train - relevant"]
    c_df_relevant = c_df[c_df["type"] == "train - relevant"].sample(frac=red_n, random_state=iteration)
    df_reduced = pd.concat([c_df_other, c_df_relevant]).reset_index(drop=True)

    num_train_relevant = df_reduced[df_reduced["type"] == "train - relevant"].shape[0]
    num_train_other = df_reduced[df_reduced["type"] == "train - other"].shape[0]
    num_validation_train = df_reduced[df_reduced["type"] == "validation - train"].shape[0]
    num_validation_other = df_reduced[df_reduced["type"] == "validation - other"].shape[0]

    # Create output directory for the subset
    c_output_dir = f"{output_dir}/subset_traRel{num_train_relevant}_it{iteration}/"
    os.makedirs(c_output_dir, exist_ok=True)

    log_path = os.path.join(c_output_dir, "log.txt")

    with open(log_path, "a", encoding="utf-8", buffering=1) as log_file, \
         redirect_stdout(log_file), redirect_stderr(log_file):
        print(f"\n=== START reduction={red_name}, iteration={iteration}, pid={os.getpid()} ===")
        try:
            print(
                f"Number of rows where type is 'train - relevant': {num_train_relevant}, "
                f"'train - other': {num_train_other}, "
                f"'validation - train': {num_validation_train}, "
                f"'validation - other': {num_validation_other}"
            )

            print(f"Processing with reduction: {red_name}, iteration: {iteration}")
            print("##############################################################################")

            # Set random seeds for reproducibility
            set_random_seeds(42)

            # Train and generate overviews
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.model_selection import StratifiedKFold
            classifiers_to_compare = {
                "default_set": {
                    "classifiers": {
                        "LDA": LinearDiscriminantAnalysis(solver="svd", store_covariance=True, n_components=1),
                    },
                    "cross_validation": StratifiedKFold(n_splits=10, random_state=42, shuffle=True),
                    "min_prediction_threshold": 120,
                    "description": "Default set of diverse classifiers for comparison",
                }
            }
            df_train, df_validation, _, _, _ = train_and_classify(
                df_reduced, subsets=lambda x: True, output_dir=c_output_dir,
                #classifiers_to_compare=classifiers_to_compare
            )
            generate_prediction_overview(
                df_reduced, df_train, c_output_dir, "train", min_prediction_threshold=13
            )
            generate_prediction_overview(
                df_reduced, df_validation, c_output_dir, "validation", min_prediction_threshold=13
            )

            print(f"=== END reduction={red_name}, iteration={iteration}, status=ok ===")
            return {
                "red_name": red_name,
                "iteration": iteration,
                "status": "ok",
                "n_train_relevant": num_train_relevant,
            }
        except Exception as e:
            print(f"Error during training and classification for {red_name} it{iteration}: {e}")
            print(traceback.format_exc())
            print(f"=== END reduction={red_name}, iteration={iteration}, status=error ===")
            return {
                "red_name": red_name,
                "iteration": iteration,
                "status": "error",
                "error": str(e),
            }


if False:
    ## Process all reductions and iterations
    # build task list
    tasks = [(red_name, red_n, iteration) for red_name, red_n in [
        ("all", 1.0),
        ("90Percent", 0.9),
        ("80Percent", 0.8),
        ("70Percent", 0.7),
        ("60Percent", 0.6),
        ("50Percent", 0.5),
        ("40Percent", 0.4),
        ("30Percent", 0.3),
        ("20Percent", 0.2),
        ("10Percent", 0.1),
    ] for iteration in range(1, 6)]

    n_jobs = min(multiprocessing.cpu_count(), len(tasks))
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_subset)(red_name, red_n, iteration, df, output_dir) for red_name, red_n, iteration in tasks
    )

    print("Parallel run finished. Summary:")
    print(results)


## Summary
# Generate an overview of the validation datasets measured in-house
print(f"\nGenerating an overview of the validation datasets measured in-house")
# Initialize an empty list to store DataFrames
all_validation_results = []

# Iterate through all folders in the output directory
for folder_name in os.listdir(output_dir):
    folder_path = os.path.join(output_dir, folder_name)
    if os.path.isdir(folder_path):  # Check if it's a directory
        file_path = os.path.join(folder_path, "validation_data.xlsx")
        if os.path.exists(file_path):  # Check if the file exists
            print(f"   - Found validation dataset in folder: {folder_name}")
            # Read the contents of the sheet 'overall'
            df = pd.read_excel(file_path, sheet_name="overall")
            # Add a new column with the folder name
            df["subset"] = folder_name
            # Append the DataFrame to the list
            all_validation_results.append(df)

if len(all_validation_results) == 0:
    print("No validation datasets found in the output directory.")
    all_validation_results = None
else:
    # Concatenate all DataFrames into a single DataFrame
    all_validation_results = pd.concat(all_validation_results, ignore_index=True)
    all_validation_results.rename(columns={"row_count": "n_features"}, inplace=True)
    all_validation_results["percent_features"] = (100.0 * all_validation_results["n_features"] / all_validation_results.groupby(["source", "subset"])["n_features"].\
        transform("sum")).round(1)
    # Split the 'subset' column into three new columns using the regex pattern
    all_validation_results[["reduction", "iteration"]] = all_validation_results["subset"].str.extract(r"subset_(.*)_(.*)")
    all_validation_results["fragmentation_method"] = "hcd"
    all_validation_results["polarity"] = "pos"
    all_validation_results["collision_energy"] = "stepped"
    all_validation_results["from_file"] = all_validation_results["source"].str.replace(r"^filtered_(.*)$", r"\1", regex=True).\
        str.replace(r"^(.*)_(matched|noMatch).mgf$", r"\1", regex=True).str.replace(r"^(.*)\.mgf$", r"\1", regex=True)
    # make from_file more readable to user
    all_validation_results["from_file"] = all_validation_results["from_file"].\
        str.replace(r"^HCD_pos__sirius$", r"Prenylated Flavonoids", regex=True).\
        str.replace(r"^Wheat_pos__sirius$", r"Wheat", regex=True).\
        str.replace(r"^n_wheat_pos__sirius$", r"Wheat", regex=True).\
        str.replace(r"^BOKU_iBAM$", r"Mixed standard (iBAM)", regex=True)
    all_validation_results[["source", "gt_type"]] = all_validation_results["type"].str.extract(r"(.*) - (other|relevant)")
    # Order the DataFrame by 'source', 'subset', and 'annotated_as'
    all_validation_results.sort_values(by=["source", "polarity", "fragmentation_method", "collision_energy", "reduction", "iteration", "gt_type", "annotated_as"], inplace=True)
    # Reorder the columns
    all_validation_results = all_validation_results[
        ["source", "from_file", "polarity", "fragmentation_method", "collision_energy", "reduction", "iteration", "gt_type", "annotated_as", "n_features", "percent_features"]
    ]
    # convert gt_type and annotated_as to TP, FP, TN, FN
    all_validation_results["classification_type"] = all_validation_results.apply(lambda row: f"{row['gt_type']}: {row['annotated_as']}", axis=1)
    all_validation_results["classification_type"] = all_validation_results["classification_type"].replace({
        "other: other": "TN",
        "other: relevant": "FP",
        "relevant: other": "FN",
        "relevant: relevant": "TP",
    })
    # convert to categorical with order
    all_validation_results["classification_type"] = pd.Categorical(all_validation_results["classification_type"], categories=["TN", "FP", "FN", "TP"], ordered=True)


# Aggregate the dataset
aggregated_results = (
    all_validation_results.groupby(["reduction", "gt_type", "annotated_as", "source", "polarity", "fragmentation_method", "collision_energy"])
    .agg(
        mean_percent_features=("percent_features", "mean"), median_percent_features=("percent_features", "median"), sd_percent_features=("percent_features", "std"), 
        count=("percent_features", "count")
    )
    .reset_index()
)

# Display the aggregated table
display(aggregated_results)

# Aggregate the aggregated_results a second time without grouping by rows
second_aggregation = aggregated_results.agg(
    avg_sd_percent_features=("sd_percent_features", "mean"),
    median_sd_percent_features=("sd_percent_features", "median"),
    min_sd_percent_features=("sd_percent_features", "min"),
    max_sd_percent_features=("sd_percent_features", "max"),
)

# Display the second aggregation
display(second_aggregation)

# Ensure the 'reduction' column is sorted in natural order
all_validation_results["reduction"] = all_validation_results["reduction"].str.replace("^traRel", "", regex=True)
all_validation_results["reduction"] = pd.Categorical(all_validation_results["reduction"], categories=natsorted(all_validation_results["reduction"].unique()), ordered=True)

p = (
    p9.ggplot(all_validation_results, p9.aes(x="reduction", y="percent_features", colour="classification_type"))
    + p9.theme_bw()
    + p9.geom_boxplot(alpha=0.2)
    + p9.geom_jitter(size=1, height=0, width=0.1)
    + p9.facet_grid("from_file ~ classification_type")
    + p9.theme(
        axis_text_x=p9.element_text(rotation=90, hjust=1),
        legend_position="top",
        panel_grid_major_x=p9.element_blank(),
        panel_grid_minor_x=p9.element_blank(),
        strip_text=p9.element_text(size=8)
    )
    + p9.labs(title="Overview of Number of Relevant Spectra used for Training", x="Number of MS/MS spectra of class 'relevant' used for training", y="Percent of features", colour="Result Type")
    + p9.scale_color_manual(values={"TN": "#00BFFF", "FP": "#FF7256", "FN": "#FF7256", "TP": "#00BFFF"})
)
display(p)
# Save the plot to a file
output_plot_file = os.path.join(output_dir, "validation_overview.png")
p.save(output_plot_file, width=12, height=12/1.618, dpi=300)

# Export the two tables to an Excel file
output_excel_file = os.path.join(output_dir, "summary_tables.xlsx")
with pd.ExcelWriter(output_excel_file, engine="openpyxl") as writer:
    if all_validation_results is not None:
        all_validation_results.to_excel(writer, sheet_name="all_validation_results")

print(f"Exported tables to {output_excel_file}")