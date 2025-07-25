import pandas as pd

inFile = "I:/Tomas_PrenylatedCompounds/analysis/results/HCD_{ion_mode}__sirius/canopus_formula_summary-15.tsv"
feature_column = "mappingFeatureId"
class_column = "ClassyFire#most specific class"
relevant_classes = [
    "6-prenylated flavanones",
    "6-prenylated isoflavanones",
    "8-prenylated flavanones",
    "3'-prenylated flavanones",
    "3-prenylated flavones",
    "3-prenylated chalcones",
    "8-prenylated isoflavanones",
    "Pyranoisoflavonoids",
    "8-prenylated flavones",
    "6-prenylated flavones",
    "Pyranoflavonoids",
    "2-prenylated xanthones",
    "4-prenylated xanthones",
    "8-prenylated xanthones",
]

for ion_mode in ["pos", "neg"]:
    print(f"\n\n\n\n#####################################################################")
    print(f"Processing ion mode: {ion_mode}")

    # Read the input file
    df = pd.read_csv(inFile.format(ion_mode=ion_mode), sep="\t")

    # Print the number of unique features
    print("\n---------------------------------------------------------------------")
    print(f"Number of unique features: {len(df[feature_column].unique())}")

    # Print the number of predictions (i.e., the number of rows)
    print("\n---------------------------------------------------------------------")
    print(f"Number of total predictions (any rank): {df.shape[0]}")

    # Print the unique classes
    unique_classes = df[class_column].unique()
    print("\n---------------------------------------------------------------------")
    print(f"Unique classes (truncated)")
    class_counts = df[class_column].value_counts()
    for anno_idx, (cls, count) in enumerate(class_counts.items()):
        print(f"   - {cls}: {count}")
        if anno_idx > 10:
            print("   - ...")
            break

    # Filter the DataFrame for relevant classes
    df_filtered = df[df[class_column].isin(relevant_classes)]
    num_annotated_features = df_filtered[feature_column].nunique()
    print("\n---------------------------------------------------------------------")
    print(
        f"Number of featureIds annotated at least once as a relevant class: {num_annotated_features}, that is {num_annotated_features / len(df[feature_column].unique()) * 100:.2f}% of all {len(df[feature_column].unique())} features."
    )

    # Aggregate by class_column and "formulaRank", count occurrences
    agg_stats = df_filtered.groupby([class_column, "formulaRank"]).agg(spectra_count=("mappingFeatureId", "size"), unique_features=("mappingFeatureId", "nunique")).reset_index()
    print("\n---------------------------------------------------------------------")
    print("Counts by class and formulaRank (spectra and unique features):")
    pd.set_option("display.max_rows", None)
    print(agg_stats)
