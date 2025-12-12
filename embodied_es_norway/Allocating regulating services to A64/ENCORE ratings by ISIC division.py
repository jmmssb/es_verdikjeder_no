import pandas as pd
import numpy as np

from pathlib import Path

p = Path(r"C:\Users\dth\OneDrive - Statistisk sentralbyrå\VSCode\ENCORE\06. Dependency mat ratings.csv")

columns_to_keep = ["ISIC Section", "ISIC Division", "ISIC Group", "ISIC Class", "ISIC level used for analysis ", "Air Filtration", "Flood mitigation services", "Local (micro and meso) climate regulation", "Recreation related services"]
df = pd.read_csv(p, usecols=columns_to_keep, na_values=[], keep_default_na=False)
print(df.head())
print(df.columns.tolist())

letter_to_value = {
    "N/A" : 0,
    "VL": 1,
    "L": 2,
    "M": 3,
    "H": 4,
    "VH": 5,
    "ND": np.nan,
}

# Define map_rating function
def map_rating(x):
    s = str(x).strip()
    if s in ("N/A", "NA", "n/a", "na"):
        return 0
    if s == "":
        return np.nan
    if pd.isna(x):
        return np.nan
    return letter_to_value.get(s, np.nan)

#New columns with numeric values
ecosystem_services = ["Air Filtration", "Flood mitigation services", "Local (micro and meso) climate regulation", "Recreation related services"]
value_columns = [f"{service}_value" for service in ecosystem_services]

#Raw values to catch typos
print("\nRaw unique values per service (sample):")
for s in ecosystem_services:
    print(s, "->", pd.Series(df[s].dropna().astype(str).str.strip().unique()[:20]))

for service in ecosystem_services:
    new_col_name = f"{service}_value"
    df[new_col_name] = df[service].apply(map_rating) 

print("\nMapped numeric values per service (sample):")
for col in value_columns:
    print(f"{col} -> {df[col].dropna().unique()[:20]}")

value_columns = [f"{service}_value" for service in ecosystem_services]
print("\nValue counts after mapping (including zeros and NaN):")
for col in value_columns:
    print(col)
    print(df[col].value_counts(dropna=False))

print(df.columns.tolist())

value_columns = [f"{service}_value" for service in ecosystem_services]
print(df[value_columns].head())  # Print only the new numeric columns

#Check value distribution
for col in value_columns:
    print(f"Value counts for {col}:")
    print(df[col].value_counts(dropna=False))

#Simple averages by ISIC Section
averages_by_section = df.groupby("ISIC Section")[value_columns].mean()
print(averages_by_section)

#Weight by number of unique groups and classes in each section
def weighted_average(group):
    n_groups = group["ISIC Group"].nunique()
    n_classes = group["ISIC Class"].nunique()
    weight = n_groups * n_classes
    
    weighted_means = {}
    for col in value_columns:
        weighted_means[col] = (group[col].mean() * weight)/ weight
    return pd.Series(weighted_means)

# Weighted-by-(group,class) averages that stay between 1 and 5:
# For each ISIC Section, take the mean for each unique (ISIC Group, ISIC Class),
# then average those subgroup means so each group/class has equal weight.
def weighted_by_group_class(section_df):
    subgroup_means = section_df.groupby(["ISIC Group", "ISIC Class"])[value_columns].mean()
    return subgroup_means.mean()  # mean() across subgroup rows -> stays in 1-5

weighted_averages = df.groupby("ISIC Section").apply(weighted_by_group_class)
print("\nWeighted-by-(group,class) averages by section (1-5):")
print(weighted_averages)

weighted_averages.to_csv("ecosystem_services_weighted_by_group_class.csv")

print("\n" + "="*80)
print("VERIFICATION: Raw Letters vs. Mapped Numbers")
print("="*80)

for service in ecosystem_services:
    col_name = f"{service}_value"
    print(f"\n{service}:")
    
    # Create a temporary dataframe with raw and mapped values
    check_df = pd.DataFrame({
        "raw": df[service],
        "mapped": df[col_name]
    }).drop_duplicates().sort_values("mapped")
    
    print(check_df.to_string())

print("\n" + "="*80)
print("Weighted-by-class averages by ISIC Group")
print("="*80)

# Weighted-by-(class) averages for ISIC Group
def weighted_by_class_for_group(group_df):
    class_means = group_df.groupby("ISIC Class")[value_columns].mean()
    # drop rows where all services are NaN
    class_means = class_means.dropna(how="all")
    if len(class_means) == 0:
        return pd.Series({c: np.nan for c in value_columns})
    return class_means.mean(axis=0, skipna=True)

weighted_averages_by_group = df.groupby("ISIC Group").apply(weighted_by_class_for_group)
print(weighted_averages_by_group)

weighted_averages_by_group.to_csv("ecosystem_services_weighted_by_isic_group.csv")


print("\n" + "="*80)
print("Weighted-by-group averages by ISIC Division")
print("="*80)

# Weighted-by-(group) averages for ISIC Division
def weighted_by_group_for_division(division_df):
    group_means = division_df.groupby("ISIC Group")[value_columns].mean()
    # drop rows where all services are NaN
    group_means = group_means.dropna(how="all")
    if len(group_means) == 0:
        return pd.Series({c: np.nan for c in value_columns})
    return group_means.mean(axis=0, skipna=True)

weighted_averages_by_division = df.groupby("ISIC Division").apply(weighted_by_group_for_division)
print(weighted_averages_by_division)

weighted_averages_by_division.to_csv("ecosystem_services_weighted_by_isic_division.csv")

# Combine all averages into one table
print("\n" + "="*80)
print("COMBINED: All Weighted Averages by Level")
print("="*80)

# Create a combined dataframe with level labels
combined_averages = pd.concat([
    weighted_averages.rename_axis("Level").assign(Level="Section"),
    weighted_averages_by_division.rename_axis("Level").assign(Level="Division"),
    weighted_averages_by_group.rename_axis("Level").assign(Level="Group")
])

# Reorder columns: Level first, then the categories, then services
combined_averages = combined_averages[["Level"] + [col for col in combined_averages.columns if col != "Level"]]

print(combined_averages)

combined_averages.to_csv("ecosystem_services_all_averages.csv")
print("\nSaved to: ecosystem_services_all_averages.csv")