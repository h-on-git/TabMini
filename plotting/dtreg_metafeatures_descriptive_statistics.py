import pandas as pd

from dtreg.load_datatype import load_datatype
from dtreg.to_jsonld import to_jsonld


# Data Analysis
dt1 = load_datatype("https://doi.org/21.T11969/feeb33ad3e4440682a4d")

# Descriptive Statistics
dt2 = load_datatype("https://doi.org/21.T11969/5b66cb584b974b186f37")


metafeatures = pd.read_csv("metafeatures.csv")

selected_columns = [
    "nr_inst",
    "nr_attr",
    "freq_class.min",
    "EPV",
    "nr_bin",
]

selected_columns = [col for col in selected_columns if col in metafeatures.columns]

summary_statistics = (
    metafeatures[selected_columns]
    .describe()
    .T
    .reset_index()
    .rename(columns={"index": "metafeature"})
)

da1 = dt1.data_analysis(
    has_part=dt2.descriptive_statistics(
        label=(
            "Descriptive statistics of PMLBmini dataset meta-features, "
            "including sample size and feature set size."
        ),
        executes=dt2.software_method(
            label="Metafeature summary and visualization in walkthrough.ipynb",
            is_implemented_by="""
metafeatures = CSV.read("metafeatures.csv", DataFrame)

Plots.boxplot(
    ["TabMini"],
    metafeatures[:, "nr_inst"],
    ylabel="Sample size"
)

Plots.scatter(
    metafeatures[:, "nr_inst"],
    metafeatures[:, "nr_attr"],
    xlabel="Sample size",
    ylabel="Feature set size"
)
""",
        ),
        has_input=dt2.data_item(
            label="PMLBmini dataset meta-features from metafeatures.csv",
            source_table=metafeatures[selected_columns],
        ),
        has_output=dt2.data_item(
            label=(
                "Summary statistics for selected PMLBmini meta-features "
                "used for the boxplot and scatter plot."
            ),
            source_table=summary_statistics,
        ),
    )
)

json = to_jsonld(da1)

with open("pmlbmini_metafeatures_descriptive_statistics.json", "w", encoding="utf-8") as f:
    f.write(json)

print("Saved: pmlbmini_metafeatures_descriptive_statistics.json")
print(summary_statistics)