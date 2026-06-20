import pandas as pd

from dtreg.load_datatype import load_datatype
from dtreg.to_jsonld import to_jsonld


# Data Analysis
dt1 = load_datatype("https://doi.org/21.T11969/feeb33ad3e4440682a4d")

# Descriptive Statistics
dt2 = load_datatype("https://doi.org/21.T11969/5b66cb584b974b186f37")


methods = ["AutoPrognosis", "AutoGluon", "TabPFN", "HyperFast"]

category_counts = {
    "Clustering": [0, 0, 0, 1],
    "Complexity": [3, 2, 0, 0],
    "Concept": [0, 4, 0, 0],
    "General": [0, 0, 0, 0],
    "Info theory": [2, 0, 0, 0],
    "Itemset": [2, 0, 0, 0],
    "Landmarking": [0, 1, 0, 0],
    "Model-based": [0, 1, 0, 0],
    "Statistical": [3, 2, 10, 9],
}

rows = []

for category, counts in category_counts.items():
    for method, count in zip(methods, counts):
        rows.append({
            "approach": method,
            "metafeature_category": category,
            "count": count,
        })

correlation_category_summary = pd.DataFrame(rows)

da1 = dt1.data_analysis(
    has_part=dt2.descriptive_statistics(
        label=(
            "Summary of PMLBmini meta-feature categories correlated with "
            "machine learning method performance, used for Figure 3."
        ),
        executes=dt2.software_method(
            label="Metafeature correlation category summary and stacked bar plot in walkthrough.ipynb",
            is_implemented_by="""
methods = ["AutoPrognosis", "AutoGluon", "TabPFN", "HyperFast"]

py"generate_correlations"(
    methods,
    "results/test_scores_wide_3600.csv"
)

StatsPlots.groupedbar(
    [clustering complexity concept general infotheory itemset landmarking modelbased statistical],
    bar_position=:stack,
    xticks=(1:4, ["AutoPrognosis" "AutoGluon" "TabPFN" "HyperFast"]),
    label=["Clustering" "Complexity" "Concept" "General" "Info theory" "Itemset" "Landmarking" "Model-based" "Statistical"]
)

savefig("plots/bar.svg")
savefig("plots/bar.pdf")
""",
        ),
        has_input=dt2.data_item(
            label=(
                "PMLBmini 3600-second benchmark test scores and generated "
                "meta-feature correlations from correlations.txt."
            )
        ),
        has_output=dt2.data_item(
            label=(
                "Counts of correlated meta-feature categories by machine learning method "
                "for the Figure 3 stacked bar plot."
            ),
            source_table=correlation_category_summary,
        ),
    )
)

json = to_jsonld(da1)

with open("pmlbmini_figure3_metafeature_correlations.json", "w", encoding="utf-8") as f:
    f.write(json)

print("Saved: pmlbmini_figure3_metafeature_correlations.json")
print(correlation_category_summary)