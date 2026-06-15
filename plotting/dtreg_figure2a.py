import pandas as pd

from dtreg.load_datatype import load_datatype
from dtreg.to_jsonld import to_jsonld


# DOI identifiers from the professor's README:
# Data Analysis + Algorithm Evaluation
dt1 = load_datatype("https://doi.org/21.T11969/feeb33ad3e4440682a4d")
dt2 = load_datatype("https://doi.org/21.T11969/5e782e67e70d0b2a022a")


# This is the same CSV used for Figure 2a in walkthrough.ipynb.
# Important: the file is semicolon-separated.
results_wide = pd.read_csv("results/test_scores_wide_3600.csv", sep=";")


methods = [
    "AutoPrognosis",
    "AutoGluon",
    "TabPFN",
    "HyperFast",
    "Logistic regression",
]

sample_size_ranges = {
    "32 to 100": range(0, 12),
    "101 to 200": range(12, 22),
    "201 to 300": range(22, 31),
    "301 to 400": range(31, 39),
    "401 to 500": range(39, 44),
}


# Recreate the median values used in Figure 2a.
rows = []

for sample_range_label, row_range in sample_size_ranges.items():
    row = {"sample_size_range": sample_range_label}

    for method in methods:
        row[method] = results_wide.iloc[list(row_range)][method].median()

    rows.append(row)

auc_summary = pd.DataFrame(rows)


# Statement 1:
# AutoPrognosis, AutoGluon, TabPFN, HyperFast, and Logistic regression
# are evaluated on PMLBmini datasets using mean test AUC across sample-size ranges.

da1 = dt1.data_analysis(
    has_part=dt2.algorithm_evaluation(
        label=(
            "Evaluation of machine learning methods on PMLBmini datasets "
            "using mean test AUC across sample-size ranges."
        ),

        executes=dt2.software_method(
            label="Figure 2a generation in walkthrough.ipynb",
            is_implemented_by="""
methods = ["AutoPrognosis" "AutoGluon" "TabPFN" "HyperFast" "Logistic regression"]
sample_size_ranges = [1:12, 13:22, 23:31, 32:39, 40:44]

Q3s[idx_r, idx_a] = quantile(results_wide[sample_size_range, approach], 0.75)
Q2s[idx_r, idx_a] = quantile(results_wide[sample_size_range, approach], 0.5)
Q1s[idx_r, idx_a] = quantile(results_wide[sample_size_range, approach], 0.25)

Plots.plot(Q2s, ribbon=(Q2s .- Q1s, Q3s .- Q2s))
""",
            part_of=dt2.software_library(
                label="Plots.jl, CSV.jl, DataFrames.jl, StatsBase.jl",
                part_of=dt2.software(
                    label="Julia",
                    version_info="1.12"
                )
            )
        ),

        evaluates=dt2.algorithm(
            label="AutoPrognosis, AutoGluon, TabPFN, HyperFast, Logistic regression"
        ),

        evaluates_for=dt2.task(
            label="Binary tabular classification in data-scarce applications"
        ),

        has_input=dt2.data_item(
            label="PMLBmini benchmark test scores from results/test_scores_wide_3600.csv",
            source_table=results_wide
        ),

        has_output=dt2.data_item(
            label="Median AUC values by sample-size range and method for Figure 2a",
            source_table=auc_summary
        )
    )
)


json = to_jsonld(da1)

with open("pmlbmini_figure2a_algorithm_evaluation.json", "w", encoding="utf-8") as f:
    f.write(json)

print("Saved: pmlbmini_figure2a_algorithm_evaluation.json")
print(auc_summary)