import pandas as pd

from dtreg.load_datatype import load_datatype
from dtreg.to_jsonld import to_jsonld


dt1 = load_datatype("https://doi.org/21.T11969/feeb33ad3e4440682a4d")  # Data Analysis
dt2 = load_datatype("https://doi.org/21.T11969/5e782e67e70d0b2a022a")  # Algorithm Evaluation


time_limits = [30, 60, 300, 900, 3600]

methods = [
    "AutoPrognosis",
    "AutoGluon",
    "TabPFN",
    "HyperFast",
    "Logistic regression",
]


rows = []

for time_limit in time_limits:
    df = pd.read_csv(f"results/test_scores_wide_{time_limit}.csv", sep=None, engine="python")
    df.columns = df.columns.str.strip()

    available_methods = [col for col in df.columns if col != "PMLB dataset"]

    for method in available_methods:
        rows.append({
            "time_limit_seconds": time_limit,
            "approach": method,
            "median_auc": df[method].median(),
            "number_of_datasets": len(df)
        })

runtime_summary = pd.DataFrame(rows)


da1 = dt1.data_analysis(
    has_part=dt2.algorithm_evaluation(
        label=(
            "Runtime-budget comparison of machine learning methods on "
            "PMLBmini datasets using test AUC."
        ),

        executes=dt2.software_method(
            label="Figure 5 runtime plot generation in walkthrough.ipynb",
            is_implemented_by="""
py"generate_runtime_plot"(
    [30, 60, 300, 900, 3600],
    ["AutoPrognosis", "AutoGluon", "TabPFN", "HyperFast", "Logistic regression"],
    "results"
)
""",
            part_of=dt2.software_library(
                label="Python helper generate_runtime_plot, pandas, matplotlib",
                part_of=dt2.software(
                    label="Python",
                    version_info="3"
                )
            )
        ),

        evaluates=dt2.algorithm(
            label="AutoPrognosis, AutoGluon, TabPFN, HyperFast, Logistic regression"
        ),

        evaluates_for=dt2.task(
            label="Binary tabular classification under different runtime budgets"
        ),

        has_input=dt2.data_item(
            label=(
                "PMLBmini benchmark test score files for runtime budgets "
                "30, 60, 300, 900, and 3600 seconds."
            )
        ),

        has_output=dt2.data_item(
            label="Median test AUC by method and runtime budget for Figure 5",
            source_table=runtime_summary
        )
    )
)


json = to_jsonld(da1)

with open("pmlbmini_figure5_runtime_algorithm_evaluation.json", "w", encoding="utf-8") as f:
    f.write(json)

print("Saved: pmlbmini_figure5_runtime_algorithm_evaluation.json")
print(runtime_summary)