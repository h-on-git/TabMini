import pandas as pd

from dtreg.load_datatype import load_datatype
from dtreg.to_jsonld import to_jsonld


dt1 = load_datatype("https://doi.org/21.T11969/feeb33ad3e4440682a4d")  # Data Analysis
dt2 = load_datatype("https://doi.org/21.T11969/5e782e67e70d0b2a022a")  # Algorithm Evaluation


# Same input file used by Figure 2b in walkthrough.ipynb
results_long = pd.read_csv("results/test_scores_long_3600.csv", sep=";")


# Recreate the main ranking idea behind the critical difference diagram.
# Higher AUC is better, so rank 1 means best performance on a dataset.
ranked_results = results_long.copy()
ranked_results["rank"] = ranked_results.groupby("dataset")["auc"].rank(
    method="average",
    ascending=False
)

rank_summary = (
    ranked_results
    .groupby("approach", as_index=False)["rank"]
    .mean()
    .rename(columns={"rank": "mean_rank"})
    .sort_values("mean_rank")
)


# Statement 2:
# The evaluated machine learning methods are compared by their AUC ranks
# across PMLBmini datasets using a critical difference diagram.

da1 = dt1.data_analysis(
    has_part=dt2.algorithm_evaluation(
        label=(
            "Critical difference comparison of machine learning methods "
            "on PMLBmini datasets using AUC ranks."
        ),

        executes=dt2.software_method(
            label="Figure 2b critical difference diagram generation in walkthrough.ipynb",
            is_implemented_by="""
cdd_plot = CriticalDifferenceDiagrams.plot(
    results_long,
    :approach,
    :dataset,
    :auc,
    maximize_outcome=true
)

PGFPlots.save("plots/cdd.tex", cdd_plot)
""",
            part_of=dt2.software_library(
                label="CriticalDifferenceDiagrams.jl, PGFPlots.jl, CSV.jl, DataFrames.jl",
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
            label="PMLBmini benchmark test scores from results/test_scores_long_3600.csv",
            source_table=results_long
        ),

        has_output=dt2.data_item(
            label="Mean AUC ranks by method and critical difference diagram saved as plots/cdd.tex",
            source_table=rank_summary
        )
    )
)


json = to_jsonld(da1)

with open("pmlbmini_figure2b_algorithm_evaluation.json", "w", encoding="utf-8") as f:
    f.write(json)

print("Saved: pmlbmini_figure2b_algorithm_evaluation.json")
print(rank_summary)