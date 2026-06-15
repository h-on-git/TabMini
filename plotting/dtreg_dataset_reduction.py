import pandas as pd

from dtreg.load_datatype import load_datatype
from dtreg.to_jsonld import to_jsonld


dt1 = load_datatype("https://doi.org/21.T11969/feeb33ad3e4440682a4d")  # Data Analysis
dt2 = load_datatype("https://doi.org/21.T11969/5e782e67e70d0b2a022a")  # Algorithm Evaluation


results_long = pd.read_csv("results/test_scores_long_3600.csv", sep=";")

datasets_reduced = [
    "analcatdata_aids", "analcatdata_asbestos", "analcatdata_bankruptcy",
    "analcatdata_creditscore", "analcatdata_cyyoung8092",
    "analcatdata_cyyoung9302", "analcatdata_fraud",
    "analcatdata_japansolvent", "labor", "lupus", "parity5",
    "postoperative_patient_data", "analcatdata_boxing1",
    "analcatdata_boxing2", "appendicitis", "glass2",
    "molecular_biology_promoters", "mux6", "hungarian",
    "bupa", "colic", "horse_colic", "clean1", "house_votes_84"
]

results_long_reduced = results_long[results_long["dataset"].isin(datasets_reduced)].copy()

ranked_results = results_long_reduced.copy()
ranked_results["rank"] = ranked_results.groupby("dataset")["auc"].rank(
    method="average",
    ascending=False
)

rank_summary_reduced = (
    ranked_results
    .groupby("approach", as_index=False)["rank"]
    .mean()
    .rename(columns={"rank": "mean_rank"})
    .sort_values("mean_rank")
)

da1 = dt1.data_analysis(
    has_part=dt2.algorithm_evaluation(
        label=(
            "Critical difference comparison of machine learning methods "
            "on the reduced PMLBmini dataset subset using AUC ranks."
        ),

        executes=dt2.software_method(
            label="Dataset reduction and reduced critical difference diagram in walkthrough.ipynb",
            is_implemented_by="""
results_long_reduced = DataFrame([String[], String[], Float64[]], names(results_long))
for datasets in datasets_reduced
    for dataset in datasets
        append!(results_long_reduced, results_long[results_long.dataset .== dataset, :])
    end
end

cdd_plot_reduced = CriticalDifferenceDiagrams.plot(
    results_long_reduced,
    :approach,
    :dataset,
    :auc,
    maximize_outcome=true
)

PGFPlots.save("plots/cdd_reduced.tex", cdd_plot_reduced)
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
            label="Binary tabular classification on datasets not meta-trained on by TabPFN"
        ),

        has_input=dt2.data_item(
            label="Reduced PMLBmini benchmark test scores from results/test_scores_long_3600.csv",
            source_table=results_long_reduced
        ),

        has_output=dt2.data_item(
            label="Mean AUC ranks by method for the reduced critical difference comparison",
            source_table=rank_summary_reduced
        )
    )
)

json = to_jsonld(da1)

with open("pmlbmini_dataset_reduction_algorithm_evaluation.json", "w", encoding="utf-8") as f:
    f.write(json)

print("Saved: pmlbmini_dataset_reduction_algorithm_evaluation.json")
print(rank_summary_reduced)