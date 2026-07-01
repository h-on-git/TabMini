import json
import pandas as pd


INPUT_FILE = "results/test_scores_wide_3600.csv"

OUTPUT_CSV = "pmlbmini_figure2a_structured_values.csv"
OUTPUT_JSON = "pmlbmini_figure2a_structured_records.json"


methods = [
    "AutoPrognosis",
    "AutoGluon",
    "TabPFN",
    "HyperFast",
    "Logistic regression",
]

sample_size_groups = [
    {
        "sample_size_range": "32 to 100",
        "start_index": 0,
        "end_index": 12,
    },
    {
        "sample_size_range": "101 to 200",
        "start_index": 12,
        "end_index": 22,
    },
    {
        "sample_size_range": "201 to 300",
        "start_index": 22,
        "end_index": 31,
    },
    {
        "sample_size_range": "301 to 400",
        "start_index": 31,
        "end_index": 39,
    },
    {
        "sample_size_range": "401 to 500",
        "start_index": 39,
        "end_index": 44,
    },
]


results_wide = pd.read_csv(INPUT_FILE, sep=None, engine="python")
results_wide.columns = results_wide.columns.str.strip()

rows = []
record_id = 1

for group in sample_size_groups:
    sample_size_range = group["sample_size_range"]
    start_index = group["start_index"]
    end_index = group["end_index"]

    group_data = results_wide.iloc[start_index:end_index]

    for method in methods:
        value = round(float(group_data[method].quantile(0.5)), 3)

        statement = (
            f"For PMLBmini datasets in the sample-size range {sample_size_range}, "
            f"{method} achieved a median mean test AUC of {value:.3f}."
        )

        rows.append({
            "id": record_id,
            "benchmark_suite": "PMLBmini",
            "task": "binary tabular classification",
            "sample_size_range": sample_size_range,
            "method": method,
            "metric": "mean test AUC",
            "aggregation": "median",
            "quantile": 0.5,
            "value": value,
            "source_file": INPUT_FILE,
            "source_figure": "Figure 2a / AUC plot",
            "statement": statement,
        })

        record_id += 1


structured_table = pd.DataFrame(rows)

structured_table.to_csv(OUTPUT_CSV, index=False)

records = structured_table.to_dict(orient="records")

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print(f"Saved: {OUTPUT_CSV}")
print(f"Saved: {OUTPUT_JSON}")
print()
print(structured_table)