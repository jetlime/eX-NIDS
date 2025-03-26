import pandas as pd
from os import listdir

models = ["gpt-4-0613", "Llama-3-70B-Instruct"]
datasets = ["unsw-nb15", "cse-cic-ids2018"]
dataframes = {}

for model in models:
    dataframes[model] = {}
    for dataset in datasets:
        response_entries = [0] * 50
        prompt_entries = [0] * 50
        responses_file_names = listdir(f"./results-{model}-{dataset}")
        for file_name in responses_file_names:
            with open(f"./results-{model}-{dataset}/{file_name}", "r") as file:
                response = file.read()
            response_entries[int(file_name.split("-")[1]) - 1] = response

        prompts_file_names = listdir(f"./evaluation-set-{dataset}")
        for file_name in prompts_file_names:
            with open(f"./evaluation-set-{dataset}/{file_name}", "r") as file:
                # Remove first prompt sentence to keep solely the NetFlow
                prompt = file.read()[263:].replace(',', '\n')
            prompt_entries[int(file_name.split("-")[1]) - 1] = prompt
        dataframes[model][dataset] = pd.DataFrame(
            {
                "Prompts": prompt_entries,
                "Responses": response_entries,
                "Expert #1 Correctness": [0] * 50,
                "Expert #1 Hallucination": [0] * 50,
                "Expert #1 Feature Consistency": [0] * 50,
                "Expert #2 Correctness": [0] * 50
            }
        )

with pd.ExcelWriter("evaluation_grids.xlsx", engine="xlsxwriter") as writer:
    for model in models:
        for dataset in datasets:
            dataframes[model][dataset].to_excel(
                writer, sheet_name=f"{model}-{dataset.split('-')[0]}"
            )
