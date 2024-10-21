from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from datasets import load_dataset
import pandas as pd
import json
import os

directory = os.fsencode('.')
file_per_prompt = {}
prompts = [1,2,3,4]


target_names = ['benign', 'malicious']

# Initialize lists to store the metrics
result_per_prompt = {}
result_per_prompt["Prompt"] = []
result_per_prompt["Metric"] = []
result_per_prompt["Score"] = []
result_per_prompt["Model"] = []

for prompt in prompts:
    file_name = f'gpt-4-results-prompt-{str(prompt)}.jsonl'
    dataset = load_dataset("Jetlime/NF-UNSW-NB15-v2", streaming=False, split="test")
    dataset = dataset.train_test_split(test_size=0.2, seed=1234, stratify_by_column="Attack")
    dataset = dataset["test"]
    dataset

    true_labels = []
    for i in tqdm(dataset):
        true_labels.append(int(i["output"]))


    with open(file_name) as f:
        json_list = [json.loads(line) for line in f]
        
    true_labels = true_labels[:len(json_list)]

    # Strip whitespace characters (like \n) from the end of each line
    prediction_labels = []
    index = 0
    for line in json_list:
        prediction = line["response"]["body"]["choices"][0]["message"]["content"]
        if prediction == "1" or prediction == "0":
            prediction = int(prediction)
            prediction_labels.append(prediction)
        else:
            del true_labels[index]
        index += 1

    print(len(prediction_labels))
    print(len(true_labels))

    classification_report_results = classification_report(true_labels, prediction_labels, digits=4, target_names=target_names)

    print(classification_report_results) 
    # Split the report into lines
    report_lines = classification_report_results.split('\n')

    # Extract the line containing macro avg
    for line in report_lines:
        if 'macro avg' in line:
            macro_avg_line = line.split()
            break

    # Extract precision, recall, and f1-score from the macro_avg_line
    macro_avg_precision = float(macro_avg_line[2]) * 100
    macro_avg_recall = float(macro_avg_line[3]) * 100

    result_per_prompt["Prompt"].append(str(prompt))
    result_per_prompt["Prompt"].append(str(prompt))

    result_per_prompt["Metric"].append("Precision")
    result_per_prompt["Metric"].append("Recall")

    result_per_prompt["Score"].append(macro_avg_precision)
    result_per_prompt["Score"].append(macro_avg_recall)
    result_per_prompt["Model"].append("GPT-4")
    result_per_prompt["Model"].append("GPT-4")


print(result_per_prompt)