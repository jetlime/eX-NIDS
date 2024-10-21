from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from datasets import load_dataset
import pandas as pd

import os

directory = os.fsencode('.')
file_per_prompt = {}
prompts = [1, 2,3, 4]
for prompt in prompts:
    file_per_prompt[str(prompt)] = []
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".txt"): 
        file_per_prompt[filename.split("prompt")[1][0]].append(filename)
    else:
        continue

target_names = ['benign', 'malicious']


# Initialize lists to store the metrics
data = {
    "Prompt": [],
    "Metric": [],
    "Score": [],
}

for prompt in prompts:
    for file_result in file_per_prompt[str(prompt)]:
        seed_random = file_result.split('-')[3].split(".")[0]
        dataset = load_dataset("Jetlime/NF-UNSW-NB15-v2", streaming=False, split="test")
        dataset = dataset.train_test_split(test_size=0.2, seed=int(seed_random), stratify_by_column="Attack")
        dataset = dataset["test"]
        dataset

        correctness_percentages = []

        true_labels = []
        for i in tqdm(dataset):
            true_labels.append(int(i["output"]))

        total_labels = len(true_labels)

        filtered_prediction_labels = []
        filtered_true_labels = []

        with open(file_result, 'r') as file:
            # Read all lines from the file
            lines = file.readlines()

        # Strip whitespace characters (like \n) from the end of each line
        prediction_labels = [line.strip() for line in lines]

        # Iterate over the lists and filter labels
        for pred, true in zip(prediction_labels, true_labels):
            if pred == "0" or pred =="1":
                filtered_prediction_labels.append(int(pred))
                filtered_true_labels.append(true)

        classification_report_results = classification_report(filtered_true_labels, filtered_prediction_labels, digits=4, target_names=target_names)
        print(prompt)
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

        # Append the data to the lists
        data["Prompt"].extend([prompt] * 2)
        data["Metric"].extend(["Precision", "Recall"])
        data["Score"].extend([macro_avg_precision, macro_avg_recall])



# # Create a DataFrame
df = pd.DataFrame(data)
print(data)
gpt_results = {'Prompt': ['1', '1', '1', '1', '2', '2', '2', '2', '3', '3','3', '3', '4', '4', '4', '4'], 'Metric': ['Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall'], 'Score': [48.86000000000001, 51.559999999999995, 48.8900000000000, 52.214, 48.849999999999994, 51.47, 49.75, 50.41, 50.23, 52.86, 50.35, 51.95, 50.81, 53.92, 50.11, 50.92], 'Model': ['GPT-4', 'GPT-4', 'GPT-4', 'GPT-4', 'GPT-4', 'GPT-4', 'GPT-4', 'GPT-4', 'GPT-4', 'GPT-4', 'GPT-4', 'GPT-4', 'GPT-4', 'GPT-4', 'GPT-4', 'GPT-4']}
df_gpt = pd.DataFrame(gpt_results)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 25))  # Create two subplots

# Plot for df
sns.barplot(x="Prompt", y="Score", hue="Metric", ax=ax1, data=df, capsize=0.05, errwidth=1.1, width=0.4)
ax1.axhline(y=50, color='royalblue', linestyle='--')
ax1.get_xaxis().set_visible(False)
ax1.tick_params(axis='y', labelsize=35)
ax1.set_ylim(0, 100)
ax1.set_ylabel('Performance Score (%)', fontsize=30)  # Set y-axis label font size
ax1.legend(title='Avg. Macro Performance Metric', fontsize=35, title_fontsize=38)
ax1.set_title("LLama3-8b-Instruct", fontsize=40)

# Plot for df_gpt
sns.barplot(x="Prompt", y="Score", hue="Metric", ax=ax2, data=df_gpt, capsize=0.05, errwidth=1.1, width=0.4)
ax2.axhline(y=50, color='royalblue', linestyle='--')
ax2.set_ylim(0, 100)  # Ensure y-axis scaling is the same as the first plot
ax2.tick_params(axis='x', labelsize=35)
ax2.tick_params(axis='y', labelsize=35)
ax2.set_xlabel('Prompt ID', fontsize=30)  # Set x-axis label font size
ax2.set_ylabel('Performance Score (%)', fontsize=30)  # Set y-axis label font size
ax2.get_legend().set_visible(False)
ax2.set_title("GPT-4", fontsize=40)

plt.savefig("./prompt-variation-result-cse-cic-ids2018.png")
