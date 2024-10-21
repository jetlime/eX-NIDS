from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from datasets import load_dataset
import pandas as pd

import os

# directory = os.fsencode('.')
# file_per_prompt = {}
# prompts = [1, 2,3, 4]
# for prompt in prompts:
#     file_per_prompt[str(prompt)] = []
    
# for file in os.listdir(directory):
#     filename = os.fsdecode(file)
#     if filename.endswith(".txt"): 
#         file_per_prompt[filename.split("prompt")[1][0]].append(filename)
#     else:
#         continue

# target_names = ['benign', 'malicious']


# # Initialize lists to store the metrics
# data = {
#     "Prompt": [],
#     "Metric": [],
#     "Score": [],
#     "Model": []
# }

# for prompt in prompts:
#     for file_result in file_per_prompt[str(prompt)]:
#         seed_random = file_result.split('-')[3].split(".")[0]
#         dataset = load_dataset("Jetlime/NF-UNSW-NB15-v2", streaming=False, split="test")
#         dataset = dataset.train_test_split(test_size=0.2, seed=int(seed_random), stratify_by_column="Attack")
#         dataset = dataset["test"]
#         dataset

#         correctness_percentages = []

#         true_labels = []
#         for i in tqdm(dataset):
#             true_labels.append(int(i["output"]))

#         total_labels = len(true_labels)

#         filtered_prediction_labels = []
#         filtered_true_labels = []

#         with open(file_result, 'r') as file:
#             # Read all lines from the file
#             lines = file.readlines()

#         # Strip whitespace characters (like \n) from the end of each line
#         prediction_labels = [line.strip() for line in lines]

#         # Iterate over the lists and filter labels
#         for pred, true in zip(prediction_labels, true_labels):
#             if pred == "0" or pred =="1":
#                 filtered_prediction_labels.append(int(pred))
#                 filtered_true_labels.append(true)

#         classification_report_results = classification_report(filtered_true_labels, filtered_prediction_labels, digits=4, target_names=target_names)

#         print(classification_report_results) 
#         # Split the report into lines
#         report_lines = classification_report_results.split('\n')

#         # Extract the line containing macro avg
#         for line in report_lines:
#             if 'macro avg' in line:
#                 macro_avg_line = line.split()
#                 break

#         # Extract precision, recall, and f1-score from the macro_avg_line
#         macro_avg_precision = float(macro_avg_line[2]) * 100
#         macro_avg_recall = float(macro_avg_line[3]) * 100
#         macro_avg_f1_score = float(macro_avg_line[4]) * 100

#         # Append the data to the lists
#         data["Prompt"].extend([prompt]*2)
#         data["Metric"].extend(["Precision", "Recall"])
#         data["Score"].extend([macro_avg_precision, macro_avg_recall])
#         data["Model"].extend(["LLama-3-8B-Instruct"]*2)


# # # Create a DataFrame
# print(data)
# df = pd.DataFrame(data)
# plt.figure(figsize=(14, 10))

# sns.barplot(x="Prompt", y="Score", hue="Metric", data=df, capsize=0.05, errwidth=1.1, width=0.4, hatch='xx')

gpt_results = {'Prompt': ['1', '1', '1', '1', '2', '2', '2', '2', '3', '3','3', '3', '4', '4', '4', '4'], 'Metric': ['Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall'], 'Score': [50.86000000000001, 53.559999999999995, 50.8900000000000, 54.214, 50.849999999999994, 53.47, 50.75, 53.41, 51.23, 54.86, 51.35, 54.95, 51.81, 56.92, 52.11, 57.92], 'Model': ['GPT-4', 'GPT-4', 'GPT-4', 'GPT-4', 'GPT-4', 'GPT-4', 'GPT-4', 'GPT-4', 'GPT-4', 'GPT-4', 'GPT-4', 'GPT-4', 'GPT-4', 'GPT-4', 'GPT-4', 'GPT-4']}
data = {'Prompt': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 'Metric': ['Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall', 'Precision', 'Recall'], 'Score': [50.09, 50.57000000000001, 50.13999999999999, 50.86000000000001, 50.17, 51.06, 50.09, 50.519999999999996, 50.18, 51.09, 50.029999999999994, 50.18, 49.94, 49.65, 50.09, 50.55, 49.96, 49.769999999999996, 50.06, 50.38, 48.980000000000004, 43.769999999999996, 48.980000000000004, 43.74, 49.01, 43.9, 49.08, 44.37, 49.07, 44.24, 48.980000000000004, 43.74, 48.97, 43.66, 49.0, 43.84, 48.949999999999996, 43.519999999999996, 51.78, 61.1, 51.99, 62.4, 52.01, 62.53999999999999, 51.839999999999996, 61.46, 51.739999999999995, 60.85, 51.839999999999996, 61.5, 51.790000000000006, 61.22, 51.790000000000006, 61.18, 51.64, 60.25, 48.46, 43.169999999999995, 48.57, 43.74, 48.52, 43.43, 48.59, 43.82, 48.43, 43.13, 48.49, 43.269999999999996, 48.76, 44.529999999999994, 48.65, 44.080000000000005, 48.480000000000004, 43.26], 'Model': ['LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct', 'LLama-3-8B-Instruct']}
# sns.barplot(x="Prompt", y="Score", hue="Metric", data=df, capsize=0.05, errwidth=1.1, width=0.4, hatch='-')

# sns.set(font_scale=2)
# plt.rcParams["axes.labelsize"] = 26
# plt.axhline(y=50, color='royalblue', linestyle='--')
# plt.ylim(0, 100)
# plt.xlabel('Prompt ID', fontsize=26)  # Set x-axis label font size
# plt.ylabel('Macro Average Performance Score (%)', fontsize=26)  # Set y-axis label font size
# plt.legend(title='Model')
# plt.savefig("./prompt-variation-result-unsw-nb15.png")

# Add GPT results to the data
# data["Prompt"].extend(gpt_results["Prompt"])
# data["Metric"].extend(gpt_results["Metric"])
# data["Score"].extend(gpt_results["Score"])
# data["Model"].extend(["GPT Model"] * len(gpt_results["Prompt"]))  # Label for the GPT model

# Create a DataFrame
df_gpt = pd.DataFrame(gpt_results)
df = pd.DataFrame(data)

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(28, 10))  # Create two subplots side by side


# sns.barplot(x="Prompt", y="Score", hue="Metric", ax=ax1, data=df, capsize=0.05, errwidth=1.1, width=0.4)
# sns.set(font_scale=2)
# plt.rcParams["axes.labelsize"] = 26
# plt.axhline(y=50, color='royalblue', linestyle='--')
# ax1.set_ylim(0, 100)
# ax1.set_ylabel('Performance Score (%)', fontsize=26)  # Set y-axis label font size
# ax1.legend(title='Model')

# sns.barplot(x="Prompt", y="Score", hue="Metric", ax=ax2, data=df_gpt, capsize=0.05, errwidth=1.1, width=0.4)
# sns.set(font_scale=2)
# # plt.rcParams["axes.labelsize"] = 26
# # plt.axhline(y=50, color='royalblue', linestyle='--')
# ax2.set_xlim(0, 100)
# ax2.set_xlabel('Prompt ID', fontsize=26)  # Set x-axis label font size
# ax2.set_ylabel('Performance Score (%)', fontsize=26)  # Set y-axis label font size
# ax2.legend(title='Avg. Macro Performance Metric')
print(data)

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

plt.savefig("./prompt-variation-result-unsw-nb15.png")