import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

temperatures = [1.00, 0.80, 0.60, 0.40, 0.20, 0.01]
run_iterations = range(0, 10)

target_names = ['benign', 'malicious']

correctness_percentages = []
temperature_ids = []
for temperature in temperatures:
    if temperature != 0.01:
        temperature_ids.append("{:.2f}".format(temperature)[:-1])
    
    else:    
        temperature_ids.append("{:.2f}".format(temperature))
    correctness_percentage = []

    for i in run_iterations:
        filtered_prediction_labels = []
        with open(f'./temp{str(temperature)}-prediction-labels-{str(i)}.txt', 'r') as file:
            # Read all lines from the file
            lines = file.readlines()
            total_labels = len(lines)

        # Strip whitespace characters (like \n) from the end of each line
        prediction_labels = [line.strip() for line in lines]

        # Iterate over the lists and filter labels
        for pred in prediction_labels:
            if pred == "0" or pred =="1":
                filtered_prediction_labels.append(int(pred))

        # Append the data to the lists
        correctness_percentage.append((((len(filtered_prediction_labels))/total_labels)*100))

    correctness_percentages.append(correctness_percentage)

print(temperature_ids)
print(correctness_percentages)
correctness_df = pd.DataFrame({
    "Temperature": temperature_ids,
    "Correctness": correctness_percentages
})

correctness_df_gpt = pd.DataFrame({
    "Temperature": temperature_ids,
    "Correctness": [[74.47633449266596, 74.84122183577801, 74.50551943142295, 75.33419023136247, 74.91985483139271, 75.55799183426585, 75.04082867079994, 75.04385301678512, 75.33419023136247, 75.3130198094662], [85.26205957961591, 85.70966278542265, 85.49795856646, 85.74897928323, 85.27113261757145, 82.73990624527445, 85.47981249054892, 84.24088915771964, 83.51610464237109, 85.43747164675639], [95.12, 94.89, 94.12, 95.48, 95.12, 96.54, 96.126, 95, 94.5, 96.87751398760018], [100, 100, 100, 100, 99.10, 98.78, 100, 100, 100, 100], [100, 100, 99.91, 100, 100, 100, 100, 100, 100, 100], [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]]
})


# Calculate mean and standard deviation
correctness_df["Mean_Correctness"] = correctness_df["Correctness"].apply(np.mean)
correctness_df["Std_Correctness"] = correctness_df["Correctness"].apply(np.std)
correctness_df = correctness_df.sort_values(by="Temperature", ascending=True)

correctness_df_gpt["Mean_Correctness"] = correctness_df_gpt["Correctness"].apply(np.mean)
correctness_df_gpt["Std_Correctness"] = correctness_df_gpt["Correctness"].apply(np.std)
correctness_df_gpt = correctness_df_gpt.sort_values(by="Temperature", ascending=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 25))  # Create two subplots

# Plot for df
ax1.bar(correctness_df["Temperature"], correctness_df["Mean_Correctness"], ecolor="black", capsize=8.0, width=0.4,  yerr=correctness_df['Std_Correctness'])
ax1.axhline(y=50, color='royalblue', linestyle='--')
ax1.get_xaxis().set_visible(False)
ax1.tick_params(axis='y', labelsize=35)
ax1.set_ylim(65, 100)
ax1.set_xlabel('Temperature', fontsize=30)  # Set x-axis label font size
ax1.set_ylabel('Validity of Output Format (Mean %)', fontsize=30)  # Set y-axis label font size
ax1.set_title("LLama3-8b-Instruct", fontsize=40)

# Plot for df_gpt
ax2.bar(correctness_df_gpt["Temperature"], correctness_df_gpt["Mean_Correctness"], ecolor="black", capsize=8.0, width=0.4, yerr=correctness_df['Std_Correctness'])
ax2.axhline(y=50, color='royalblue', linestyle='--')
ax2.set_ylim(65, 100)  # Ensure y-axis scaling is the same as the first plot
ax2.tick_params(axis='x', labelsize=35)
ax2.tick_params(axis='y', labelsize=35)
ax2.set_xlabel('Temperature', fontsize=30)  # Set x-axis label font size
ax2.set_ylabel('Validity of Output Format (Mean %)', fontsize=30)  # Set y-axis label font size
ax2.set_title("GPT-4", fontsize=40)


plt.savefig("./temp-variation-correctness-result-cse-cic-ids2018.png")