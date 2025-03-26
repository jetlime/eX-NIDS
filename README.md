# eX-NIDS: A Framework for Explainable Network Intrusion Detection Leveraging Large Language Models

## Dataset of NetFlow samples
The set of malicious NetFlow samples we consider in our experiments, taken from the NF-CSE-CIC-IDS2018 dataset, are located in the ``./dataset`` folder. For the basic and augmented prompt approach, there is a subfolder, containing 50 files each. Each file contains one distinct malicious NetFlow along with the complete input prompt.

## Basic-Prompt Explainer
The ``./basic-prompt-explainer`` folder contains the experiments performed with the basic prompt. The output result, for each input file, is located in a subfolder with the prefix ``results-``, followed by the model ID who generated the output. The python scripts ``gpt-4-0613.py`` and ``llama3-70b-instruct.py``, respectively send the input files to the model and generate the output results.

## Augmented-Prompt Explainer
The ``./augmented-prompt-explainer`` folder contains the experiments performed with the augmented prompt. The output result, for each input file, is located in a subfolder with the prefix ``results-``, followed by the model ID who generated the output. The python scripts ``gpt-4-0613.py`` and ``llama3-70b-instruct.py``, respectively send the input files to the model and generate the output results. 

> For evaluating the GPT model we rely on the openAI API, which requires an API key to be defined as the GPT_API_KEY environement variable. The LLama model is self-hosted using the LM-Studio software.