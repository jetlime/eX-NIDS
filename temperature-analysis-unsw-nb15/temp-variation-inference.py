from dotenv import load_dotenv
from os import getenv
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
import torch
import gc
import sys
import random

temperature = float(sys.argv[1])
trial_run = str(sys.argv[2])

load_dotenv()
HUGGING_FACE_READ_TOKEN = getenv("HUGGING_FACE_READ_TOKEN")

dataset = load_dataset("Jetlime/NF-UNSW-NB15-v2", streaming=False, split="test")
dataset = dataset.train_test_split(test_size=0.2, seed=random.randint(0,1000000), stratify_by_column="Attack")
dataset = dataset["test"]
dataset

    
with open('../correctness_prompt_1.txt', 'r') as file:
    instruction_prompt = file.read()

prompts = []
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    trust_remote_code=True,
    gpu_memory_utilization=0.8
)
tokenizer = llm.get_tokenizer()

for i in tqdm(dataset, total=len(dataset)):
    prompt = tokenizer.apply_chat_template([
        {"role": "instruction", "content": instruction_prompt},
        {"role": "input", "content": i["input"]}],
        tokenize=False,
    )
    prompts.append(prompt)
print(f"Temperature: {temperature}")
outputs = llm.generate(
    prompts,
    SamplingParams(
        temperature=temperature,
        top_p=0.9,
        max_tokens=1024,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],  # KEYPOINT HERE
    )
)

predicted_labels = []

for output in outputs:
    generated_text = output.outputs[0].text
    predicted_labels.append(generated_text[-1])

true_labels = []
for i in tqdm(dataset):
    true_labels.append(i["output"])


prediction_labels = [item for item in predicted_labels]

target_names = ['benign', 'malicious']

with open(f'./temp{temperature}-prediction-labels-{trial_run}.txt', 'w') as outfile:
    outfile.write('\n'.join(str(i) for i in prediction_labels))

del llm
del tokenizer
gc.collect()
torch.cuda.empty_cache()

