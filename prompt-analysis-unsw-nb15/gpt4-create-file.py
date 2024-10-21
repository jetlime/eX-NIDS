from dotenv import load_dotenv
from os import getenv
from tqdm import tqdm
from datasets import load_dataset
import json


load_dotenv()
HUGGING_FACE_READ_TOKEN = getenv("HUGGING_FACE_READ_TOKEN")

random_seed =1234

dataset = load_dataset("Jetlime/NF-UNSW-NB15-v2", streaming=False, split="test")
dataset = dataset.train_test_split(test_size=0.2, seed=random_seed, stratify_by_column="Attack")
dataset = dataset["test"]
dataset

for prompt_id in range(1, 5):
  prompts = []
  
  with open(f'../perf_prompt_{prompt_id}.txt', 'r') as file:
      instruction_prompt = file.read()
  index = 1
  for i in tqdm(dataset, total=len(dataset)):
      
      prompt = {}
      prompt["custom_id"] = f"request-{index}"
      prompt["method"] = "POST"
      prompt["url"] = "/v1/chat/completions"
      prompt["body"] = {"model": "gpt-4o-2024-05-13", "messages":[{"role": "system", "content": instruction_prompt},{"role": "user", "content": i['input']}],"max_tokens": 1, 'temperature': 0.01}

      prompts.append(prompt)
      index += 1

  with open(f"prompts-gpt-4-prompt-{prompt_id}.jsonl", 'w') as f:
      for item in prompts:
          f.write(json.dumps(item) + "\n")