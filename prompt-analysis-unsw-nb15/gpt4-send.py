from openai import OpenAI
from dotenv import load_dotenv
from os import getenv

load_dotenv()
GPT_API_KEY = getenv("GPT_API_KEY")

client = OpenAI(api_key=GPT_API_KEY)
prompt_ids = {}

for i in range(1,5):
  batch_input_file = client.files.create(
    file=open(f"prompts-gpt-4-prompt-{str(i)}.jsonl", "rb"),
    purpose="batch"
  )

  batch_input_file_id = batch_input_file.id

  response = client.batches.create(
      input_file_id=batch_input_file_id,
      endpoint="/v1/chat/completions",
      completion_window="24h"
  )

  batch_id = response.id
  
  prompt_ids[i] = batch_id

print(prompt_ids)