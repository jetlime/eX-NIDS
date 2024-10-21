from openai import OpenAI
from dotenv import load_dotenv
from os import getenv

# prompt_ids = {1: 'batch_skz2JTMj2DmlmkzFS9Y3iUHF', 2: 'batch_NDnOl4QsFwhVmAgQTzD2KQtI', 3: 'batch_WJPxxaWypQBsGcL2j2bVWtLE', 4: 'batch_4k2VSRtq4A38s5RljBYNPM47'}
prompt_ids = {4: 'batch_4k2VSRtq4A38s5RljBYNPM47'}

load_dotenv()
GPT_API_KEY = getenv("GPT_API_KEY")

client = OpenAI(api_key=GPT_API_KEY)

for key,value in prompt_ids.items():

  response = client.batches.retrieve(value)

  if response.status == "completed" or response.status == "cancelled":
    print(response)
    file_response = client.files.content(response.output_file_id).content
    result_file_name = f"./gpt-4-results-prompt-{key}.jsonl"

    with open(result_file_name, 'wb') as file:
        file.write(file_response)
  else:
    print(response)