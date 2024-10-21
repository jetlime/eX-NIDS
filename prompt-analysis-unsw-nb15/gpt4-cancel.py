from openai import OpenAI
from dotenv import load_dotenv
from os import getenv

prompt_ids = {1: 'batch_4k2VSRtq4A38s5RljBYNPM47'}

load_dotenv()
GPT_API_KEY = getenv("GPT_API_KEY")

client = OpenAI(api_key=GPT_API_KEY)

for key,value in prompt_ids.items():

  client.batches.cancel(value)