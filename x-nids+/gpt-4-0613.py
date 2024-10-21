from openai import OpenAI
from dotenv import load_dotenv
from os import getenv, listdir
from tqdm import tqdm

load_dotenv()
GPT_API_KEY = getenv("GPT_API_KEY")

client = OpenAI(api_key="")


def evaluation(dataset_folder):
    file_names = listdir(f"./evaluation-set-{dataset_folder}")
    for file_name in tqdm(file_names):
        with open(f'./evaluation-set-{dataset_folder}/{file_name}', 'r') as file:
            prompt = file.read()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            seed=1234,
            n=1,
            temperature=0.4,
        )

        result = response.choices[0].message.content
        file = open(f'./results-gpt-4-0613-{dataset_folder}/{file_name}', 'w')
        file.write(result)
        file.close()

evaluation("cse-cic-ids2018")
