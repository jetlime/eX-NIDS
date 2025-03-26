from openai import OpenAI
from dotenv import load_dotenv
from os import getenv, listdir
from tqdm import tqdm

load_dotenv()
GPT_API_KEY = getenv("GPT_API_KEY")
client = OpenAI(api_key=GPT_API_KEY)


def evaluation():
    file_names = listdir(f"../dataset/augmented-prompt-explainer-evaluation-set/")
    for file_name in tqdm(file_names):
        with open(f'../dataset/augmented-prompt-explainer-evaluation-set/{file_name}', 'r') as file:
            prompt = file.read()
        response = client.chat.completions.create(
            model="gpt-4-0613",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            seed=1234,
            n=1,
            temperature=0.4,
        )

        result = response.choices[0].message.content
        file = open(f'./results-gpt-4-0613/{file_name}', 'w')
        file.write(result)
        file.close()

evaluation()
