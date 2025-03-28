from tqdm import tqdm
from os import listdir
from openai import OpenAI

def evaluation():
    file_names = listdir(f"../dataset/basic-prompt-explainer-evaluation-set/")
    for file_name in tqdm(file_names):
        with open(f"../dataset/basic-prompt-explainer-evaluation-set/{file_name}", "r") as file:
            user_prompt = file.read()

        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        
        completion = client.chat.completions.create(
            model="lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF",
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )

        result = completion.choices[0].message.content

        file = open(f"./results-Llama-3-70B-Instruct-{dataset_folder}/{file_name}", "w")
        file.write(result)
        file.close()

evaluation()