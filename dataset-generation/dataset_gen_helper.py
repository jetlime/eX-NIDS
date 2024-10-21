import sys
import datasets
import dask.dataframe as dd 
from dask.diagnostics import ProgressBar
from tqdm import tqdm

tqdm.pandas()

columns = []

# Merge all column entries as one single string using key-value pairs
def create_key_value_pairs_str(row):
    return ', '.join([f"{column}: {row[column]}" for column in columns])


def operation(df):
    global columns
    # Obtain the output label which shall be predicted by the LLM
    df = df.rename(columns={'Label': 'output'})

    # Merge all remaining columns
    print("Merging all columns in parralel.")
    columns = df.columns.drop("output")
    columns = columns.drop("Attack")
    df['input'] = df.apply(create_key_value_pairs_str, axis=1)

    return df[['input', 'output', 'Attack']]

# Helper Function for all datasets
def encode_dataset(DATASET_NAME):
    print(f"Opening ../data_raw/{DATASET_NAME}.csv")
    ProgressBar().register()
    df = dd.read_csv(f"../data_raw/{DATASET_NAME}.csv")
    ProgressBar().register()
    df = df.pipe(operation)
    df.to_parquet(f'../data_raw/{DATASET_NAME}-processed')


def process_arrow_data(DATASET_NAME):
    dataset = datasets.Dataset.from_parquet(f'../data_raw/{DATASET_NAME}-processed/*')
    print(dataset)
    dataset = dataset.class_encode_column("output")
    dataset = dataset.class_encode_column("Attack")
    dataset = dataset.train_test_split(test_size=0.05, seed=123, stratify_by_column="Attack")
    dataset.save_to_disk(f"./{DATASET_NAME}/")

if __name__ == "__main__":
    DATASET_NAME = sys.argv[1]
    encode_dataset(DATASET_NAME)
    process_arrow_data(DATASET_NAME)
