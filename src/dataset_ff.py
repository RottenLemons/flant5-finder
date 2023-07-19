import pandas as pd
from datasets import *

def make_dataset(split, imsol):
    dataset = load_dataset('reddit',cache_dir="/work/09386/mahirs/ls6/flant5-finder/src/cache") # Negative Samples

    df1 = pd.read_json('InstructionWild/data_v2/user_4.jsonl', lines=True) # Positive Samples
    df2 = pd.read_json('InstructionWild/data_v2/user_1.jsonl', lines=True)
    df2 = df2.drop('instruction_type', axis=1)
    df2 = df2.drop('special_tag', axis=1)
    df = pd.concat([df1, df2], axis=0)
    df.columns = ['text']
    df.reset_index(drop=True)
    df['labels'] = '1'
    dataset2 = Dataset.from_pandas(df)
    dataset2 = dataset2.remove_columns(['__index_level_0__'])
    dataset = dataset['train'].remove_columns(["body", "author", "normalizedBody", "subreddit", "subreddit_id", "id", "summary"])
    dataset = dataset.rename_column("content", "text")
    new_column = ['0'] * len(dataset)
    dataset = dataset.add_column("labels", new_column)
    if imsol == 'down-sample':
        dataset = dataset.shuffle().select(range(len(dataset)//5))
    dataset3 = concatenate_datasets([dataset, dataset2]).shuffle()
    del dataset
    del dataset2
    dataset3 = dataset3.train_test_split(test_size=split)

    return dataset3

