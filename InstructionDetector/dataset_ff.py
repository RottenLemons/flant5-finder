import pandas as pd
import datasets
import subprocess

def make_dataset(split, imsol, cache_dir):
    
     # Negative Samples
    dataset_n = datasets.load_dataset('reddit',cache_dir=cache_dir) # todo change this to work/data directory when running on server
    dataset_n = dataset_n['train'].remove_columns(["body", "author", "normalizedBody", "subreddit", "subreddit_id", "id", "summary"])
    dataset_n = dataset_n.rename_column("content", "text")
    new_column = ['no'] * len(dataset_n)
    dataset_n = dataset_n.add_column("labels", new_column)
    dataset_n = dataset_n.shuffle().select(range(20))  # todo remove this
    if imsol == 'down-sample':
        dataset_n = dataset_n.shuffle().select(range(len(dataset_n)//5))

    # Positive Samples
    subprocess.run(f"cd {cache_dir}; git clone https://github.com/XueFuzhao/InstructionWild.git; cd ../flant5-finder".split()) 
    df1 = pd.read_json(f'{cache_dir}/InstructionWild/data_v2/user_4.jsonl', lines=True) 
    df2 = pd.read_json(f'{cache_dir}/InstructionWild/data_v2/user_1.jsonl', lines=True)
    df2 = df2.drop('instruction_type', axis=1)
    df2 = df2.drop('special_tag', axis=1)
    df = pd.concat([df1, df2], axis=0)
    df.columns = ['text']
    df.reset_index(drop=True)
    df['labels'] = 'yes'
    dataset_p = datasets.Dataset.from_pandas(df)
    dataset_p = dataset_p.remove_columns(['__index_level_0__'])
    dataset_p = dataset_p.shuffle().select(range(20))  # todo remove this
    
    # mix&shuffle the positive and negative samples
    dataset_mix = datasets.concatenate_datasets([dataset_n, dataset_p]).shuffle()
    dataset_mix = dataset_mix.train_test_split(test_size=split)
    

    return dataset_mix

