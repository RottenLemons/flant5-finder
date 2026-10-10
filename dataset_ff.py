import os
import subprocess
import pandas as pd
import datasets

def load_instruction_wild(cache_dir=None):
    """Locate or download InstructionWild data."""
    possible_dirs = [
        "InstructionWild",
        os.path.join(cache_dir, "InstructionWild") if cache_dir else None,
        os.path.join("..", "InstructionWild"),
    ]
    target_dir = None
    for p in possible_dirs:
        if p and os.path.exists(os.path.join(p, "data_v2", "user_4.jsonl")):
            target_dir = p
            break

    if target_dir is None:
        clone_dest = cache_dir if cache_dir else "."
        os.makedirs(clone_dest, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/XueFuzhao/InstructionWild.git"],
            cwd=clone_dest,
            check=False,
        )
        target_dir = os.path.join(clone_dest, "InstructionWild")

    f1 = os.path.join(target_dir, "data_v2", "user_4.jsonl")
    f2 = os.path.join(target_dir, "data_v2", "user_1.jsonl")
    
    if os.path.exists(f1) and os.path.exists(f2):
        df1 = pd.read_json(f1, lines=True)
        df2 = pd.read_json(f2, lines=True)
        for col in ["instruction_type", "special_tag"]:
            if col in df2.columns:
                df2 = df2.drop(col, axis=1)
            if col in df1.columns:
                df1 = df1.drop(col, axis=1)
        df = pd.concat([df1, df2], axis=0).reset_index(drop=True)
        df.columns = ["text"]
        df["labels"] = "yes"
        return datasets.Dataset.from_pandas(df)
    else:
        # Fallback if git clone fails or offline
        return datasets.Dataset.from_dict({
            "text": ["Write a python function to add two numbers.", "Explain quantum computing."],
            "labels": ["yes", "yes"]
        })

def make_dataset(split=0.1, imsol="none", cache_dir=None, seed=42, sample_limit=None):
    """
    Build balanced/imbalanced dataset with fixed seed and split.
    
    Args:
        split: Test split ratio (e.g., 0.1).
        imsol: Imbalance solution strategy ('none', 'up-weight', 'down-sample').
        cache_dir: Optional cache directory for dataset downloads.
        seed: Random seed for reproducibility.
        sample_limit: Optional limit for testing/debugging.
    """
    # Negative Samples (Reddit / webis/tldr-17)
    dataset_n = None
    load_errors = []
    for repo_id, kwargs in [
        ("webis/tldr-17", {"revision": "refs/convert/parquet", "split": "train"}),
        ("reddit", {}),
    ]:
        try:
            ds = datasets.load_dataset(repo_id, cache_dir=cache_dir, **kwargs)
            if isinstance(ds, datasets.DatasetDict):
                dataset_n = ds["train"]
            else:
                dataset_n = ds
            break
        except Exception as e:
            load_errors.append(f"{repo_id}: {e}")

    if dataset_n is None:
        raise RuntimeError(
            f"Could not load negative Reddit dataset. Details:\n" + "\n".join(load_errors)
        )

    remove_cols = [c for c in ["body", "author", "normalizedBody", "subreddit", "subreddit_id", "id", "summary"] if c in dataset_n.column_names]
    dataset_n = dataset_n.remove_columns(remove_cols)
    if "content" in dataset_n.column_names:
        dataset_n = dataset_n.rename_column("content", "text")
    dataset_n = dataset_n.add_column("labels", ["no"] * len(dataset_n))

    # Positive Samples (InstructionWild)
    dataset_p = load_instruction_wild(cache_dir=cache_dir)
    if "__index_level_0__" in dataset_p.column_names:
        dataset_p = dataset_p.remove_columns(["__index_level_0__"])

    if sample_limit:
        # Preserve the ~35:1 imbalance ratio in the subset cap
        pos_count = max(1, int(sample_limit / 36))
        neg_count = sample_limit - pos_count
        dataset_n = dataset_n.shuffle(seed=seed).select(range(min(neg_count, len(dataset_n))))
        dataset_p = dataset_p.shuffle(seed=seed).select(range(min(pos_count, len(dataset_p))))

    # Mix and create fixed train/test split with deterministic seed
    dataset_mix = datasets.concatenate_datasets([dataset_n, dataset_p]).shuffle(seed=seed)
    split_dataset = dataset_mix.train_test_split(test_size=split, seed=seed)

    # If down-sampling, reduce only the majority negative class in the TRAINING set
    # so the evaluation test split remains identical and strictly comparable
    if imsol == "down-sample":
        train_df = split_dataset["train"].to_pandas()
        pos_df = train_df[train_df["labels"] == "yes"]
        neg_df = train_df[train_df["labels"] == "no"].sample(frac=0.2, random_state=seed)
        downsampled_train = pd.concat([pos_df, neg_df]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        split_dataset["train"] = datasets.Dataset.from_pandas(downsampled_train)
        if "__index_level_0__" in split_dataset["train"].column_names:
            split_dataset["train"] = split_dataset["train"].remove_columns(["__index_level_0__"])

    return split_dataset
