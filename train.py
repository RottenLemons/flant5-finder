import os
import sys

# Suppress torchvision to prevent C++ operator mismatch crashes in Kaggle/Colab
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
sys.modules["torchvision"] = None
sys.modules["torchvision.io"] = None
sys.modules["torchvision.ops"] = None
sys.modules["torchvision.transforms"] = None

import argparse
import logging
import random
import numpy as np
import pandas as pd
import torch
import datasets
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed as hf_set_seed,
)

import dataset_ff
from up_trainer import UpWeightedSeq2SeqTrainer
from down_trainer import DownSampledSeq2SeqTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")


def set_seed(seed: int = 42):
    """Ensure fully reproducible splits and model initialization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def build_compute_metrics(tokenizer):
    """Compute per-class precision/recall/F1 and macro-F1."""
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        cleaned_preds = [p.strip().lower() for p in decoded_preds]
        cleaned_labels = [l.strip().lower() for l in decoded_labels]

        y_true = [1 if ("yes" in l or l == "1") else 0 for l in cleaned_labels]
        y_pred = [1 if ("yes" in p or p == "1") else 0 for p in cleaned_preds]

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], zero_division=0
        )
        macro_f1 = f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        gen_lens = [np.count_nonzero(p != tokenizer.pad_token_id) for p in preds]

        logger.info(
            f"Eval breakdown - True counts: [Neg={y_true.count(0)}, Pos={y_true.count(1)}], "
            f"Pred counts: [Neg={y_pred.count(0)}, Pos={y_pred.count(1)}]"
        )

        return {
            "neg_precision": round(float(precision[0]) * 100, 2),
            "neg_recall": round(float(recall[0]) * 100, 2),
            "neg_f1": round(float(f1[0]) * 100, 2),
            "pos_precision": round(float(precision[1]) * 100, 2),
            "pos_recall": round(float(recall[1]) * 100, 2),
            "pos_f1": round(float(f1[1]) * 100, 2),
            "macro_f1": round(float(macro_f1) * 100, 2),
            "accuracy": round(float(acc) * 100, 2),
            "gen_len": round(float(np.mean(gen_lens)), 2),
        }
    return compute_metrics


def train_and_eval_strategy(strategy, tokenized_train, tokenized_test, tokenizer, data_collator, args):
    """Train and evaluate a specific class imbalance handling strategy."""
    logger.info(f"========== Running Strategy: {strategy} ==========")
    output_dir = os.path.join(args.output_dir, f"{args.hfname}-{strategy}")

    # Load fresh model weights for fair comparison
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id, cache_dir=args.cache_dir)

    training_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "predict_with_generate": True,
        "fp16": torch.cuda.is_available(),
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "logging_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": False,
        "push_to_hub": args.push_to_hub,
        "report_to": "none",
        "seed": args.seed,
    }

    if args.push_to_hub:
        training_kwargs["hub_strategy"] = "every_save"
        training_kwargs["hub_model_id"] = f"{args.hfname}-{strategy}"

    # Handle eval_strategy / evaluation_strategy compatibility
    try:
        training_kwargs["eval_strategy"] = "epoch"
        training_args = Seq2SeqTrainingArguments(**training_kwargs)
    except TypeError:
        training_kwargs.pop("eval_strategy", None)
        training_kwargs["evaluation_strategy"] = "epoch"
        training_args = Seq2SeqTrainingArguments(**training_kwargs)

    compute_metrics_fn = build_compute_metrics(tokenizer)

    if strategy == "none":
        trainer_cls = Seq2SeqTrainer
    elif strategy == "up-weight":
        trainer_cls = UpWeightedSeq2SeqTrainer
    elif strategy == "down-sample":
        trainer_cls = DownSampledSeq2SeqTrainer
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    trainer = trainer_cls(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics_fn,
        tokenizer=tokenizer,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    logger.info(f"Results for {strategy}: {eval_results}")

    trainer.save_model(os.path.join(output_dir, "model"))
    if args.push_to_hub:
        trainer.create_model_card()
        trainer.push_to_hub()

    return eval_results


def print_comparison_table(results_dict, seed, split):
    """Format and print a comparison table across evaluated strategies."""
    header = (
        f"\n{'='*105}\n"
        f"Evaluation Protocol Results (Seed: {seed}, Test Split: {split})\n"
        f"{'='*105}\n"
        f"{'Strategy':<14} | {'Neg Prec (%)':<12} | {'Neg Rec (%)':<11} | {'Neg F1 (%)':<10} | "
        f"{'Pos Prec (%)':<12} | {'Pos Rec (%)':<11} | {'Pos F1 (%)':<10} | {'Macro-F1 (%)':<12}\n"
        f"{'-'*14}-+-{'-'*12}-+-{'-'*11}-+-{'-'*10}-+-{'-'*12}-+-{'-'*11}-+-{'-'*10}-+-{'-'*12}"
    )
    print(header)
    for strat, res in results_dict.items():
        np_p = res.get("eval_neg_precision", 0.0)
        np_r = res.get("eval_neg_recall", 0.0)
        np_f = res.get("eval_neg_f1", 0.0)
        pos_p = res.get("eval_pos_precision", 0.0)
        pos_r = res.get("eval_pos_recall", 0.0)
        pos_f = res.get("eval_pos_f1", 0.0)
        macro_f1 = res.get("eval_macro_f1", 0.0)
        print(
            f"{strat:<14} | {np_p:<12.2f} | {np_r:<11.2f} | {np_f:<10.2f} | "
            f"{pos_p:<12.2f} | {pos_r:<11.2f} | {pos_f:<10.2f} | {macro_f1:<12.2f}"
        )
    print("=" * 105 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Flan-T5 Prompt/Instruction Classifier Training & Evaluation")
    parser.add_argument(
        "--imsol",
        type=str,
        default="none",
        choices=["none", "up-weight", "down-sample", "all"],
        help="Imbalance Solution: none, up-weight, down-sample, or all (runs all 3 strategies back-to-back)",
    )
    parser.add_argument("--split", type=float, default=0.1, help="Test split ratio (e.g. 0.1)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed for evaluation protocol")
    parser.add_argument("--model_id", type=str, default="google/flan-t5-base", help="Model name or path")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory (optional)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for checkpoints")
    parser.add_argument("--hfname", type=str, default="flant5-finder", help="HuggingFace model repository name")
    parser.add_argument("--push_to_hub", action="store_true", help="Optional flag to upload trained model to HF Hub")
    parser.add_argument("--hftoken", type=str, default=None, help="HuggingFace token for optional hub upload")
    parser.add_argument("--sample_limit", type=int, default=None, help="Optional subset limit for fast debugging")

    args = parser.parse_args()

    # Reproducibility
    set_seed(args.seed)

    # Optional HF Login
    if args.push_to_hub:
        if args.hftoken:
            from huggingface_hub import login
            login(token=args.hftoken)
        else:
            logger.info("Using cached Hugging Face credentials for hub upload.")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)

    # Load Full Dataset with fixed seed and split
    raw_dataset = dataset_ff.make_dataset(
        split=args.split,
        imsol="none",
        cache_dir=args.cache_dir,
        seed=args.seed,
        sample_limit=args.sample_limit,
    )

    def preprocess_function(sample, padding="max_length"):
        prompt = "[Question] Is the following sentence an instruction? Options: - yes - no [Answer] "
        inputs = [prompt + item for item in sample["text"]]
        model_inputs = tokenizer(inputs, max_length=512, padding=padding, truncation=True)
        labels = tokenizer(text_target=sample["labels"], max_length=3, padding=padding, truncation=True)

        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_full = raw_dataset.map(preprocess_function, batched=True, remove_columns=["text", "labels"])
    fixed_test_dataset = tokenized_full["test"]

    # Also prepare down-sampled training dataset if needed
    strategies = ["none", "up-weight", "down-sample"] if args.imsol == "all" else [args.imsol]
    tokenized_train_dict = {
        "none": tokenized_full["train"],
        "up-weight": tokenized_full["train"],
    }

    if "down-sample" in strategies:
        down_raw = dataset_ff.make_dataset(
            split=args.split,
            imsol="down-sample",
            cache_dir=args.cache_dir,
            seed=args.seed,
            sample_limit=args.sample_limit,
        )
        down_tokenized = down_raw["train"].map(preprocess_function, batched=True, remove_columns=["text", "labels"])
        tokenized_train_dict["down-sample"] = down_tokenized

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    results = {}
    for strat in strategies:
        train_data = tokenized_train_dict[strat]
        res = train_and_eval_strategy(
            strategy=strat,
            tokenized_train=train_data,
            tokenized_test=fixed_test_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            args=args,
        )
        results[strat] = res

    # Print comparison table across all evaluated strategies
    print_comparison_table(results, seed=args.seed, split=args.split)


if __name__ == "__main__":
    main()