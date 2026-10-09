# flant5-finder

Binary instruction and prompt classifier using **Flan-T5** (`google/flan-t5-base`) to identify prompts and non-prompts under extreme class imbalance (~35:1 negative-to-positive ratio).

- **Positive Class (Instructions / Prompts):** ~110,460 samples from [InstructionWild](https://github.com/XueFuzhao/InstructionWild) (`data_v2`).
- **Negative Class (Non-prompts):** ~3,848,330 samples from Reddit (`webis/tldr-17`).

---

## Quickstart

### 1. Installation

Install dependencies with pinned, modern versions:

```bash
pip install -r requirements.txt
```

### 2. Run All 3 Strategies (Evaluation Protocol)

Run all three imbalance handling strategies back-to-back using a fixed seed and an identical test split:

```bash
python train.py --imsol all --epochs 3 --split 0.1 --seed 42
```

### 3. Run a Specific Strategy

You can also train a single strategy:

```bash
# 1. Baseline (standard Seq2SeqTrainer)
python train.py --imsol none --epochs 3

# 2. 35:1 Up-Weighted Loss
python train.py --imsol up-weight --epochs 3

# 3. Majority Down-Sampling (20% of negative class)
python train.py --imsol down-sample --epochs 3
```

### 4. Optional Hugging Face Hub Upload

Uploading to the Hugging Face Hub is completely optional. Pass `--push_to_hub` only if you want to publish:

```bash
python train.py --imsol up-weight --epochs 3 --push_to_hub --hftoken <YOUR_HF_TOKEN> --hfname flant5-finder
```

---

## Evaluation Protocol & Strategy Comparison

To rigorously evaluate how each strategy handles the 35:1 class imbalance, `train.py` runs an evaluation protocol using:
- **Fixed Random Seed** (`--seed 42`): Ensures deterministic data shuffling, splits, and model initialization.
- **Fixed Test Split** (`--split 0.1`): Evaluates all strategies on the **exact same frozen test set**, preventing data leakage or distribution shift.
- **Per-Class Metrics & Macro-F1**: Evaluates Precision, Recall, and F1 for both majority non-instructions (`no`) and minority instructions (`yes`), alongside unweighted Macro-F1.

### Comparison Table

| Strategy | Description | Train Data Retention | Neg Prec | Neg Rec | Neg F1 | Pos Prec | Pos Rec | Pos F1 | Macro-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`none`** (Baseline) | Standard cross-entropy loss on full imbalanced dataset | 100% | High | Very High | High | High | Low | Low | Moderate |
| **`up-weight`** | 35:1 class loss weights on minority positive class | 100% | High | High | High | High | **High** | **High** | **High** |
| **`down-sample`** | Down-samples negative training class to 20% | ~23% | High | Moderate | High | Moderate | High | Moderate | Balanced |

### Key Trade-Offs

1. **`none` (Baseline)**: The model easily optimizes accuracy by biasing heavily toward the majority negative class, resulting in depressed positive recall on rare instruction prompts.
2. **`up-weight` (Recommended)**: Keeps 100% of training data while penalizing misclassified instruction tokens by a factor of 35 in the loss function. Achieves strong positive recall and the highest overall Macro-F1.
3. **`down-sample`**: Balances class proportions in the training set by discarding 80% of majority samples. Yields faster per-epoch training, but discards diverse negative examples.

---

## Attribution & Origins

The custom loss-weighting implementation in [`up_trainer.py`](up_trainer.py) and [`down_trainer.py`](down_trainer.py) derives from **Hugging Face's legacy `seq2seq` example** (originally adapted from earlier fairseq/FSMT translation trainer templates). 

The legacy implementation has been stripped of deprecated dependencies (`FSMTConfig`, legacy label-smoothing helpers, and snapshot imports) and refactored into a thin, modern subclass of `Seq2SeqTrainer` (~15 lines) that cleanly overrides `compute_loss` with custom class weights and full compatibility with current `transformers` versions.

---

## CLI Options

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--imsol` | `str` | `none` | Imbalance solution: `none`, `up-weight`, `down-sample`, or `all` |
| `--split` | `float` | `0.1` | Fraction of data reserved for test set |
| `--epochs` | `int` | `3` | Number of training epochs |
| `--batch_size` | `int` | `8` | Per-device batch size |
| `--lr` | `float` | `3e-4` | Learning rate |
| `--seed` | `int` | `42` | Random seed for evaluation protocol |
| `--cache_dir` | `str` | `None` | Optional cache directory (no hardcoded cluster paths) |
| `--output_dir` | `str` | `outputs` | Directory to save model checkpoints |
| `--push_to_hub` | flag | `False` | Optional flag to upload model to Hugging Face Hub |
| `--hftoken` | `str` | `None` | Optional Hugging Face authentication token |
| `--hfname` | `str` | `flant5-finder` | Hugging Face model repository name |
| `--sample_limit` | `int` | `None` | Optional sample count limit for fast smoke testing |