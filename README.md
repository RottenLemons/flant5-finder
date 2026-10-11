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

*(Note for Kaggle/Colab: run `pip uninstall -y torchvision` if pre-installed to avoid C++ binary operator conflicts).*

### 2. Run All 3 Strategies (Evaluation Protocol)

Run all three imbalance handling strategies back-to-back using a fixed seed, an identical test split, and an optional subset cap:

```bash
# Full evaluation protocol on 10k samples (fast, ~12-15 min total on T4 GPU)
python train.py --imsol all --epochs 3 --sample_limit 10000 --seed 42 --split 0.2
```

### 3. Run a Specific Strategy

You can also train a single strategy:

```bash
# 1. Baseline (standard Seq2SeqTrainer)
python train.py --imsol none --epochs 3 --sample_limit 10000

# 2. 35:1 Up-Weighted Loss (Recommended)
python train.py --imsol up-weight --epochs 3 --sample_limit 10000

# 3. Majority Down-Sampling (20% of negative class)
python train.py --imsol down-sample --epochs 3 --sample_limit 10000
```

### 4. Optional Hugging Face Hub Upload

Uploading to the Hugging Face Hub is completely optional. Pass `--push_to_hub` only if you want to publish:

```bash
python train.py --imsol up-weight --epochs 3 --push_to_hub --hftoken <YOUR_HF_TOKEN> --hfname flant5-finder
```

---

## Evaluation Protocol & Results

To rigorously compare the three strategies under identical conditions:
- **Fixed Random Seed (`--seed 42`)**: Guarantees deterministic data shuffling, splits, and fresh model initializations.
- **Fixed Test Split (`--split 0.2`)**: Evaluates all strategies on the **exact same frozen test set**, preventing data leakage or distribution shifts.
- **Per-Class Metrics & Macro-F1**: Evaluates Precision, Recall, and F1 for both majority non-instructions (`no`) and minority instructions (`yes`), alongside unweighted Macro-F1.

### Benchmark Results

| Strategy | Train Data Retention | Neg Prec (%) | Neg Rec (%) | Neg F1 (%) | Pos Prec (%) | Pos Rec (%) | Pos F1 (%) | Macro-F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`none`** (Baseline) | 100% | 99.79 | 99.43 | 99.61 | 85.14 | 94.03 | 89.36 | 94.49 |
| **`up-weight`** (**Best**) | **100%** | **99.90** | **99.74** | **99.82** | **92.86** | **97.01** | **94.89** | **97.35** |
| **`down-sample`** | ~23% | 99.84 | 99.28 | 99.56 | 82.05 | 95.52 | 88.28 | 93.92 |

---

## Why `up-weight` is the best:

**Why it beats `none` (Baseline)**:
In the unweighted baseline, the 35:1 negative majority overwhelms the gradient updates. The loss function barely penalizes misclassifying rare instructions because predicting "no" is safe 97% of the time. This causes the baseline model to hesitate on ambiguous prompts, dragging down positive precision to 85.14%. 35:1 loss-weighting equalizes the gradient influence of rare prompts without changing the data.

**Why it beats `down-sample`**:
Down-sampling discards ~80% of majority Reddit text during training. While this balances class ratios, it destroys negative vocabulary diversity. Lacking exposure to varied conversational text patterns, the down-sampled model suffers from false alarms on unseen Reddit text, causing its Positive Precision to collapse to 82.05% (worst of all three strategies).

---

## Attribution & Origins

The custom loss-weighting implementation in [`up_trainer.py`](up_trainer.py) and [`down_trainer.py`](down_trainer.py) derives from **Hugging Face's legacy `seq2seq` example** (originally adapted from earlier fairseq/FSMT translation trainer templates). 

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
| `--sample_limit` | `int` | `None` | Cap total samples while preserving the 35:1 ratio (e.g. `10000`) |
| `--cache_dir` | `str` | `None` | Optional cache directory (no hardcoded cluster paths) |
| `--output_dir` | `str` | `outputs` | Directory to save model checkpoints |
| `--push_to_hub` | flag | `False` | Optional flag to upload model to Hugging Face Hub |
| `--hftoken` | `str` | `None` | Optional Hugging Face authentication token |
| `--hfname` | `str` | `flant5-finder` | Hugging Face model repository name |