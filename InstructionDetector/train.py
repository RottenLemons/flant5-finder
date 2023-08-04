import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import glob
import datasets
import argparse
import dataset_ff
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from huggingface_hub import login
import wandb
from up_trainer import UpWeightedSeq2SeqTrainer
from down_trainer import DownSampledSeq2SeqTrainer
from sophia import SophiaG
from tqdm import tqdm
from torch.utils.data import DataLoader

nltk.download("punkt")
metric = evaluate.load("f1",cache_dir="/work/09386/mahirs/ls6/flant5-finder/src/cache")
torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--imsol', type=str, required=False, default='none', choices=['none', 'up-weight', 'down-sample'], help='Imbalance Solution Allowed values: none, up-weight, down-sample')
parser.add_argument('--split', type=float, required=False, default=0.1, help='Test Split Percentage')
parser.add_argument('--epochs', type=int, required=False, default=3, help='Number of epochs')
parser.add_argument('--hftoken', type=str, required=True, help='HuggingFace Token')
parser.add_argument('--hfname', type=str, required=False, default='flant5-finder', help='HuggingFace Model Name')
parser.add_argument('--wname', type=str, required=False, default='flant5-finder',help='W&B Project Name')

arg = parser.parse_args()

login(token=arg.hftoken)

def main():
    
    # load data and model
    dataset = dataset_ff.make_dataset(split=0.1, imsol=arg.imsol)
    model_id="google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    # collect data length
    tokenized_inputs = datasets.concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["text"], truncation=True), 
                                                                                     batched=True, remove_columns=['text', 'labels'])
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    min_source_length = min([len(x) for x in tokenized_inputs["input_ids"]])
    print(f"Max source length: {max_source_length}, Min source length: {min_source_length}, Total data entries: {tokenized_inputs['input_ids']}")

    max_target_length = 3
    del tokenized_inputs


    def preprocess_function(sample, padding="max_length"):
        prompt = "Is the following sentence an instruction? Answer yes or no: "
        
        inputs = [prompt+item for item in sample["text"]]

        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        labels = tokenizer(sample["labels"], max_length=max_target_length, padding=padding, truncation=True)

        # if padding == "max_length":
        #     labels["input_ids"] = [
        #         [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        #     ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['text', 'labels'])

    def postprocess_text(preds, labels): # Convert output to text
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(preds, labels):
        
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, average='macro')
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    ) # Pad to reach max-len

    repository_id = arg.hfname

    optimizer = SophiaG(model.parameters(), lr=3e-4)

    train_dl = DataLoader(
        tokenized_dataset['train'], shuffle=True, batch_size=128, collate_fn=data_collator
    )

    test_dl = DataLoader(
        tokenized_dataset['test'], shuffle=True, batch_size=128, collate_fn=data_collator
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        fp16=True,
        learning_rate=3e-4,

        num_train_epochs=arg.epochs,
        logging_dir=f"{repository_id}/logs",
        logging_strategy="epoch",
        evaluation_strategy="no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        push_to_hub=True,
        hub_strategy="every_save",
        hub_model_id=repository_id,
        report_to="wandb",
        run_name=arg.wname,
    )

    epochs = 1
    training_steps = epochs * len(train_dl)

    if arg.imsol == 'none':
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            compute_metrics=compute_metrics,
    )
        # loss_fn = torch.nn.CrossEntropyLoss(ignore_index=label_pad_token_id)


    elif arg.imsol == 'up-weight':
        #     trainer = UpWeightedSeq2SeqTrainer(
        #     model=model,
        #     args=training_args,
        #     data_collator=data_collator,
        #     train_dataset=tokenized_dataset["train"],
        #     eval_dataset=tokenized_dataset["test"],
        #     compute_metrics=compute_metrics,
        # )
    else:
        # trainer = DownSampledSeq2SeqTrainer(
        #     model=model,
        #     args=training_args,
        #     data_collator=data_collator,
        #     train_dataset=tokenized_dataset["train"],
        #     eval_dataset=tokenized_dataset["test"],
        #     compute_metrics=compute_metrics,
        # )

    # trainer.train()
    # trainer.evaluate()
    # trainer.save_model(f"{repository_id}/model")
    # trainer.create_model_card()
    # trainer.push_to_hub()

    # progress_bar = tqdm(range(training_steps))
    # model = torch.nn.DataParallel(model).to(device)
    # model.train()
    # for epoch in range(epochs):
    #     for batch in train_dl:
    #         # Run a batch through the model
    #         outputs = model.genrate(**batch.to(device))
    #         loss = loss_fn(outputs, batch)
    #         loss.backward()

    #         optimizer.step()
    #         optimizer.zero_grad()
    #         progress_bar.update(1)

    #         if progress_bar.n % 100 == 0:
    #             print(f"F1: {loss.item()}")

if __name__ == "__main__":
    main()