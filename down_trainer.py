import os
import sys

os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
if "torchvision" not in sys.modules:
    sys.modules["torchvision"] = None

import torch
from transformers import Seq2SeqTrainer


class DownSampledSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        token_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)).view(labels.size())
        # Custom weights: 7 for down-sampled negative majority ('no' / '0'), 1 for positive
        neg_ids = torch.tensor([150, 3], device=labels.device)
        weights = torch.where(torch.isin(labels, neg_ids), 7.0, 1.0)

        loss = (token_loss * weights).sum() / weights[labels != -100].sum().clamp(min=1)
        return (loss, outputs) if return_outputs else loss
