import torch
from transformers import Seq2SeqTrainer


class UpWeightedSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        token_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)).view(labels.size())
        # 35:1 class weights: 35 for minority positive ('yes' / '1'), 1 for majority negative
        pos_ids = torch.tensor([4273, 209], device=labels.device)
        weights = torch.where(torch.isin(labels, pos_ids), 35.0, 1.0)

        loss = (token_loss * weights).sum() / weights[labels != -100].sum().clamp(min=1)
        return (loss, outputs) if return_outputs else loss
