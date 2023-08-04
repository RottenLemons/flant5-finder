from transformers import Seq2SeqTrainer, logging, PreTrainedModel
from transformers.models.fsmt.configuration_fsmt import FSMTConfig
import torch

logger = logging.get_logger(__name__)

class DownSampledSeq2SeqTrainer(Seq2SeqTrainer): # Editted trainer with higher loss for negative samples as well as less samples (majority)
    def __init__(self, config=None, data_args=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if config is None:
            assert isinstance(self.model, PreTrainedModel), (
                "If no `config` is passed the model to be trained has to be of type `PreTrainedModel`, but is"
                f" {self.model.__class__}"
            )
            self.config = self.model.config
        else:
            self.config = config

        self.data_args = data_args
        self.vocab_size = self.config.tgt_vocab_size if isinstance(self.config, FSMTConfig) else self.config.vocab_size

        if self.args.label_smoothing != 0 or (self.data_args is not None and self.data_args.ignore_pad_token_for_loss):
            assert self.config.pad_token_id is not None, (
                "Make sure that `config.pad_token_id` is correcly defined when ignoring `pad_token` for loss"
                " calculation or doing label smoothing."
            )

        if self.config.pad_token_id is None and self.config.eos_token_id is not None:
            logger.warning(
                f"The `config.pad_token_id` is `None`. Using `config.eos_token_id` = {self.config.eos_token_id} for"
                " padding.."
            )

        if self.args.label_smoothing == 0:
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id, weight=torch.tensor([1.0, 7.0]))
        else:
            # dynamically import label_smoothed_nll_loss
            from utils import label_smoothed_nll_loss

            self.loss_fn = label_smoothed_nll_loss



