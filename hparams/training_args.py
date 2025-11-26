from transformers import Seq2SeqTrainingArguments
from dataclasses import dataclass, field

@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    r"""
    Arguments pertaining to the trainer.
    """

    def __post_init__(self):
        Seq2SeqTrainingArguments.__post_init__(self)