import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainer, TrainingArguments,HfArgumentParser
from datasets import load_from_disk, Dataset
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from hparams import ModelArguments,TrainingArguments,DataArguments,LoraArguments,HyperAdapterArguments
from datasets import load_dataset
import torch
from utils import find_all_linear_modules_wo_hyper,is_hypernetwork_module,print_trainable_parameters
from extras.logging import get_logger
import os
from peft import get_peft_model, LoraConfig, TaskType, PeftModel



logger = get_logger(__name__)

parser = HfArgumentParser( (ModelArguments,TrainingArguments,DataArguments,LoraArguments,HyperAdapterArguments))

model_args,training_args,data_args,lora_args,hyper_args = parser.parse_args_into_dataclasses()






pretrain_model_name_or_path = model_args.model_name_or_path

save_model_path = model_args.new_structural_model_path




model = AutoModelForCausalLM.from_pretrained(pretrain_model_name_or_path,torch_dtype=torch.bfloat16, hyper_config=hyper_args)
tokenizer=AutoTokenizer.from_pretrained(pretrain_model_name_or_path)


tokenizer.save_pretrained(save_model_path)


model.save_pretrained(save_model_path,safe_serialization=False)



print_trainable_parameters(model)









