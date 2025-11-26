from transformers import AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainer, TrainingArguments,HfArgumentParser
from datasets import load_from_disk, Dataset
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from hparams import ModelArguments,TrainingArguments,DataArguments,LoraArguments,HyperAdapterArguments
from datasets import load_dataset
import torch
from utils import find_all_linear_modules_wo_hyper,is_hypernetwork_module,print_trainable_parameters,find_all_linear_modules
from extras.logging import get_logger
import os
import numpy as np
import h5py





logger = get_logger(__name__)

parser = HfArgumentParser( (ModelArguments,TrainingArguments,DataArguments,LoraArguments,HyperAdapterArguments))

model_args,training_args,data_args,lora_args,hyper_args = parser.parse_args_into_dataclasses()


print(data_args.train_dataset)
print(model_args.model_name_or_path)


tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
IGNORE_INDEX = -100


img_feature_path = hyper_args.img_path


def Sample_Preprocess_function(examples):

    lang_list="english,russian,indonesian,urdu"
    per_lang_pair_batch_size = hyper_args.per_lang_pair_batch_size

    lang_dic={key: index for index, key in enumerate(lang_list.split(","))}


    inputs = []
    labels = []
    attention_masks = []
    lang_pair = []

    #add img
    img_file_ids =  [sample.strip().split("\t\t")[2].split("\t")[:3] for sample in examples["text"]]
    
    for sample in examples["text"]:
        
        source_text,target_text,_ = sample.strip().split("\t\t")


        src_lang,tgt_lang = source_text.strip().split(" : ")[0].split(" - ")

        lang_pair.append(source_text.strip().split(" : ")[0])

        src_text = source_text.strip().split(" : ")[1]

        instruct = f"Summarize the following {src_lang} text into a {tgt_lang} abstract: "

        src_text = instruct + src_text 
        


        tgt_text = "\nAnswer:" + target_text

        

        src_text_ids = tokenizer.encode(
                        src_text,
                        truncation=True,
                        max_length=data_args.cutoff_len,
                        )

    
        source_len = len(src_text_ids)+3
        
       

        src_wtih_tgt_ids = tokenizer.encode(
                            tgt_text,
                            truncation=True,
                            add_special_tokens=False,
                            max_length=128,
                         )

        
        
        src_wtih_tgt_ids = src_text_ids+src_wtih_tgt_ids


        src_wtih_tgt_length = len(src_wtih_tgt_ids)


        label_ids = [IGNORE_INDEX]*source_len +src_wtih_tgt_ids[source_len:]
        

        src_wtih_tgt_ids = src_wtih_tgt_ids + [tokenizer.eos_token_id]
        label_ids = label_ids + [tokenizer.eos_token_id]

        attention_mask = [1]*(src_wtih_tgt_length+1)

        inputs.append(src_wtih_tgt_ids)
        labels.append(label_ids)
        attention_masks.append(attention_mask)
    

    assert len(img_file_ids)==len(inputs)




    inputs_dic = []
    

    for i in range(0, len(lang_pair), per_lang_pair_batch_size):

        if i+per_lang_pair_batch_size>len(lang_pair)+1:
            if len(set(lang_pair[-per_lang_pair_batch_size:])) >1:
                continue

            inputs_dic.append(dict(
                ids=inputs[-per_lang_pair_batch_size:],
                mask=attention_masks[-per_lang_pair_batch_size:],
                label=labels[-per_lang_pair_batch_size:],
                src=lang_dic[lang_pair[i].strip().split(" - ")[0]],
                tgt=lang_dic[lang_pair[i].strip().split(" - ")[1]],
                img_ids=img_file_ids[-per_lang_pair_batch_size:], #add img
                img_feature_path=f'{img_feature_path}/{lang_pair[i].replace(" ","")}/train_boxes36.h5'
                ))

        else:
            if len(set(lang_pair[i:i+per_lang_pair_batch_size])) >1:
                continue
            
            inputs_dic.append(dict(
                ids=inputs[i:i+per_lang_pair_batch_size],
                mask=attention_masks[i:i+per_lang_pair_batch_size],
                label=labels[i:i+per_lang_pair_batch_size],
                src=lang_dic[lang_pair[i].strip().split(" - ")[0]],
                tgt=lang_dic[lang_pair[i].strip().split(" - ")[1]],
                img_ids=img_file_ids[i:i+per_lang_pair_batch_size], #add img
                img_feature_path=f'{img_feature_path}/{lang_pair[i].replace(" ","")}/train_boxes36.h5'
                ))

        


    return dict(
            input_ids=inputs_dic,
            labels=inputs_dic,
            attention_mask=inputs_dic
    )







@dataclass
class DataCollatorForLoraDataset(object):
    """Collate examples for supervised fine-tuning."""


    def __call__(self, instances):

        mask_pad_id = 0

        IGNORE_INDEX = -100
      
        # print(instances)
        if isinstance(instances[0]['input_ids'], dict):
            
            

            input_ids = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(l) for l in instances[0]['input_ids']['ids']], batch_first=True, padding_value=tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(x) for x in instances[0]['input_ids']['label']], batch_first=True, padding_value=IGNORE_INDEX)
            attention_mask = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(l) for l in instances[0]['input_ids']['mask']], batch_first=True, padding_value=mask_pad_id)
           
            src_lang = torch.LongTensor([instances[0]['input_ids']['src']])
            tgt_lang = torch.LongTensor([instances[0]['input_ids']['tgt']])
            
            
            ### add img start
            image_feature = []
            img = np.zeros([len(instances[0]['input_ids']['img_ids']), 108, 2048])
            img_len = []
            n_boxes = 36
            f = instances[0]['input_ids']['img_feature_path']
     
            f = h5py.File(f, 'r')
            
            for k, item in enumerate(instances[0]['input_ids']['img_ids']):
                i = 0
                for j, img_id in enumerate(item):
                    feats = np.zeros(shape=(n_boxes, 2048), dtype=np.float32)
                    if img_id in f.keys():
                        f[f'{img_id}/features'].read_direct(feats) 
                        if i == 0:
                            image_feature = feats
                        else:
                            image_feature = np.concatenate((image_feature, feats), axis=0)
                        i += 1
                if i == 0:
                    image_feature = np.zeros(shape=(1, 2048), dtype=np.float32)
                    img_len.append(image_feature.shape[0])
                else:
                    img_len.append(image_feature.shape[0])
                img[k][:image_feature.shape[0]] = image_feature
            img = img[:,:max(img_len)]
            
            img = torch.tensor(img, dtype=torch.bfloat16)
            ### add img end

   
            return dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,    
                src_lang=src_lang,
                tgt_lang=tgt_lang,   
                image_features=img,   ### add img
                    )

        else:

            input_ids = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(l['input_ids']) for l in instances], batch_first=True, padding_value=tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(l['labels']) for l in instances], batch_first=True, padding_value=IGNORE_INDEX)
            attention_mask = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(l['attention_mask']) for l in instances], batch_first=True, padding_value=mask_pad_id)
            

            return dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,       
                    )






data_files = {"train": data_args.train_dataset}
raw_datasets = load_dataset("text", data_files=data_files)

train_dataset = raw_datasets["train"]

column_names = raw_datasets["train"].column_names
train_dataset = train_dataset.map(Sample_Preprocess_function, batched=True,num_proc=data_args.preprocessing_num_workers,remove_columns=column_names).shuffle()


# Load data collator 
data_collator = DataCollatorForLoraDataset()








# ##### load lora finetune

from peft import get_peft_model, LoraConfig, TaskType

model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16,hyper_config=hyper_args)

if hyper_args.lora_train_hyper:

    save_additional_target = ["hypernetwork.src_lang_emb","hypernetwork.tgt_lang_emb","hypernetwork.layer_emb"]
    target_modules = find_all_linear_modules(model)

    peft_kwargs = {
                "r": lora_args.lora_rank,
                "target_modules": target_modules,
                "lora_alpha": lora_args.lora_alpha,
                "lora_dropout": lora_args.lora_dropout,
                "use_rslora": lora_args.use_rslora,
                "use_dora": lora_args.use_dora,
                "modules_to_save": save_additional_target,
            }

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        **peft_kwargs,
    )


else:
    
    save_additional_target = ["hypernetwork"]
    for name, param in model.named_parameters():
        if is_hypernetwork_module(name):
            save_additional_target.append(name)

    print(save_additional_target)



    target_modules = find_all_linear_modules_wo_hyper(model)
    peft_kwargs = {
                "r": lora_args.lora_rank,
                "target_modules": target_modules,
                "lora_alpha": lora_args.lora_alpha,
                "lora_dropout": lora_args.lora_dropout,
                "use_rslora": lora_args.use_rslora,
                "use_dora": lora_args.use_dora,
                "modules_to_save": save_additional_target,
            }

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        **peft_kwargs,
    )



model = get_peft_model(model, lora_config)


print_trainable_parameters(model)
trainer = Seq2SeqTrainer(
                    model=model,
                    args=training_args,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    callbacks=None,
                    compute_metrics=None,
                    train_dataset=train_dataset,
                  
)





# Training
if training_args.do_train:
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    if trainer.is_world_process_zero() and data_args.plot_loss:
        plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])















