from transformers import AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainer, TrainingArguments,HfArgumentParser
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from datasets import load_from_disk, Dataset
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from hparams import ModelArguments,TrainingArguments,DataArguments,LoraArguments,HyperAdapterArguments
from datasets import load_dataset
import torch
import re
import os
from utils import find_all_linear_modules,print_trainable_parameters
from extras.logging import get_logger
from rouge_score import rouge_scorer,scoring
import nltk
from tqdm import tqdm
import h5py
import numpy as np






DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

logger = get_logger(__name__)

parser = HfArgumentParser( (ModelArguments,DataArguments,HyperAdapterArguments))

model_args,data_args,hyper_args = parser.parse_args_into_dataclasses()



tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
IGNORE_INDEX = -100
img_feature_path = hyper_args.img_path



def add_newline_to_end_of_each_sentence(x: str) -> str:
    """This was added to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
    re.sub("<n>", "", x)  # remove pegasus newline char
    return "\n".join(nltk.sent_tokenize(x))



def list_files(directory):
    file_list = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
            # print(os.path.join(root, file))

    return file_list



def Sample_Preprocess_function(examples):

    inputs = []
    labels = []
    attention_masks = []

    img_ids_list = []
    img_feature_path_list = []

    lang_pair = [sample.strip().split(" : ")[0]  for sample in examples["text"]]
    img_file_ids =  [sample.strip().split("\t\t")[2].split("\t")[:3] for sample in examples["text"]]

    for sample in examples["text"]:
        
        source_text,target_text,_ = sample.strip().split("\t\t")


        src_lang,tgt_lang = source_text.strip().split(" : ")[0].split(" - ")

        src_text = source_text.strip().split(" : ")[1]

        instruct = f"Summarize the following {src_lang} text into a {tgt_lang} abstract: "

        # src_text = instruct + src_text +"\nAnswer:"
        src_text = instruct + src_text 

        inputs.append(src_text)
        labels.append(target_text)


        lang_pair = sample.strip().split(" : ")[0] 

        img_ids = sample.strip().split("\t\t")[2].split("\t")[:3]
        img_feature_path=f'{img_feature_path}/{lang_pair.replace(" ","")}/train_boxes36.h5'

        img_ids_list.append(img_ids)
        img_feature_path_list.append(img_feature_path)


    return dict(
            input_ids=inputs,
            labels=labels,
            img_ids_list=img_ids_list,
            img_feature_path_list=img_feature_path_list
    )









 

def cal_score(test_dataset_path,checkpoint_name,output_prediction_path,predictions,labels):

    language = os.path.basename(test_dataset_path).split(".")[0].split("-")[1]
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2','rougeL'], use_stemmer=True, lang=language)
    print(language)

    aggregator = scoring.BootstrapAggregator()

    rouge1_score = 0
    rouge2_score = 0
    rougeL_score = 0
    sum_num = 0


    if not os.path.exists(os.path.join(output_prediction_path, os.path.basename(checkpoint_name.strip("/")))):
        # 如果不存在，则创建文件夹
        os.makedirs(os.path.join(output_prediction_path, os.path.basename(checkpoint_name.strip("/"))))


    output_prediction_file = os.path.join(output_prediction_path, os.path.basename(checkpoint_name.strip("/")),os.path.basename(test_dataset_path))
    print(model_args.model_name_or_path)
    print(os.path.basename(model_args.model_name_or_path))
    print(output_prediction_file)         
    aggregator = scoring.BootstrapAggregator()

    print(len(predictions))
    with open(output_prediction_file, "w", encoding="utf-8") as writer:
        for p,l in zip(predictions,labels):
            writer.write(l+"\t"+p+"\n")
            
            pred = add_newline_to_end_of_each_sentence(p)
            tgt = add_newline_to_end_of_each_sentence(l)
            scores = scorer.score(pred,tgt)
            aggregator.add_scores(scores)

            #scores = scorer.score(p,l)
            
        direction=os.path.basename(test_dataset_path).strip(".txt")
        print(direction)
        result = aggregator.aggregate()
            
        print({k: float(round(v.mid.fmeasure * 100, 4)) for k, v in result.items()})
        result_file = os.path.join(output_prediction_path, os.path.basename(checkpoint_name.strip("/")),"all-result-score.txt")
                
        with open(result_file,"a",encoding="utf-8") as f:
            f.write(str(direction)+"\t\t"+str({k: float(round(v.mid.fmeasure * 100, 4)) for k, v in result.items()})+"\n")





def batch_prediction(model,tokenizer,test_dataset,batch_size):
    label = []
    preds = []
    max_token=128

    for index in tqdm(range(0, len(test_dataset), batch_size)):
        # 如果 i+4 大于列表长度，则将批次设置为从 i 到列表结束
        if index + 4 > len(test_dataset):
            batch = test_dataset[index:]  # 获取从 i 到列表末尾的部分
        else:
            batch = test_dataset[index:index + batch_size]  # 获取 4 个元素

      

        for i in batch["labels"]:
            label.append(i)

        input_text = batch["input_ids"]

        input_ids = tokenizer(input_text,return_tensors="pt",padding=True,truncation=True,max_length=2048).to(model.device)



        Answer_token = "\nAnswer:"
        special_token = tokenizer(Answer_token,return_tensors="pt",add_special_tokens=False).to(model.device)

        repeat_size = input_ids["input_ids"].shape[0]

        input_ids["input_ids"] = torch.cat((input_ids["input_ids"],special_token["input_ids"].repeat(repeat_size,1)),dim=-1)
        input_ids["attention_mask"] = torch.cat((input_ids["attention_mask"],special_token["attention_mask"].repeat(repeat_size,1)),dim=-1)



        


        ### add img start
        image_feature = []
        img = np.zeros([len(batch["img_ids_list"]), 108, 2048])
        img_len = []
        n_boxes = 36
        f = batch["img_feature_path_list"][0]
    
        f = h5py.File(f, 'r')
        
        for k, item in enumerate(batch["img_ids_list"]):
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
        
        img = torch.tensor(img, dtype=torch.bfloat16).to(model.device)
            ### add img end





        generate_kwargs = dict(
                input_ids=input_ids["input_ids"],
                attention_mask=input_ids["attention_mask"],
                image_features=img,
                do_sample=True,
                temperature=0.6, 
                top_p=0.9, 
                pad_token_id = tokenizer.eos_token_id,
                max_new_tokens=max_token,
                )
        
        outputs = model.generate(**generate_kwargs)
        model_pred = tokenizer.batch_decode(outputs[:,input_ids["input_ids"].shape[-1]:],skip_special_tokens = True)
        for i in model_pred:
            preds.append(i)

    return preds,label




def get_adapter_state_dict(model, adapter_name):
    adapter_state_dict = model.state_dict()
    
    adapter_weights = {key: value for key, value in adapter_state_dict.items() if adapter_name in key}

    for key, value in adapter_state_dict.items():
        if adapter_name in key:
            print(key)

    # return adapter_weights




import time
import peft


start_time = time.time()

test_file_list = list_files("/mnt/lvbo/hamms-hyper-na/data/test_data")

for test_dataset_file in test_file_list:


    lora_path = data_args.test_checkpoint
    print(lora_path)

    output_prediction_path = data_args.output_prediction_path

    data_files = {"test": test_dataset_file}
    raw_datasets = load_dataset("text", data_files=data_files)

    test_dataset = raw_datasets["test"]

    column_names = raw_datasets["test"].column_names
    test_dataset = test_dataset.map(Sample_Preprocess_function, batched=True,num_proc=data_args.preprocessing_num_workers,remove_columns=column_names)
    



    num = 0
    batch_size = 12


    if hyper_args.hyper_predict:
        hyper_lang_list=hyper_args.language_list
        hyperlang_dic={key: index for index, key in enumerate(hyper_lang_list.split(","))}
        src_lang,tgt_lang = os.path.basename(test_dataset_file).strip(".txt").split("-") 
        hyper_args.hyper_src_lang = hyperlang_dic[src_lang]
        hyper_args.hyper_tgt_lang = hyperlang_dic[tgt_lang]

    print("lang pair")
    print(hyper_args.hyper_src_lang,hyper_args.hyper_tgt_lang)


    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16,hyper_config=hyper_args)

    model = PeftModel.from_pretrained(
                model, lora_path, is_trainable=False
            ).to(DEVICE)
    
 

    preds,labels = batch_prediction(model,tokenizer,test_dataset,batch_size)


    cal_score(test_dataset_file,lora_path,output_prediction_path,preds,labels)

    

# 记录结束时间
end_time = time.time()

# 计算运行时间（秒）
elapsed_time = end_time - start_time

# 转换为分钟
elapsed_minutes = elapsed_time / 60

print(f"程序运行了 {elapsed_minutes:.2f} 分钟")

