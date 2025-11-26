import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from extras.logging import get_logger


IGNORE_INDEX = -100


logger = get_logger(__name__)




def find_all_linear_modules(model: "PreTrainedModel") -> List[str]:
    r"""
    Finds all available modules to apply LoRA, GaLore or APOLLO.
    """
    model_type = getattr(model.config, "model_type", None)
    forbidden_modules = {"lm_head"}


    module_names = set()
    for name, module in model.named_modules():
        if any(forbidden_module in name for forbidden_module in forbidden_modules):
            continue

        if "Linear" in module.__class__.__name__ and "Embedding" not in module.__class__.__name__:
            module_names.add(name.split(".")[-1])

    logger.info_rank0("Found linear modules: {}".format(",".join(module_names)))
    return list(module_names)



def find_all_linear_modules_wo_hyper(model: "PreTrainedModel") -> List[str]:
    r"""
    Finds all available modules to apply LoRA, GaLore or APOLLO.
    """
    model_type = getattr(model.config, "model_type", None)
    forbidden_modules = {"lm_head"}


    module_names = set()
    for name, module in model.named_modules():
        if any(forbidden_module in name for forbidden_module in forbidden_modules):
            continue

        if "hypernetwork" in name.lower():
            continue

        if "Linear" in module.__class__.__name__ and "Embedding" not in module.__class__.__name__:
            module_names.add(name.split(".")[-1])

    logger.info_rank0("Found linear modules: {}".format(",".join(module_names)))
    return list(module_names)

    

def single_inference(input_text,model,tokenizer,max_token):
    messages = [{"role": "user", "content": input_text},]
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    input_ids = tokenizer.encode(
                        input_text,
                        truncation=True,
                        max_length=30,
        
                        )

    generate_kwargs = dict(
            input_ids=input_ids,
            do_sample=True,
            temperature=0.6, 
            top_p=0.9, 
            pad_token_id = tokenizer.eos_token_id,
            max_new_tokens=max_token,
            )
    outputs = model.generate(**generate_kwargs)
    
    response = outputs[0][input_ids.shape[-1]:]
    
    return tokenizer.decode(response)






def find_different_lang_id(lis):
        previous_langpair=""
        for index in range(0,len(lis)):
            if index == 0:
                previous_langpair=lis[index]
            else:
                if previous_langpair != lis[index]:
                    return index
                

def Sample_Preprocess_function(examples):

    lang_list="english,russian,indonesian,urdu"
 

    inputs = []
    labels = []
    attention_masks = []

    lang_pair = [sample.strip().split(" : ")[0]  for sample in inputs]

    for sample in examples["text"]:
        
        source_text,target_text,_ = sample.strip().split("\t\t")


        src_lang,tgt_lang = source_text.strip().split(" : ")[0].split(" - ")

        src_text = source_text.strip().split(" : ")[1]

        instruct = f"Summarize the following {src_lang} text into a {tgt_lang} abstract: "

        src_text = instruct + src_text +"\nAnswer: "

        src_wtih_tgt = src_text + target_text

        src_text_ids = tokenizer.encode(
                        src_text,
                        truncation=True,
                        max_length=data_args.cutoff_len,
                        )

        source_len = len(src_text_ids)



        src_wtih_tgt_ids = tokenizer.encode(
                            src_wtih_tgt,
                            truncation=True,
                            max_length=data_args.cutoff_len,
                         )

        src_wtih_tgt_length = len(src_wtih_tgt_ids)


        assert src_wtih_tgt_length > source_len

        label_ids = [IGNORE_INDEX]*source_len +src_wtih_tgt_ids[source_len:]
        
        src_wtih_tgt_ids = src_wtih_tgt_ids + [tokenizer.eos_token_id]
        label_ids = label_ids + [tokenizer.eos_token_id]

        attention_mask = [1]*(src_wtih_tgt_length+1)

        inputs.append(src_wtih_tgt_ids)
        labels.append(label_ids)
        attention_masks.append(attention_mask)



    return dict(
            input_ids=inputs,
            labels=labels,
            attention_mask=attention_masks
    )


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter {name} is trainable (requires_grad=True)")
        else:
            print(f"Parameter {name} is frozen (requires_grad=False)")

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")



def is_hypernetwork_module(name):
    return "hypernetwork" in name.lower()





