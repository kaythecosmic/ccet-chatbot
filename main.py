import json
import re
from pprint import pprint
import torch
import numpy as np
import pandas as pd

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from datasets import Dataset, load_dataset, DatasetDict
from pathlib import Path

from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model, TaskType

MODEL_ID = "meta-llama/Llama-3.1-8B"
# MODEL_ID = "meta-llama/Meta-Llama-3-8B"            # Memory full error on 10 epochs
DATASET_PATH = "/kaggle/input/ccet-qa/dataset.csv"

# !pip install -U "huggingface_hub[cli]"
# !pip install peft

# !pip3 install pipreqsnb


import os
for dirname, _, filenames in os.walk('/kaggle/'):  
    for filename in filenames:  
        print(os.path.join(dirname, filename))


# !pipreqsnb --savepath ./requirements.txt path_to_ipynb_file_from_step_2.ipynb

from IPython.display import FileLink 
FileLink(r'/kaggle/working/requirements.txt')



dataDF = pd.read_csv(DATASET_PATH)
trainDataDF, tempDataDF = train_test_split(dataDF, test_size=0.3, random_state=42)
valDataDF, testDataDF = train_test_split(tempDataDF, test_size=0.3333, random_state=42)
trainDataDF = trainDataDF.reset_index(drop=True)
valDataDF = valDataDF.reset_index(drop=True)
testDataDF = testDataDF.reset_index(drop=True)


trainDataset = Dataset.from_pandas(trainDataDF)
valDataset = Dataset.from_pandas(valDataDF)
testDataset = Dataset.from_pandas(testDataDF)

dataset = DatasetDict({
    "train": trainDataset,
    "val": valDataset,
    "test": testDataset
})

print(dataset)



# !huggingface-cli login --token "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" --add-to-git-credential

def initModelAndTokenizer(modelID: str):

    model = AutoModelForCausalLM.from_pretrained(
        modelID,
        use_safetensors=True,
        trust_remote_code=True,
        device_map = "auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(modelID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

# get model and tokenizer
model, tokenizer = initModelAndTokenizer(MODEL_ID)


def tokenizeInputText(sample):
    inputs = ['### Question:\n' + instruction + '\n\n### Answer:\n' for instruction in sample["instruction"]]
    tokenizedSample = tokenizer(inputs, padding="max_length", truncation=True, return_tensors="pt", max_length = 256)
    sample["input_ids"] = tokenizedSample["input_ids"]
    sample["attention_mask"] = tokenizedSample["attention_mask"]
    labels = tokenizer(sample["response"], padding="max_length", truncation=True, return_tensors="pt", max_length=256)["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    sample["labels"] = labels
    return sample


tokenizedDataset = dataset.map(tokenizeInputText, batched=True)
tokenizedDataset = tokenizedDataset.remove_columns(['instruction', 'response'])
tokenizedDataset
# del dataset



# Training Section

# from pathlib import Path
# OUTPUT_DIR = "/kaggle/working/llama31-8B-train-1"
# Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# trainArguments = TrainingArguments(
#     output_dir = PEFT_OUTPUT_DIR,
#     save_strategy='epoch',
#     num_train_epochs=1,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     gradient_accumulation_steps=1,
#     learning_rate=5e-5,
#     seed=42,
#     fp16=True, 
# )

# trainer = Trainer(
#     model=model,
#     args=trainArguments,
#     train_dataset=tokenizedDataset['train'],
#     eval_dataset=tokenizedDataset['val'],
#     tokenizer=tokenizer,
# )

PEFT_OUTPUT_DIR = "/kaggle/working/llama31-8B-train-1-peft"
Path(PEFT_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


import gc
gc.collect()
torch.cuda.empty_cache()
loraConfig = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "mlp_proj"],
    lora_dropout=0.04,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
peftModel = get_peft_model(model, loraConfig)
model.gradient_checkpointing_enable()
peftTrainingArgs = TrainingArguments(
    output_dir=PEFT_OUTPUT_DIR,
    save_strategy='epoch',
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    seed=42,
    fp16=True,
    optim="adamw_bnb_8bit"
)
peftTrainer = Trainer(
    model=peft_model,
    args=peftTrainingArgs,
    train_dataset=tokenized_datasets["train"],
)


peftTrainer.train()


model.eval() 
testIndex = 10
inputIds = torch.stack([torch.tensor(ids) for ids in tokenizedDataset["test"]["input_ids"][testIndex:testIndex+1]]).to(device)
attentionMask = torch.stack([torch.tensor(mask) for mask in tokenizedDataset["test"]["attention_mask"][testIndex:testIndex+1]]).to(device)

outputs = model.generate(input_ids=inputIds, max_new_tokens=50, temperature=0.6, attention_mask=attentionMask, pad_token_id=tokenizer.eos_token_id)

textedOutput = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(textedOutput)