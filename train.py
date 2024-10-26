from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import numpy as np
import pandas as pd
import torch

DATASET_PATH = Path("./json/archive/dataset.csv")

# loading the dataset
dataset = pd.read_csv(DATASET_PATH)
print(dataset["instruction"].nunique())
print(dataset["response"].nunique())

# # large language model
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
