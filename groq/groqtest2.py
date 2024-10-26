import re
import os
import time
import json
import shutil
import pickle
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from transformers import AutoTokenizer

from huggingface_hub import login

login(token="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")

load_dotenv()
groqAPI = os.getenv("GROK_API_KEY")

client = Groq(
    api_key=groqAPI,
)

inputDir = os.path.join(os.getcwd(), "src/data/").replace("\\", "/")
inputDirDone = os.path.join(os.getcwd(), "src/data/done/").replace("\\", "/")
outputFile = os.path.join(os.getcwd(), "src/json/liveDataset.csv").replace("\\", "/")
logFilePath = os.path.join(os.getcwd(), "src/logs/run.log").replace("\\", "/")
linksPKLFIle = os.path.join(os.getcwd(), "src/json/pickles/links.pkl").replace(
    "\\", "/"
)

LINKS = dict()
with open(linksPKLFIle, "rb") as linksFile:
    LINKS = pickle.load(linksFile)

# prefixPrompt = """Extract question-answer pairs from a college website's markdown document.
# Steps: 1. Formulate questions covering major sections, sub-sections, instructions, processes, and key terms.
# 2. Answer questions like a helpdesk assistant: clear, accurate, and step-by-step guidance when needed.
# Output format: Return only the list of Python objects, starting with '[' and ending with ']'. Omit any beginning or ending text.
# Format:
# [
# {
# "instruction": (question),
# "response": (answer)
# },
# ...
# ]
# """

prefixPrompt = """Extract question-answer pairs from a college website's markdown document.
1. Formulate questions that cover important sections, sub-sections, instructions, processes, and key terms in the document.
2. Provide clear, thorough, and accurate responses, as if answering as a knowledgeable helpdesk assistant.

Return only a Python list of dictionaries with the following structure.

[
{
"instruction": "The formulated question here",
"response": "The detailed answer here"
}
]
Ensure that the output starts with '[' and ends with ']' without any additional text.
"""


pattern = r"\[.*?\](?!.*\])"
dataset = pd.DataFrame(columns=["instruction", "response"])

modelName = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(modelName)


def clipToTokenLimit(prefixPrompt, document, maxTokens=18000):
    prefixTokens = tokenizer.encode(prefixPrompt)
    documentTokens = tokenizer.encode(document)
    remainingTokens = maxTokens - len(prefixTokens)
    if len(documentTokens) > remainingTokens:
        print(len(documentTokens))
        documentTokens = documentTokens[:remainingTokens]
    clippedDocument = tokenizer.decode(documentTokens)
    finalInput = f"{prefixPrompt}\n\nDocument:\n ```{clippedDocument}```"
    return finalInput


def getFromGroq(document: str) -> str:
    inputPrompt = clipToTokenLimit(prefixPrompt, document)
    chatCompletion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": inputPrompt,
            }
        ],
        # model="llama3-70b-8192",
        model="llama-3.1-70b-versatile",
        # model="llama3-8b-8192",
    )
    return chatCompletion.choices[0].message.content


inputFilesList = os.listdir(inputDir)
totalFiles = len(inputFilesList)

for idx, fileName in enumerate(inputFilesList):
    print(f"progress: {idx+1}/{totalFiles}\t\tfile:{fileName}")
    inputFilePath = f"{inputDir}/{fileName}"

    with open(inputFilePath, mode="r", encoding="utf-8") as inputFile:
        doc = inputFile.read()

    if doc is not None:
        try:
            output = getFromGroq(doc)
            # print(output)
        except Exception as e:
            if e.response.status_code == 429:
                retryafter = e.response.headers["retry-after"]
                print(f"waiting for {int(retryafter) + 1} seconds")
                time.sleep(int(retryafter) + 1)
        matches = re.search(pattern, output, re.DOTALL)
        if matches:
            try:
                with open(logFilePath, mode="a", encoding="utf-8") as logsFile:
                    logsFile.write(f"progress: {idx+1}/{totalFiles}\n\n")
                objectList = json.loads(matches.group(0))
                newDF = pd.DataFrame(objectList, columns=["instruction", "response"])
                dataset = pd.concat([dataset, newDF], ignore_index=True)
                dataset.to_csv(outputFile, index=False)
                shutil.move(inputFilePath, inputDirDone)
            except Exception as e:
                with open(logFilePath, mode="a", encoding="utf-8") as logsFile:
                    logsFile.write(f"{e}\t\tin file: {fileName}\n\n")
                print("Error parsing JSON:", e)
    time.sleep(1)
