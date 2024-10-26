from pathlib import Path
import re
import time
from groq import Groq
import os
from dotenv import load_dotenv
import pandas as pd
import json
import shutil

load_dotenv()
groqAPI = os.getenv("GROK_API_KEY")

client = Groq(
    api_key=groqAPI,
)

inputDir = os.path.join(os.getcwd(), "src/data/").replace("\\", "/")
inputDirDone = os.path.join(os.getcwd(), "src/data/done/").replace("\\", "/")
outputFile = os.path.join(os.getcwd(), "src/json/liveDataset.csv").replace("\\", "/")
logFilePath = os.path.join(os.getcwd(), "src/logs/run.log").replace("\\", "/")


prefixPrompt = """Extract question-answer pairs from a college website's markdown document.
Steps: 1. Formulate questions covering major sections, sub-sections, instructions, processes, and key terms.
2. Answer questions like a helpdesk assistant: clear, accurate, and step-by-step guidance when needed.
3. Retain relevant links in answers for additional resources.
4. If additional resources are available, provide a brief summary and include the link for further reading.
Output format: Return only the list of Python objects, starting with '[' and ending with ']'. Omit any beginning or ending text.
Format:
[
{
"instruction": (question),
"response": (answer)
},
...
]
"""

pattern = r"\[.*?\]$"
dataset = pd.DataFrame(columns=["instruction", "response"])


def getFromGroq(document: str) -> str:
    input = f"{prefixPrompt}\n\nDocument:\n ```{document}```".split(" ")
    if len(input) >= 6000:
        input = " ".join(input[:5600])
    else:
        input = input = " ".join(input)

    chatCompletion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": input,
            }
        ],
        model="llama3-70b-8192",
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
        output = ""
        try:
            output = getFromGroq(doc)
        except Exception as e:
            print(e)
            if e.response.status_code == 429:
                retryafter = e.response.headers["retry-after"]
                print(f"waiting for {int(retryafter) + 1} seconds")
                time.sleep(int(retryafter) + 1)

        matches = re.search(pattern, output, re.DOTALL)
        if matches:
            try:
                objectList = json.loads(matches.group(0))

                newDF = pd.DataFrame(objectList)
                dataset = pd.concat([dataset, newDF], ignore_index=True)
                dataset.to_csv(outputFile, index=False)
                shutil.move(inputFilePath, inputDirDone)
                print("done and shifted")
            except Exception as e:
                print("Error parsing JSON:", e)
    time.sleep(1)
