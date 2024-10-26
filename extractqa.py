import json
import os
from pathlib import Path

dataFilePath = "json/qaDataFile.json"


if os.path.exists(dataFilePath):
    with open(dataFilePath, "r") as file:
        dataset = json.load(file)
else:
    dataset = []


def addQAtoDataset(question, answer):
    quesAnsPair = {"instruction": question, "response": answer}
    dataset.append(quesAnsPair)
    with open(dataFilePath, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"\n== Added ==\n{quesAnsPair}")


print("type exit to end")

while True:
    question = input("enter question, or exit to end").strip()
    if question.lower() == "exit":
        break
    answer = input("Enter the answer: ").strip()

    addQAtoDataset(question, answer)

print(f"saved at: {dataFilePath}")
