
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import marko
import marko.patterns
import time

OUTPUT_DIR = Path("./data")
JINA_PREFIX = "https://r.jina.ai/"
LINKS = dict(domain=set(), external=set())


def getRequest(link: str) -> str:
    response = requests.get(link)
    return response.text


def getWithJina(link: str) -> str:
    appendedLink = JINA_PREFIX + link
    return getRequest(appendedLink)


def getSoup(link: str):
    response = getRequest(link)
    soup = BeautifulSoup(response, "html.parser")
    return soup


# pageName = "Contact"
# response = getWithJina("https://www.ccet.ac.in/degreeCourse.php")
# with open(OUTPUT_DIR / f"{pageName}.md", mode="w") as file:
#     file.write(response)
# print(len(response))

from marko.inline import Link


def getAllPageLinks(response) -> set:
    markoparsed = marko.parse(response)
    domainLinks = set()
    externalLinks = set()

    def walkElementsOfMD(element):
        if isinstance(element, Link):
            if "ccet.ac.in" in element.dest:
                domainLinks.add(element.dest)
            else:
                externalLinks.add(element.dest)
        for child in getattr(element, "children", []):
            walkElementsOfMD(child)

    walkElementsOfMD(markoparsed)
    return domainLinks, externalLinks


home = getWithJina("https://www.ccet.ac.in")
domain, external = getAllPageLinks(home)

LINKS["domain"].update(domain)
LINKS["external"].update(external)

QUEUE = set(domain)
pageCounter = 1
mediaExtensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".pdf",
    ".mp4",
    ".mp3",
    ".avi",
    ".mov",
    ".wav",
)

while len(QUEUE) != 0:
    currLink = QUEUE.pop()
    if currLink.lower().endswith(mediaExtensions):
        print(f"Skipping media link: {currLink}")
        continue
    currResponse = getWithJina(currLink)
    with open(OUTPUT_DIR / f"{pageCounter}.md", mode="w", encoding="utf-8") as file:
        file.write(currResponse)
    domain, external = getAllPageLinks(currResponse)
    onlyNewLinks = domain.difference(LINKS["domain"])
    LINKS["domain"].update(onlyNewLinks)
    LINKS["external"].update(external)
    QUEUE.update(onlyNewLinks)
    print(f"Progress: {pageCounter}/{len(LINKS['domain'])}\t Queue: {len(QUEUE)}")
    pageCounter += 1
    time.sleep(1.5)