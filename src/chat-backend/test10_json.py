import os
import requests
from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.core.node_parser import JSONNodeParser
import re
from bs4 import BeautifulSoup
load_dotenv()

url = os.getenv('SUCUREMA_CRM')






try:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    print(data["data"])
except requests.exceptions.RequestException as e:
    print(f"Error fetching JSON: {e}")
except ValueError:
    print("Invalid JSON format")


def clean_text(html_text: str):
    if not html_text:
        return ""
    
    if "</li>" in html_text:
        html_text= html_text.replace("</li>", ";")

    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    
    return text


document_objects = [
    Document(
        id_=f'{doc["id"]}',
        text=clean_text(doc["answer"]), 
        metadata={
            "title": doc["question"],
            "updated": doc["updated"],
            "category": doc["category"],
            # "": doc[""],
            # "": doc[""],

        }
    )
    for doc in data["data"]
]

for d in document_objects:
    
    print(d)



# parser = JSONNodeParser()

# nodes = parser.get_nodes_from_documents(data)

# for n in nodes:
#     print(n)