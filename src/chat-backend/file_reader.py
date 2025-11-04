from langgraph.graph import StateGraph
# from langchain_chroma import Chroma
import pprint
import chromadb
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import MetadataMode
import os


# embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
# llm = Ollama(
#     model="llama3.2:1b",
#     model="llama3.2:latest",
#     request_timeout=120.0 
# )


# pdf_dir = "./pdfs"

# dir_list = os.listdir(pdf_dir)
# print(dir_list)

# for doc in dir_list:
#     file_path = os.path.join(pdf_dir, doc)
#     with open(file_path, "rb") as file:  # Use "rb" for PDFs
#         print(file.read())

from PyPDF2 import PdfReader
import os

pdf_dir = "./pdfs"
dir_list = os.listdir(pdf_dir)

for doc in dir_list:
    file_path = os.path.join(pdf_dir, doc)
    reader = PdfReader(file_path)
    print(f"\n--- {doc} ---")

    print(len(reader.pages))

    #document = ""

    docs = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            #reader = PdfReader(file_path)

            # full_text = ""
            # for page in reader.pages:
            #     text = page.extract_text()
            #     if text:
            #         full_text += text + "\n"

            docs = SimpleDirectoryReader(pdf_dir).load_data()

            # Create one Document per PDF
            #doc = Document(text=full_text, metadata={"filename": filename})
            #docs.append(doc)
        
    
            print(len(docs))
            print(docs[0].metadata)
            print("---------")
            print(docs[1].metadata)

# docs = SimpleDirectoryReader(pdf_dir).load_data()

# for doc in docs:
#     print(doc.get_content())
#     print(len(doc.get_content()))
#     print("---------")

# text_splitter = SentenceSplitter(separator=" ", chunk_size=512, chunk_overlap=50)