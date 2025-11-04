import os
import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Document
import time
#from trulens_eval import Tru, TruLlama, Feedback

from sentence_transformers import SentenceTransformer, util
import numpy as np
#from utils_ollama import context_relevance_ollama, answer_relevance_ollama, groundedness_with_ollama
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever

pdf_dir = "./pdfs"
chroma_path = "./chroma_db1"
#collection_name = "faqs_new"

collection_name_window = "faqs_window"
collection_name_split = "faqs_split"


embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


def chroma_db_exists(path):
    return os.path.exists(path) and len(os.listdir(path)) > 0

if not chroma_db_exists(chroma_path):
    print("No ChromaDB found — running document pipeline...")

    documents = SimpleDirectoryReader(pdf_dir).load_data()
    #raw_text = "\n\n".join([doc.text for doc in documents])
    #cleaned_text = raw_text.replace("\n", " ").replace("  ", " ")
    #document = Document(text=cleaned_text)
    document = Document(text="\n\n".join([doc.text for doc in documents]))



    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )


    #splitter = SentenceSplitter(separator=" ", chunk_size=512, chunk_overlap=50)
    #pipeline = IngestionPipeline(transformations=[splitter])

    text_splitter = SentenceSplitter(separator=" ", chunk_size=512, chunk_overlap=50)

   # Settings.chunk_size = 520
    #Settings.chunk_overlap = 100
    Settings.text_splitter = text_splitter

    Settings.llm = Ollama(model="llama3.2:latest", request_timeout=60.0)
    #Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
   # embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    pipeline = IngestionPipeline(transformations=[text_splitter])
    base_nodes = pipeline.run(documents=[document], in_place=True, show_progress=True)

    #nodes = pipeline.run(documents=[document], in_place=True, show_progress=True)
    nodes = node_parser.get_nodes_from_documents(documents)
    #base_nodes = text_splitter.get_nodes_from_documents(documents)


    db = chromadb.PersistentClient(path=chroma_path)
    #collection = db.get_or_create_collection(collection_name)
    collection_window = db.get_or_create_collection(collection_name_window)
    
    vector_store_window = ChromaVectorStore(chroma_collection=collection_window)
    storage_context = StorageContext.from_defaults(vector_store=vector_store_window)

    index = VectorStoreIndex.from_documents(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    

    # Collection für SentenceSplitter
    collection_split = db.get_or_create_collection(collection_name_split)
    vector_store_split = ChromaVectorStore(chroma_collection=collection_split)
    storage_context_split = StorageContext.from_defaults(vector_store=vector_store_split)

    base_index = VectorStoreIndex.from_documents(
        base_nodes,
        storage_context=storage_context_split,
        embed_model=embed_model,
    )

    bm25_retriever = BM25Retriever.from_defaults(nodes=base_nodes, similarity_top_k=2)

    #sentence_index = VectorStoreIndex(nodes)

    #base_index = VectorStoreIndex(base_nodes)


    print("✅ Index built and stored.")
else:
    print("✅ ChromaDB found — loading index...")

    db = chromadb.PersistentClient(path=chroma_path)
    # collection = db.get_or_create_collection(collection_name)
    # vector_store = ChromaVectorStore(chroma_collection=collection)
    # storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # index = VectorStoreIndex.from_vector_store(
    #     vector_store=vector_store,
    #     storage_context=storage_context,
    #     #embed_model=embed_model,
    # )

#query_engine = index.as_query_engine(similarity_top_k=3)

retriever = index.as_retriever()

vector_retriever = VectorIndexRetriever(index)


base_retriever = base_index.as_retriever()
base_query_engine = base_index.as_query_engine(similarity_top_k=3)
while True:
    query = input("Ask a question about the PDFs (or type 'exit'): ")
    if query.strip().lower() == "exit":
        break

    
    #response = query_engine.query(query)
    # print("\nTop Matching Chunks:")
    # results = retriever.retrieve(query)
    # for i, node in enumerate(results):
    #     print(f"\n--- Chunk {i+1} ---\n{node.get_content()}")

    results = vector_retriever.retrieve(query)
    print("\nTop Matching Chunks (Vector Retriever):")
    for i, node_with_score in enumerate(results):
        print(f"\n--- Chunk {i+1} ---")
        print(node_with_score.node.get_content())
    print("##################")

    results2 = bm25_retriever.retrieve(query)
    print("\nTop Matching Chunks (BM25):")
    for i, node_with_score in enumerate(results2):
        print(f"\n--- Chunk {i+1} ---")
        print(node_with_score.node.get_content())
    print("##################")


    results_2 = base_retriever.retrieve(query)
    for i, node in enumerate(results_2):
        print(f"\n--- Chunk {i+1} ---\n{node.get_content()}")

    response3 = base_query_engine.query(query)
    print("\nAntwort 2:\n", response3)
#Wie viele Mitarbieter arbeiten bei ilume?

# questions = [
#     "Wie viele Abteilungen hat ilume?",
#     "Wie viele Mitarbieter arbeiten bei ilume?",
#     "Wo wurde das Pflichtpraktikum gemacht?",
#     "Wo befindet sich der Hauptsitz von ilume?",
#     "Wann wurde das Pflichtpraktikum gemacht?",
#      "Wann wurde ilume gegründet?"
# ]