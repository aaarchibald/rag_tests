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

from trulens_eval import Tru, TruLlama, Feedback, feedback
from trulens.dashboard import run_dashboard
from sentence_transformers import SentenceTransformer, util
import numpy as np

# === Config Paths ===
pdf_dir = "./pdfs"
chroma_path = "./chroma_db1"
collection_name = "faqs"

# === Load Local Models ===
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = Ollama(model="llama3.2:latest", request_timeout=120.0)

# === Initialize Local Embedding Model for Feedback ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Feedback Function (semantic similarity between query and response) ===
def semantic_similarity(query, response):
    query_emb = embedder.encode(query, convert_to_tensor=True)
    resp_emb = embedder.encode(str(response), convert_to_tensor=True)
    return util.pytorch_cos_sim(query_emb, resp_emb).item()

#feedback = Feedback(semantic_similarity, name="Semantic Similarity").on_input_output()

# === Ensure ChromaDB Exists or Build Index ===
def chroma_db_exists(path):
    return os.path.exists(path) and len(os.listdir(path)) > 0

if not chroma_db_exists(chroma_path):
    print("📦 No ChromaDB found — running document pipeline...")

    documents = SimpleDirectoryReader(pdf_dir).load_data()
    raw_text = "\n\n".join([doc.text for doc in documents])
    cleaned_text = raw_text.replace("\n", " ").replace("  ", " ")
    document = Document(text=cleaned_text)

    splitter = SentenceSplitter(separator=" ", chunk_size=512, chunk_overlap=50)
    pipeline = IngestionPipeline(transformations=[splitter])
    nodes = pipeline.run(documents=[document], in_place=True, show_progress=True)

    db = chromadb.PersistentClient(path=chroma_path)
    collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    print("✅ Index built and stored.")
else:
    print("✅ ChromaDB found — loading index...")

    db = chromadb.PersistentClient(path=chroma_path)
    collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

# === Query Engine ===
query_engine = index.as_query_engine(llm=llm)
retriever = index.as_retriever(similarity_top_k=3)

# === Wrap with TruLlama ===
tru = Tru()


tru_query_engine = TruLlama(
    query_engine,
    #app=query_engine,
    #app_id="ollama-pdf-app",
    #feedbacks=[feedback]
)

hugs = feedback.Huggingface()
feedback.
#openai = #
# Define a language match feedback function using HuggingFace.
f_lang_match = Feedback(hugs.language_match).on_input_output()
# By default this will check language match on the main app input and main app
# output.

# Question/answer relevance between overall question and answer.
f_qa_relevance = Feedback(openai.relevance).on_input_output()

# Question/statement relevance between question and each context chunk.
f_qs_relevance = Feedback(openai.qs_relevance).on_input().on(
    TruLlama.select_source_nodes().node.text
).aggregate(np.min)

# === Interactive Loop ===
while True:
    query = input("Ask a question about the PDFs (or type 'exit'): ")
    if query.strip().lower() == "exit":
        break

    # Automatically records query, response, feedback
    # with tru_query_engine as recording:
    #     response = tru_query_engine.query(query)
    with tru_query_engine as recording:
        response = query_engine.query(query)
        response2 = tru_query_engine(query)
    print("\nTop Matching Chunks:")
    results = retriever.retrieve(query)
    for i, node in enumerate(results):
        print(f"\n--- Chunk {i+1} ---\n{node.get_content()}")

    print("\nAntwort:\n", response)
    print("\nAntwort 2 ....:\n", response2)
