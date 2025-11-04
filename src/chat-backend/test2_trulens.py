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
from trulens.core.feedback import Feedback
from trulens.apps.llamaindex import TruLlama
from trulens.providers.openai import OpenAI
from trulens.providers.litellm import LiteLLM
from trulens.dashboard import run_dashboard
from sentence_transformers import SentenceTransformer, util
import numpy as np
from trulens.core import TruSession
#from utils_ollama import context_relevance_ollama, answer_relevance_ollama, groundedness_with_ollama
from llama_index.core import Settings

# === Config Paths ===
pdf_dir = "./pdfs"
chroma_path = "./chroma_db1"
collection_name = "faqs"

session = TruSession()
#session.reset_database()


Settings.chunk_size = 128
Settings.chunk_overlap = 16

Settings.llm = Ollama(model="llama3.2:latest", request_timeout=60.0)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


# documents = SimpleDirectoryReader("data").load_data()
# index = VectorStoreIndex.from_documents(documents)

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

query_engine = index.as_query_engine(similarity_top_k=3)




#response = query_engine.query("Wie viele Mitarbeiter hat ilume?")
#print(response)


############## Initialize Feedback Function(s)
#provider = Ollama(model="llama3.2:latest")
#provider = OpenAI(model_engine="gpt-4.1-mini")
provider = LiteLLM(
    model_engine="ollama/llama3.2:latest",  
    completion_kwargs={
        "api_base": "http://localhost:4000",  # Gateway address
        "api_key": "sk-h2k4hd847hi3",          # Optional if key required
    }
)


# Define a groundedness feedback function
f_groundedness = (
    Feedback(
        provider.groundedness_measure_with_cot_reasons, 
        name="Groundedness"
    )
    #Feedback(groundedness_with_ollama, name="Groundedness")
    .on_context(collect_list=True)
    .on_output()
    .on_input()
)
# Question/answer relevance between overall question and answer.
f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
    #Feedback(answer_relevance_ollama, name="Answer Relevance")
    .on_input()
    .on_output()
)

# Context relevance between question and each context chunk.
f_context_relevance = (
    Feedback(
        provider.context_relevance_with_cot_reasons, name="Context Relevance"
    )
    #Feedback(context_relevance_ollama, name="Context Relevance")
    .on_input()
    .on_context(collect_list=False)
    .aggregate(np.mean)  # choose a different aggregation method if you wish
)

#######

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_similarity(query, response):
    query_emb = embedder.encode(query, convert_to_tensor=True)
    resp_emb = embedder.encode(str(response), convert_to_tensor=True)
    return util.pytorch_cos_sim(query_emb, resp_emb).item()

# Wrap in Feedback object
feedback = Feedback(semantic_similarity, name="Semantic Similarity").on_input_output()

tru_query_engine_recorder = TruLlama(
    query_engine,
    app_name="LlamaIndex_Ollama",
    #feedbacks=[feedback]
    feedbacks=[
        f_groundedness, 
        f_answer_relevance, 
        f_context_relevance
    ],
)
# or as context manager
# with tru_query_engine_recorder as recording:
#     response = query_engine.query("Wie viele Abteilungen hat ilume?")
#     print("\n💬 Antwort:\n", response)

#     recording.wait_for_feedback_results()

questions = [
    "Was mache ich wenn ich krank werde?",
    "Wie kann ich eine Reise mit der Bahn buchen?"
    #"Wie viele Abteilungen hat ilume?",
    #"Wie viele Mitarbieter arbeiten bei ilume?",
    #"Wo wurde das Pflichtpraktikum gemacht?",
    #"Wo befindet sich der Hauptsitz von ilume?",
    #"Wann wurde das Pflichtpraktikum gemacht?"
]


for q in questions:

    #q = questions[2]
    with tru_query_engine_recorder as recording:
        response = query_engine.query(q)
        print("################################")
        print(f"{q}:")
        print("Antwort:", response)
        print("################################")
        




# record = recording.get()

# for result in record.feedback_results:
#     if result.result is not None and result.result < 0.5:
#         print(f"Low {result.name} score ({result.result:.2f}) for: {q}")


# while True:
#     query = input("Your Question: ")
#     if query.strip().lower() == "exit":
#         break

#     with tru_query_engine_recorder as recording:
#         response = query_engine.query(query)
    
#         print("################################")
#         print("\nAntwort:\n", response)
#         recording.retrieve_feedback_results(10)

        # if record is not None:
        #     print(record.feedback_results)


time.sleep(190)
print("Fertig")
