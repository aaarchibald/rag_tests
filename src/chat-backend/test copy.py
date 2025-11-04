from typing import TypedDict
# from langchain_ollama import OllamaLLM, OllamaEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
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

text = """Im Zeitraum vom 01.03.2025 – 31.08.2025 habe ich mein Pflichtpraktikum bei der 
ilume Informatik AG in Mainz absolviert. Es handelt sich um ein IT-Beratungshaus, das 
sich  auf  CRM-Systeme,  die  Digitalisierung  von  Geschäftsprozessen  sowie  Cloud-
Lösungen spezialisiert. Neben der Integration von Standardsoftware in bestehende IT-
Landschaften entwickelt ilume auch individuelle Softwarelösungen für Kunden aus 
verschiedensten Branchen, darunter Pharma, Chemie, Bankwesen, Versicherungen 
und Logistik. 
Das Unternehmen wurde im Jahr 2000 gegründet und beschäftigt mittlerweile rund 
150  Mitarbeitende,  die  in  drei  Abteilungen  organisiert  sind:  DEV  (Custom 
Development),  SMA  (Smart  Automation)  und  CRM  (Customer  Relationship 
Management). Während meines Praktikums war ich im Bereich DEV tätig und dort 
dem  Team  Digitalization  zugeordnet,  das  sich  vor  allem  mit  Web-  und  Mobile-
Entwicklung  beschäftigt  und  sowohl  externe  Kunden-  als  auch  interne  Projekte 
umsetzt. 
Vor dem Pflichtpraktikum war ich bereits als Werkstudentin bei ilume tätig, wodurch ich 
mit  vielen  Unternehmensprozessen  vertraut  war  und  mein  Praktikum  effizient 
beginnen  konnte.  Etwa  5–10 %  meiner Aufgaben  lagen  im  Content-Management-
Bereich, in dem ich sowohl Magnolia CMS-Seiten als auch AWS-basierte Webseiten 
für  Kunden  aus  der  Pharma-Branche  betreute.  Dabei  passte  ich  Inhalte  an, 
implementierte Änderungen an Funktionen und übernahm die Bildbearbeitung. Ein 
wesentlicher Bestandteil meiner Tätigkeit war die Mitarbeit am internen KI-basierten 
Webprojekt „Blog Writer“. In den letzten beiden Monaten, von Juli bis August, lag mein 
Schwerpunkt auf der Einarbeitung in Compose Multiplatform."""

#docs = [Document(text=text)]


embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = Ollama(
    #model="llama3.2:1b",
    model="llama3.2:latest",
    request_timeout=120.0 
)

# embeddings = embed_model.get_text_embedding("Hello World!")
# print(len(embeddings))
# print(embeddings[:5])

pdf_dir = "./pdfs"
docs = SimpleDirectoryReader(pdf_dir).load_data()

for doc in docs:
    print(doc.get_content())
    print(len(doc.get_content()))
    print("---------")

text_splitter = SentenceSplitter(separator=" ", chunk_size=512, chunk_overlap=50)
title_extractor = TitleExtractor(llm=llm, nodes=5)
qa_extractor = QuestionsAnsweredExtractor(llm=llm, questions=3)





# transofrm docs
pipeline = IngestionPipeline(
    transformations=[
        text_splitter,
        title_extractor,
        qa_extractor
    ],
    #vector_store = vector_store,
)

pipeline.persist("./pipeline_storage1")

nodes = pipeline.run(
    documents=docs,
    in_place = True,
    show_progress = True
)
print(len(nodes))
print(nodes[1].get_content(metadata_mode=MetadataMode.EMBED))

pprint.pprint(nodes[1].__dict__)



db = chromadb.PersistentClient(path="./chroma_db1")
croma_collection = db.get_or_create_collection("faqs")
# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=croma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


index = VectorStoreIndex.from_documents(
    nodes, 
    storage_context=storage_context, 
    embed_model=embed_model,
    #transformations=[text_splitter],
    #show_progress=True
)

query_engine = index.as_query_engine(
    llm=llm,
    #text_qa_template=qa_template
)

#response = query_engine.query("Bei welcher Firma in Mainz war der Pflichtpraktikum?")
#question = "Wie viele Mitarbeiter arbeiten bei ilume?"
# response = query_engine.query(question)
# print(response)

retriever = index.as_retriever(similarity_top_k=3)
nodes = retriever.retrieve("Wie viele Mitarbeiter arbeiten bei ilume?")

# print("-----------------")
# for node in nodes:
#     print(node.get_content())



# while True:
#     query = input("Ask a question about the PDF (or type 'exit'): ")
#     if query.strip().lower() == "exit":
#         break
    
#     result = retriever.retrieve(query)
#     # result = query_engine.query(query)
#     print("\nAnswer:", result, "\n")
#     # result2 = retriever.retrieve(query)
#     # print("\nAnswer:", result2, "\n")



from trulens_eval import TruLlama, Feedback, Tru
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Create a local embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_similarity(input_text, output_text):
    emb1 = embedder.encode(input_text, convert_to_tensor=True)
    emb2 = embedder.encode(output_text, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

# Wrap it in a Feedback object
similarity_feedback = Feedback(semantic_similarity, name="Semantic Similarity").on_input_output()


# Create TruLlama wrapper
tru_query_engine = TruLlama(
    query_engine=query_engine,
    app_id="ollama-pdf-app",
    feedbacks=[similarity_feedback]
)

# Optional: Initialize Tru (only once per session or script)
tru = Tru()


while True:
    query = input("Ask a question about the PDF (or type 'exit'): ")
    if query.strip().lower() == "exit":
        break

    # Automatically records inputs, outputs, and feedback
    with tru_query_engine as recording:
        response = tru_query_engine.query(query)

    print("\nAnswer:", response, "\n")



# Settings.llm = llm
# Settings.embed_model = embed_model
# Settings.text_splitter = text_splitter



#pprint.pprint(docs)
# for doc in documents:
#     print("Here")
#     print(doc.text[:500])  # Vorschau: ersten 500 Zeichen







# qa_template = PromptTemplate(
#     "Beantworte die folgende Frage so genau wie möglich basierend auf dem bereitgestellten Kontext.\n\n"
#     "Kontext:\n{context_str}\n\n"
#     "Frage: {query_str}\n"
#     "Antwort:"
# )




# retriever = index.as_retriever(similarity_top_k=5)
# retrieved_nodes = retriever.retrieve(question)

# print("\n🔍 Top 5 Chunks vom Retriever:")
# for i, node in enumerate(retrieved_nodes):
#     print(f"\n--- Chunk {i+1} ---")
#     print(node.get_content())

