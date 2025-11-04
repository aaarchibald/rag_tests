from llama_index.core.node_parser import JSONNodeParser
from llama_index.core import Document
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.core.node_parser import SentenceWindowNodeParser, SemanticSplitterNodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import SimilarityPostprocessor, SentenceTransformerRerank, SentenceEmbeddingOptimizer
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.ingestion import IngestionPipeline
import chromadb
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

text = """
Im Zeitraum vom 01.03.2025 – 31.08.2025 habe ich mein Pflichtpraktikum bei der 
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
Schwerpunkt auf der Einarbeitung in Compose Multiplatform.

1. Woche (03.03. – 07.03.2025) 
In der ersten Woche meines Praktikums fand ein Meeting mit dem Management des 
DEV-Teams statt, bei dem das Thema für das Projekt vorgestellt wurde. Die Grundidee 
war, ein Tool zu entwickeln, das täglich aktuelle Nachrichtenartikel sammelt, diese 
anhand der individuellen Interessen eines ilume-Mitarbeiters bewertet und daraus KI-
gestützte  LinkedIn-Beiträge  generiert.  Diese  Beiträge  werden  per  E-Mail  an  die 
Mitarbeiter geschickt und können dort für LinkedIn genutzt werden.  
Das Projekt wurde in einem Zweierteam umgesetzt, wobei ich die Verantwortung für 
die Backend-Entwicklung übernahm. Zur Abstimmung führten wir tägliche Meetings 
(Dailys)  durch,  in  denen  wir  den  Projektfortschritt  sowie  offene  Fragen  und 
Herausforderungen besprachen.  
Als  ersten  Schritt  erarbeitete  ich  einen  Fragebogen  für  die  Kollegen,  der  die 
notwendigen Profilinformationen abfragte und es der KI dadurch erleichterte, Artikel 
gezielt den jeweiligen Interessen zuzuordnen.  

"""

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # CPU-friendly
OLLAMA_MODEL = "llama3.2"
COMPANY_NAME = "ilume"
PDF_DIR = "./pdfs"
CHROMA_PATH = "./chroma_db1"
COLLECTION_NAME_WINDOW = "faqs_window"
COLLECTION_NAME_SPLIT = "faqs_split"


embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL, 
    #max_length=512
)
llm = Ollama(model="llama3.2:latest", request_timeout=60.0)


Settings.llm = llm
Settings.embed_model = embed_model
#Settings.text_splitter = text_splitter


# url = "https://housesigma.com/bkv2/api/search/address_v2/suggest"

# payload = {"lang": "en_US", "province": "ON", "search_term": "Mississauga, ontario"}

# headers = {
#     'Authorization': 'Bearer 20240127frk5hls1ba07nsb8idfdg577qa'
# }
#response = requests.post(url, headers=headers, data=payload)


# Create a Document object with the JSON response
document = Document(text=text)
#print(response.text)

# Initialize the JSONNodeParser
#parser = JSONNodeParser()
#parser = SimpleFileNodeParser()

#parser = LangchainNodeParser(RecursiveCharacterTextSplitter())


def load_docs():
    documents = SimpleDirectoryReader(PDF_DIR).load_data()
    document = Document(text="\n\n".join([doc.text for doc in documents]))

    #return documents
    return [document]


def create_sentence_window_splitter():

    sentence_window = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    # Parse nodes from the JSON document
    #nodes = sentence_window.get_nodes_from_documents(documents)

    return sentence_window
    #sentence_index = VectorStoreIndex(nodes)

    # query_engine = sentence_index.as_query_engine(
    # similarity_top_k=2,
    # # the target key defaults to `window` to match the node_parser's default
    # node_postprocessors=[
    #     MetadataReplacementPostProcessor(target_metadata_key="window")
    # ],
    # )
    # window_response = query_engine.query(
    #     "Wie viele Mitarbeiter hat ilume?"
    # )
    # print(window_response)
    # window = window_response.source_nodes[0].node.metadata["window"]
    # sentence = window_response.source_nodes[0].node.metadata["original_text"]

    # print(f"##Window: {window}")
    # print("------------------")
    # print(f"##Original Sentence: {sentence}")


def create_sentence_splitter():
    
    text_splitter = SentenceSplitter(separator=" ", chunk_size=512, chunk_overlap=50)
    return text_splitter

    #base_nodes = text_splitter.get_nodes_from_documents(documents)

    # base_index = VectorStoreIndex(base_nodes)

    # print("##################")
    # query_engine = base_index.as_query_engine(similarity_top_k=2)
    # vector_response = query_engine.query(
    #     "Wie viele Mitarbeiter hat ilume?"
    # )
    # print(vector_response)
    # base_retriever = base_index.as_retriever



def create_semantic_splitter():

    semantic_splitter = SemanticSplitterNodeParser(
        buffer_size=1, 
        breakpoint_percentile_threshold=90, 
        embed_model=Settings.embed_model,
        show_progress=True,
        include_metadata=True,
    )
    

    #semantic_nodes = semantic_splitter.get_nodes_from_documents(documents)

    #return semantic_nodes
    return semantic_splitter


def build_vectore_store():
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    croma_collection = db.get_or_create_collection(COLLECTION_NAME_SPLIT)
    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=croma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    #print(f"Total embeddings stored: {croma_collection.count()}")
    #print(croma_collection.peek(1))

    return storage_context
    

def build_index(nodes):
    #return VectorStoreIndex(nodes)

    index = VectorStoreIndex.from_documents(
        nodes, 
        storage_context=build_vectore_store(), 
        embed_model=embed_model,
        #transformations=[text_splitter],
        show_progress=True
    )
    return index

def create_pipeline(documents: list[Document]):
    semantic_splitter = create_semantic_splitter()
    sentence_window = create_sentence_window_splitter()
    sentence_splitter = create_sentence_splitter()
    
    #question_extractor = QuestionsAnsweredExtractor(questions_per_chunk=3)
    #keyword_extractor = KeywordExtractor()


    pipeline = IngestionPipeline(transformations=[
        sentence_splitter,
        #sentence_window,
        #semantic_splitter,
        #question_extractor,
        #keyword_extractor
    ])

    nodes = pipeline.run(
        documents=documents,
        in_place = True,
        show_progress = True
    )


    print(f"NODES len: {len(nodes)}")
    # for i, node in enumerate(nodes):
    #         print(f"--- Node {i+1} ---")
    #         print(node.text[:300])  # print first 300 chars
    #         print(node.score)
    #         print()

    # print("-/-/"*20)
    return nodes

def create_hybrid_retriever(index, nodes):


    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=4)
    
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=4)
    #sparse_retriever = BM25Retriever.from_documents(documents, similarity_top_k=8)

    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        mode="reciprocal_rerank"
    )

   
    reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=2)
    #postprocessors = [reranker, SimilarityPostprocessor(similarity_cutoff=0.75)]

    return fusion_retriever, reranker




def query_pipeline(index, documents, questions: list[str]):
    retriever, postprocessor = create_hybrid_retriever(index, documents)
    

    # Hybrid retrieval

    for q in questions:
        #vector_response = query_engine.query(q)
        print("##################")
        print(f"Frage: {q}")
        #retrieved_nodes = hybrid_retriever.retrieve(q)
        retrieved_nodes = retriever.retrieve(q)

        print(f"> Retrieved {len(retrieved_nodes)} nodes before reranking:")
        for i, node in enumerate(retrieved_nodes):
            print(f"--- Node {i+1} ---")
            print(node.text)  # print first 300 chars
            #print(node.score)
            print()

        results = postprocessor.postprocess_nodes(retrieved_nodes, query_str=q)
            
         

        print("########")
        print(f"> Retrieved {len(results)} nodes   after reranking.")
        if results:
            for i, node in enumerate(results):
                meta = node.metadata or {}
                text = meta.get("original_text") or meta.get("text") or node.text
                print(f"---- {i+1}")
                print(text)
                print("********")

        print("-"*25)   
            #print(vector_response)

        # nodes = base_retriever.retrieve(q)

        # print("-----------------")
        # for i, node in enumerate(nodes):
            
        #     print(f"----{i}")
        #     print(node.get_content())
        #     print("********")

    




if __name__ == "__main__":

    questions = [
        "Wie viele Abteilungen hat ilume?",
        "Wie viele Mitarbieter arbeiten bei ilume?",
        "Wo wurde das Pflichtpraktikum gemacht?",
        "Wo befindet sich der Hauptsitz von ilume?",
        "Wann wurde das Pflichtpraktikum gemacht?",
        "Wann wurde ilume gegründet?"
    ]

    docs = load_docs()

    #nodes = create_semantic_splitter(docs)
    nodes = create_pipeline(documents=docs)
    #splitter = create_semantic_splitter()
    #nodes = splitter.get_nodes_from_documents(docs)

    #sent_nodes = create_sentence_window_splitter()
    #nodes = sent_nodes.get_nodes_from_documents(docs)

    #splitter = create_sentence_splitter()
    #nodes = splitter.get_nodes_from_documents(docs)

    index = build_index(nodes=nodes)

    # query_engine = index.as_query_engine(llm=llm)

    # for q in questions:
        
    #     response = query_engine.query(q)
    #     print("################################")
    #     print(f"{q}:")
    #     print("Antwort:", response)
    #     print(response.source_nodes)
    #     print("################################")

    query_pipeline(index, nodes, questions)

# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.response_synthesizers import CompactAndRefine

# retriever = index.as_retriever(similarity_top_k=4)
# response_synthesizer = CompactAndRefine.

# query_engine = RetrieverQueryEngine.from_args(
#     retriever=retriever,
#     response_synthesizer=response_synthesizer
# )


