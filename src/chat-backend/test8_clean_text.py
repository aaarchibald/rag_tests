from llama_index.core.schema import TextNode
from llama_index.core.node_parser import JSONNodeParser
from llama_index.core import Document
import requests
import re
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
from llama_index.core.prompts import PromptTemplate
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


document = Document(text=text)


def clean_text(text: str) -> str:
    # text = re.sub(r'-\s*\n\s*', '', text)               # fix hyphenated line breaks
    # text = re.sub(r'\n{2,}', '[PARAGRAPH]', text)       # mark real paragraph breaks
    # text = re.sub(r'\n', ' ', text)                     # remove single line breaks
    # text = re.sub(r'\s+', ' ', text)                    # collapse whitespace
    # text = text.replace('[PARAGRAPH]', '\n\n')          # restore paragraph breaks
    
    text = re.sub(r'Nur zur internen Verwendung', '', text)
    text = re.sub(r'[0-9] ', '', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'[ \t]*\n[ \t]*', '\n', text) # newline + space combinations to new line
    text = re.sub(r'\n{2,}', '[PARAGRAPH]', text) 
    #text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text) 
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    text = text.replace('[PARAGRAPH]', '\n\n') 
    text = text.strip()

    # for i, c in enumerate(text):
    #     if c.isspace():
    #         print(f"{i}: WHITESPACE -> {repr(c)} | ord: {ord(c)}")
    #     else:
    #         print(f"{i}: {c} | ord: {ord(c)}")
    return text

def load_docs():
    documents = SimpleDirectoryReader(PDF_DIR).load_data()
    document = Document(text="\n\n".join([doc.text for doc in documents]))

    #document.text

    #document = Document(text=text)

    # Clean the text but keep metadata intact
    cleaned_text = clean_text(document.text)
    

    doc = Document(
        text=cleaned_text,
        metadata=document.metadata  
    )
    print(doc.text)

    #return documents
    return [doc]


load_docs()