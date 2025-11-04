from llama_index.core.schema import TextNode
from llama_index.core.node_parser import JSONNodeParser
from llama_index.core import Document
import requests
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser, HierarchicalNodeParser
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
from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes
from llama_index.core.retrievers import AutoMergingRetriever



EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   
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


#document = Document(text=text)

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
#document = Document(text=text)
#print(response.text)

# Initialize the JSONNodeParser
#parser = JSONNodeParser()
#parser = SimpleFileNodeParser()

#parser = LangchainNodeParser(RecursiveCharacterTextSplitter())

QUESTION_GEN_PROMPT = PromptTemplate(
    "Lies den folgenden Textausschnitt aufmerksam durch.\n"
    "Erstelle dann **genau drei Fragen auf Deutsch**, die sich **direkt und ausschließlich** auf die im Text enthaltenen Fakten beziehen.\n"
    "- Die Fragen müssen **konkret beantwortbar** durch diesen Text sein.\n"
    "- Keine hypothetischen, interpretierenden oder weiterführenden Fragen.\n"
    "- Gib **nur** die drei Fragen zurück, nummeriert von 1 bis 3, keine Kommentaren von dir.\n\n"
    "Text:\n{chunk}\n\n"
    "Fragen:\n1."
)

def generate_questions_for_chunk(llm, chunk: str) -> list[str]:
    prompt = QUESTION_GEN_PROMPT.format(chunk=chunk)
    response = llm.complete(prompt)  # or llm.predict(prompt), depending on your LLM wrapper
    raw = response.text.strip()

    # Parse questions — assumes format: "1. question\n2. question\n3. question"
    questions = [line.split(".", 1)[1].strip() for line in raw.splitlines() if line.strip().startswith(tuple("123"))]
    return questions[:3]

def create_question_nodes(original_nodes, llm):
    question_nodes = []

    for original_node in original_nodes:
        chunk_text = original_node.text
        original_metadata = original_node.metadata or {}
        original_id = original_node.node_id

        questions = generate_questions_for_chunk(llm, chunk_text)

        print(f"Chunk:\n{chunk_text}")
        print(questions)

        for i, q in enumerate(questions):
            new_node = TextNode(
                text=q,
                metadata={
                    **original_metadata,               
                    "source_node_id": original_id,     
                    "source_chunk": chunk_text         
                },
                id_=f"{original_id}_q{i+1}"            # unique id per question
            )
            question_nodes.append(new_node)

    return question_nodes

def clean_text(text: str) -> str:
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
             # restore paragraph breaks
    return text

def load_docs():
    documents = SimpleDirectoryReader(PDF_DIR).load_data()
    document = Document(text="\n\n".join([doc.text for doc in documents]))

    document.text

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


def create_sentence_window_splitter():

    sentence_window = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        #original_text_metadata_key="original_text",
        include_metadata=True
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
    
    text_splitter = SentenceSplitter(separator="\n\n", chunk_size=512, chunk_overlap=50)
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
        breakpoint_percentile_threshold=95, 
        embed_model=Settings.embed_model,
        show_progress=True,
        include_metadata=True,
    )
    

    #semantic_nodes = semantic_splitter.get_nodes_from_documents(documents)

    #return semantic_nodes
    return semantic_splitter



    


def create_pipeline(documents):
    semantic_splitter = create_semantic_splitter()
    sentence_window = create_sentence_window_splitter()
    sentence_splitter = create_sentence_splitter()
    
    #question_extractor = QuestionsAnsweredExtractor(questions_per_chunk=3)
    #keyword_extractor = KeywordExtractor()


    # pipeline = IngestionPipeline(transformations=[
    #     sentence_splitter,
    #     #sentence_window,
    #     semantic_splitter,
    #     #question_extractor,
    #     #keyword_extractor
    # ])

    # nodes = pipeline.run(
    #     documents=documents,
    #     in_place = True,
    #     show_progress = True
    # )

    node_parser = HierarchicalNodeParser.from_defaults()
    nodes = node_parser.get_nodes_from_documents(documents)
    len(nodes)

    leaf_nodes = get_leaf_nodes(nodes)
    len(leaf_nodes)

    root_nodes = get_root_nodes(nodes)
    len(root_nodes)

    print(f"NODES len: {len(nodes)}")
    for i, node in enumerate(nodes):
            print(f"--- Node {i+1} ---")
            print(node.text)  # print first 300 chars
            print()

    print("-/-/"*20)
    return nodes, leaf_nodes, root_nodes


def build_vectore_store(nodes):
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    croma_collection = db.get_or_create_collection(COLLECTION_NAME_SPLIT)
    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=croma_collection)
    vector_store.add(nodes)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    #print(f"Total embeddings stored: {croma_collection.count()}")
    #print(croma_collection.peek(1))

    return storage_context

def build_storage_context(leaf_nodes, all_nodes):
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME_SPLIT)

    # Create Chroma vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Add only leaf nodes to Chroma
    vector_store.add(leaf_nodes)

    # Store all nodes (roots + leaves) in docstore
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    storage_context.docstore.add_documents(all_nodes)

    return storage_context


def build_index(storage_context: StorageContext):
    #return VectorStoreIndex(nodes)

    question_nodes = []
    #question_nodes = create_question_nodes(nodes, llm)
    if question_nodes:
        new_nodes = question_nodes + nodes
    else:
        new_nodes = nodes

    index = VectorStoreIndex.from_documents(
        #question_nodes,
        #documents=new_nodes,
        vector_store=storage_context.vector_store, # ??????
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    #return index, new_nodes
    return index

def create_hybrid_retriever(index, nodes):


    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=7)
    
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=7)
    #sparse_retriever = BM25Retriever.from_documents(documents, similarity_top_k=8)

    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        mode="reciprocal_rerank",
        similarity_top_k=8
    )

   
    reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=2)
    #postprocessors = [reranker, SimilarityPostprocessor(similarity_cutoff=0.75)]

    return fusion_retriever, reranker


def create_auto_merge_retr(index: VectorStoreIndex, storage_context: StorageContext ):
    base_retriever = index.as_retriever(similarity_top_k=6)
    retriever = AutoMergingRetriever(
        base_retriever, 
        storage_context, 
        verbose=True
    )

    return retriever


def query_pipeline(index, documents, questions: list[str]):
    retriever, postprocessor = create_hybrid_retriever(index, documents)


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
            #print(node.source_chunk)
            print(">> Source Chunk:", node.metadata.get("source_chunk"))
            print()

        results = postprocessor.postprocess_nodes(retrieved_nodes, query_str=q)
            
         

        print("########")
        print(f"> Retrieved {len(results)} nodes   after reranking.")
        if results:
            for i, node in enumerate(results):
                #text = meta.get("original_text") or meta.get("text") or node.text
                print(f"---- {i+1}")
                print(node.text) 
                print(">> Source Chunk:", node.metadata.get("source_chunk"))
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

    # questions = [
    #     "Wie viele Abteilungen hat ilume?",
    #     "Wie viele Mitarbieter arbeiten bei ilume?",
    #     "Wo wurde das Pflichtpraktikum gemacht?",
    #     "Wo befindet sich der Hauptsitz von ilume?",
    #     "Wann wurde das Pflichtpraktikum gemacht?",
    #     "Wann wurde ilume gegründet?"
    # ]

    questions = [
        "Was mache ich wenn ich krank bin?",
        "Ich bin staatlich versichert, wie melde ich mich krank?",
        "Reise mit der Bahn"
    ]


    # load + clean documents → from pdf or raw text
    docs = load_docs()

    # split docs into chunks get nodes
    nodes, leaf_nodes, root_nodes = create_pipeline(documents=docs)

    #storage_context = build_vectore_store(leaf_nodes)
    storage_context = build_storage_context(leaf_nodes, nodes)
    # 
    #index, question_nodes = build_index(nodes=nodes)
    index = build_index(storage_context)

    merge_retr = create_auto_merge_retr(index, storage_context)

    questions = [
        "Was mache ich wenn ich krank bin?",
        "Ich bin staatlich versichert, wie melde ich mich krank?",
        "Reise mit der Bahn"
    ]

    for q in questions:
        print(f"\n### {q}")
        results = merge_retr.retrieve(q)
        for i, node in enumerate(results):
            print(f"\n--- Node {i+1} ---")
            print(node.text[:300])
            print("Metadata:", node.metadata)

    # query_engine = index.as_query_engine(llm=llm)

    # for q in questions:
        
    #     response = query_engine.query(q)
    #     print("################################")
    #     print(f"{q}:")
    #     print("Antwort:", response)
    #     print(response.source_nodes)
    #     print("################################")

    #query_pipeline(index, question_nodes, questions)




