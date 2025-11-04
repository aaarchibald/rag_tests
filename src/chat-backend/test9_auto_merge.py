import asyncio
from collections import defaultdict
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import JSONNodeParser
from llama_index.core import Document
import requests
import re
import os
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
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.storage.docstore import SimpleDocumentStore, BaseDocumentStore, DocumentStore
from llama_index.core import load_index_from_storage

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from llama_index.core.evaluation import DatasetGenerator, QueryResponseDataset
from llama_index.core.llama_dataset.generator import RagDatasetGenerator

from llama_index.core.evaluation import (
    CorrectnessEvaluator,
    SemanticSimilarityEvaluator,
    RelevancyEvaluator,
    FaithfulnessEvaluator,
    PairwiseComparisonEvaluator,
)
from llama_index.core.evaluation import RetrieverEvaluator

from collections import defaultdict
import pandas as pd
from llama_index.core.evaluation.eval_utils import (
    get_responses,
    get_results_df,
)
from llama_index.core.evaluation import BatchEvalRunner







#EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   
#Qwen/Qwen3-Embedding-0.6B
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B" # BAAI/bge-small-en-v1.5
OLLAMA_MODEL = "llama3.2"
COMPANY_NAME = "ilume"
PDF_DIR = "./pdfs"
CHROMA_PATH = "./chroma_db1"
COLLECTION_NAME_WINDOW = "faqs_window"
COLLECTION_NAME_SPLIT = "faqs_split"
STORAGE_DIR = "./storage"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B" # "BAAI/bge-reranker-base"

embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL, 
    #max_length=1024
)
llm = Ollama(model="llama3.2:latest", request_timeout=60.0)
#embed_model = "local:BAAI/bge-small-en-v1.5"
#embed_model = "local:Qwen/Qwen3-Embedding-4B"
#embed_model = "local:Qwen/Qwen3-Embedding-0.6B"
#document = Document(text=text)

Settings.llm = llm
Settings.embed_model = embed_model
#Settings.text_splitter = text_splitter


def clean_text(text: str) -> str:
    text = re.sub(r'Nur zur internen Verwendung', '', text)
    text = re.sub(r'\n{2,}', '[PARAGRAPH]', text) 
    #text = re.sub(r'\s[0-9]\s', '', text)
    #print(text)
    text = re.sub(r'\s*\n\s*', ' ', text)
    
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'[ \t]*\n[ \t]*', '\n', text) # newline + space combinations to new line
    
    #text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text) 
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    text = text.replace('[PARAGRAPH]', '\n\n') 
    text = text.strip()
             # restore paragraph breaks

    #print(text)
    return text



def load_docs():
    documents = SimpleDirectoryReader(PDF_DIR).load_data()
    #document = Document(text="\n\n".join([doc.text for doc in documents]))


    combined_docs = defaultdict(str)
    combined_metadata = {}

    for doc in documents:
        doc_name = doc.metadata.get('file_name')
        #text = doc.text + "\n\n"
        combined_docs[doc_name] += doc.text + "\n\n"
        #print(combined_docs[doc_name])
        
        if doc_name not in combined_metadata:
            combined_metadata[doc_name] = doc.metadata

    docs_per_file = [
        Document(text=clean_text(text.strip()), metadata=combined_metadata[fname])
        for fname, text in combined_docs.items()
    ]


    print(len(docs_per_file))
    #document = Document(text=text)

    for doc in docs_per_file:
        print(doc.text)
        print("-----------")


    return docs_per_file


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


# def create_sentence_splitter():
    
#     text_splitter = SentenceSplitter(separator="\n\n", chunk_size=512, chunk_overlap=50)
#     return text_splitter



def create_semantic_splitter():

    semantic_splitter = SemanticSplitterNodeParser(
        buffer_size=1, 
        breakpoint_percentile_threshold=95, 
        embed_model=Settings.embed_model,
        show_progress=True,
        include_metadata=True,
    )
    
    return semantic_splitter




def create_hierarchical_node_parser(documents):

   # node_parser = HierarchicalNodeParser.from_defaults()
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 512, 128],  # larger chunks at top levels
        include_metadata=False          # don’t add excessive metadata text
    )

    nodes = node_parser.get_nodes_from_documents(documents)
    
    print(f" NODES: {len(nodes)}")

    leaf_nodes = get_leaf_nodes(nodes)
    print(f"Leaf NODES: {len(leaf_nodes)}")

    print("---------")
    nodes_by_id = {node.node_id: node for node in nodes}
    print(f"Leaf NODE 1: {leaf_nodes[1].node_id} {leaf_nodes[1].text}")

    parent_node = nodes_by_id[leaf_nodes[1].parent_node.node_id]
    print(parent_node.text)
    print("---------")

    print(f"Leaf NODE 2: {leaf_nodes[2].node_id} {leaf_nodes[2].text}")

    parent_node = nodes_by_id[leaf_nodes[2].parent_node.node_id]
    print(parent_node.text)
    print("---------")
    root_nodes = get_root_nodes(nodes)
    print(f"Root NODES: {len(root_nodes)}")


    return nodes, leaf_nodes, root_nodes




def build_storage_context(leaf_nodes, all_nodes):

    docstore = SimpleDocumentStore()

    # insert nodes into docstore
    docstore.add_documents(all_nodes)


    db = chromadb.PersistentClient(path=CHROMA_PATH)
    croma_collection = db.get_or_create_collection(COLLECTION_NAME_SPLIT)
    vector_store = ChromaVectorStore(chroma_collection=croma_collection)


    storage_context = StorageContext.from_defaults(
        docstore=docstore,
        vector_store=vector_store
    )
    
    return storage_context


def build_index(leaf_nodes, storage_context: StorageContext):
    #return VectorStoreIndex(nodes)

   

    index = VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    # index = VectorStoreIndex.from_documents(
    #     leaf_nodes, 
    #     storage_context=storage_context, 
    #     embed_model=embed_model,
    # )

    return index

def build_or_load_index(docs, leaf_nodes, all_nodes):
    # Try loading from disk

    if os.path.exists(STORAGE_DIR):
        print(" Loading cached index from disk...")
        db = chromadb.PersistentClient(path=CHROMA_PATH)
        croma_collection = db.get_or_create_collection(COLLECTION_NAME_SPLIT)
        vector_store = ChromaVectorStore(chroma_collection=croma_collection)
        storage_context = StorageContext.from_defaults(
            persist_dir=STORAGE_DIR,
            vector_store=vector_store
        )
        
        index = load_index_from_storage(storage_context)
        print(" Cached index loaded successfully.")
        return index, storage_context
    
    # Otherwise build fresh
    print("Building new index...")
    storage_context = build_storage_context(leaf_nodes, all_nodes)
    index = VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    storage_context.persist(persist_dir=STORAGE_DIR)
    print(" Index persisted to disk.")
    return index, storage_context

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
    base_retriever = index.as_retriever(similarity_top_k=12)
    
    retriever = AutoMergingRetriever(
        base_retriever, 
        storage_context, 
        verbose=True # prints merging steps
    )

    return retriever, base_retriever

async def evaluate(nodes, index):
    eval_llm = llm
    dataset_generator = RagDatasetGenerator(
        #root_nodes[:20],
        nodes,
        llm=eval_llm,
        show_progress=True,
        num_questions_per_chunk=3,
    )
    #eval_dataset = await dataset_generator.agenerate_dataset_from_nodes(num=60)

    # eval_questions = dataset_generator.generate_questions_from_nodes(num=10)

    # print("🧪 Sample Evaluation Questions:")
    # for q in eval_questions[:3]:
    #     print("-", q)

    # eval_questions = [
    #     "Wie melde ich mich krank?",
    #     "Was mache ich bei einer Bahnreise im Projekt?",
    #     "Wann muss ich die Krankmeldung einreichen?"
    # ]

    evaluator_c = CorrectnessEvaluator(llm=eval_llm)
    evaluator_s = SemanticSimilarityEvaluator(embed_model)
    # evaluator_r = RelevancyEvaluator(llm=eval_llm)
    # evaluator_f = FaithfulnessEvaluator(llm=eval_llm)

    retriever_results = {}
    eval_questions = {
        "q1": "Wie melde ich mich krank?",
        "q2": "Was mache ich bei einer Bahnreise im Projekt?",
        "q3": "Wann muss ich die Krankmeldung einreichen?"
    }

    eval_responses = {
        "q1": """Im Falle einer Krankheit/Arbeitsunfähigkeit unverzüglich bzw. bis spätestens 9 Uhr (am ersten Krankheitstag) krankmelden 
                 und die voraussichtliche Dauer mitteilen. 
                 Die Krankmeldung erfolgt über das ilumeCRM (Infomail wird vom System automatisch an das ilume-Personalteam und Teamleitenden geschickt) 
                 und ggf. per Mail an den Projektleitenden. (Alternativ per Mail an personal@ilume.de, Teamleitenden und ggf. Projektleitenden).""",

        "q2": """Wird ein Bahnticket benötigt, muss dieses bitte im ilume-Office angefragt werden. 
                 Bei Fahrten mit dem Fernverkehr wird das Ticket vom ilume-Office gebucht. 
                 Fahrten im Nahverkehr können bei zu kurzer Entfernung (z. B. Mainz–Frankfurt) nicht vom ilume-Office gebucht werden 
                 und müssen vom Mitarbeitenden direkt am Ticketautomaten gebucht werden. 
                 Gebucht werden vom ilume-Office immer Flextickets, sodass eine frühere/spätere An- und Abreise möglich ist.""",

        "q3": "Die Krankmeldung erfolgt am gleichen Tag bis spätestens 9 Uhr."
    }

    #eval_dataset: QueryResponseDataset = dataset_generator.generate_dataset_from_nodes(num=10)
    eval_dataset = QueryResponseDataset(
        queries=eval_questions,
        responses= eval_responses # empty if not available
    )


    q1 = "Wie melde ich mich krank?"
    
    a1 = """Im Falle einer Krankheit/Arbeitsunfähigkeit unverzüglich bzw. bis spätestens 9 Uhr (am ersten Krankheitstag) krankmelden und die voraussichtliche Dauer mitteilen. 
                 Die Krankmeldung erfolgt über das ilumeCRM (Infomail wird vom System automatisch an das ilume-Personalteam und Teamleitenden geschickt) 
                 und ggf. per Mail an den Projektleitenden. (Alternativ per Mail an personal@ilume.de, Teamleitenden und ggf. Projektleitenden)."""


    #for name, index in indexes.items():
    retriever = index.as_retriever(similarity_top_k=5)
    results = await evaluator_s.aevaluate(query=q1, response=a1)
    # evaluator = RetrieverEvaluator.from_metric_names(
    #     ["recall", "precision", "mrr"], retriever=retriever
    # )
    # #print(f"\n🔍 Evaluating {name} retriever...")
    # results = await evaluator.aevaluate_dataset(eval_dataset)
    #retriever_results[name] = results[name]
    print(results)


if __name__ == "__main__":

    evaluator_s = SemanticSimilarityEvaluator(Settings.embed_model)
    a1 = """Im Falle einer Krankheit/Arbeitsunfähigkeit unverzüglich bzw. bis spätestens 9 Uhr (am ersten Krankheitstag) krankmelden und die voraussichtliche Dauer mitteilen. 
                 Die Krankmeldung erfolgt über das ilumeCRM (Infomail wird vom System automatisch an das ilume-Personalteam und Teamleitenden geschickt) 
                 und ggf. per Mail an den Projektleitenden. (Alternativ per Mail an personal@ilume.de, Teamleitenden und ggf. Projektleitenden)."""

    # load + clean documents → from pdf or raw text
    docs = load_docs()

    # split docs into chunks get nodes
    nodes, leaf_nodes, root_nodes = create_hierarchical_node_parser(documents=docs)

    #storage_context = build_vectore_store(leaf_nodes)
    #storage_context = build_storage_context(leaf_nodes, nodes)
    index, storage_context = build_or_load_index(docs, leaf_nodes, nodes)
    # 
    #index, question_nodes = build_index(nodes=nodes)
    #index = build_index(leaf_nodes, storage_context)

    merge_retr, base_retriever = create_auto_merge_retr(index, storage_context)

    questions = [
        "Was mache ich wenn ich krank bin?",
        #"Ich bin gesetzlich versichert, wie melde ich mich krank?",
        #"Ich bin Praktikantin, wie melde ich mich krank?",
        #"Reise mit der Bahn"
    ]

    # BAAI/bge-reranker-base
    #"cross-encoder/ms-marco-MiniLM-L-6-v2"
    # rerank = SentenceTransformerRerank(top_n=2, model="BAAI/bge-reranker-base") 

    # query_engine = RetrieverQueryEngine.from_args(
    #     merge_retr,
    #     node_postprocessors=[rerank]
    #     #llm=llm,
    #     #streaming=True
    # )
    # base_query_engine = RetrieverQueryEngine.from_args(base_retriever)



    # for q in questions:
    #     print(f"\n### {q}")

        
    #     nodes = merge_retr.retrieve(q)
    #     base_nodes = base_retriever.retrieve(q)



    #     print(f" Nodes Auto: {len(nodes)}")
    #     print(f" Base Nodes: {len(base_nodes)}")
        

    #     for i, node in enumerate(nodes):
    #         print(f"\n--- Node {i+1} ---")
    #         print("ID:", node.id_)
    #         print("Similarity:", node.score)
    #         print(len(node.text))
    #         print(node.text)

        # results = rerank.postprocess_nodes(nodes, query_str=q)

        # print("########")
        # print(f"> Retrieved {len(results)} nodes   after reranking.")
        # if results:
        #     for i, node in enumerate(results):
        #         #text = meta.get("original_text") or meta.get("text") or node.text
        #         print(f"---- {i+1}")
        #         print(node.text) 
        #         print("********")

        #print("-"*25)  

        # response = query_engine.query(q)
        # print(str(response))

        #print(f"\n--- /// ---")

    

        # base_response = base_query_engine.query(q)
        # print(str(base_response))

    #asyncio.run(evaluate(nodes, index))
    async def test_nodes():
        for q in questions:
            print(f"\n### {q}")

            # Retrieve nodes from both retrievers
            nodes = merge_retr.retrieve(q)
            base_nodes = base_retriever.retrieve(q)

            print(f" Nodes Auto: {len(nodes)}")
            print(f" Base Nodes: {len(base_nodes)}")

            # --- Evaluate Auto Merge Retriever ---
            if nodes:
                print("\n--- Evaluating Auto-Merge Retriever ---")
                total_score = 0.0

                for i, node in enumerate(nodes):
                    text = node.text.strip()
                    result = await evaluator_s.aevaluate(
                        response=text,
                        reference=a1  # your ground-truth reference answer
                    )

                    print(f"\nNode {i+1}:")
                    print(f"Similarity Score: {result.score:.4f}")
                    print(f"Length: {len(text)} chars")
                    print(f"Preview: {text}\n")

                    total_score += result.score

                avg_score = total_score / len(nodes)
                print(f"✅ Average Semantic Similarity (Auto): {avg_score:.4f}")
            # --- Evaluate Base Retriever ---
        if base_nodes:
            print("\n--- Evaluating Base Retriever ---")
            total_score = 0.0

            for i, node in enumerate(base_nodes):
                text = node.text.strip()
                result = await evaluator_s.aevaluate(
                    response=text,
                    reference=a1
                )

                print(f"\nNode {i+1}:")
                print(f"Similarity Score: {result.score:.4f}")
                print(f"Preview: {text}\n")

                total_score += result.score

            avg_score = total_score / len(base_nodes)
            print(f"✅ Average Semantic Similarity (Base): {avg_score:.4f}")

    asyncio.run(test_nodes())



