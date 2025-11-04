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
from llama_index.core.node_parser import SentenceWindowNodeParser
from trulens_eval import Select

pdf_dir = "./pdfs"
chroma_path = "./chroma_db1"
collection_name = "faqs_new"

session = TruSession()
#session.reset_database()






def chroma_db_exists(path):
    return os.path.exists(path) and len(os.listdir(path)) > 0

if not chroma_db_exists(chroma_path):
    print("📦 No ChromaDB found — running document pipeline...")

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

    text_splitter = SentenceSplitter()

    #Settings.chunk_size = 128
    #Settings.chunk_overlap = 16
    Settings.text_splitter = text_splitter
    Settings.llm = Ollama(model="llama3.2:latest", request_timeout=60.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
   # embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    #nodes = pipeline.run(documents=[document], in_place=True, show_progress=True)
    nodes = node_parser.get_nodes_from_documents(documents)
    base_nodes = text_splitter.get_nodes_from_documents(documents)


    db = chromadb.PersistentClient(path=chroma_path)
    collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    sentence_index = VectorStoreIndex.from_documents(
        nodes,
        storage_context=storage_context,
        #embed_model=embed_model,
    )

    base_index = VectorStoreIndex.from_documents(
        base_nodes,
        storage_context=storage_context,
        #embed_model=embed_model,
    )


    #sentence_index = VectorStoreIndex(nodes)

    #base_index = VectorStoreIndex(base_nodes)


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
        #embed_model=embed_model,
    )

#query_engine = index.as_query_engine(similarity_top_k=3)


# provider = LiteLLM(
#     model_engine="ollama/llama3.2:latest",  
#     completion_kwargs={
#         "api_base": "http://localhost:4000",  # Gateway address
#         "api_key": "sk-h2k4hd847hi3",          # Optional if key required
#     }
# )

provider = LiteLLM(
    model_engine="openrouter/anthropic/claude-3-haiku",  # or just "anthropic/claude-3-haiku"
    completion_kwargs={
        "api_base": "https://openrouter.ai/api/v1",       # OpenRouter's base URL
        "api_key": "sk-h2k4hd847hi3",                 # Replace with your actual key
    }
)

# Helpfulness
f_helpfulness = Feedback(provider.helpfulness).on_output()

# Question/answer relevance between overall question and answer.
f_qa_relevance = Feedback(provider.relevance_with_cot_reasons).on_input_output()

# Question/statement relevance between question and each context chunk with context reasoning.
# The context is located in a different place for the sub questions so we need to define that feedback separately
# f_context_relevance_subquestions = (
#     Feedback(provider.context_relevance_with_cot_reasons)
#     .on_input()
#     .on(Select.Record.calls[0].rets.source_nodes[:].node.text)
#     .aggregate(np.mean)
# )

# f_context_relevance = (
#     Feedback(provider.context_relevance_with_cot_reasons)
#     .on_input()
#     #.on(TruLlama.select_context())
#     .on(Select.Record.calls[0].rets.source_nodes[:].node.text)
#     .aggregate(np.mean)
# )

# Initialize groundedness
# Groundedness with chain of thought reasoning
# Similar to context relevance, we'll follow a strategy of defining it twice for the subquestions and overall question.
f_groundedness_subquestions = (
    Feedback(provider.groundedness_measure_with_cot_reasons)
    .on_context(collect_list=True)
    .on_output()
)

# f_groundedness = (
#     Feedback(provider.groundedness_measure_with_cot_reasons)
#     .on(TruLlama.select_context())
#     .on_output()
# )

from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from trulens.apps.llamaindex import TruLlama

sentence_query_engine = sentence_index.as_query_engine(
    similarity_top_k=2,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)

tru_sentence_query_engine_recorder = TruLlama(
    sentence_query_engine,
    app_name="climate query engine",
    app_version="sentence_window_index",
    feedbacks=[
        f_qa_relevance,
        #f_context_relevance,
        #f_groundedness,
        f_helpfulness,
    ],
)
with tru_sentence_query_engine_recorder:
    sentence_query_engine.query("Wie viele Mitarbieter arbeiten bei ilume?")




query_engine = base_index.as_query_engine(similarity_top_k=2)

tru_query_engine_recorder = TruLlama(
    query_engine,
    app_name="climate query engine",
    app_version="vector_store_index",
    feedbacks=[
        f_qa_relevance,
        #f_context_relevance,
        #f_groundedness,
        f_helpfulness,
    ],
)
with tru_query_engine_recorder:
    query_engine.query("Wie viele Mitarbieter arbeiten bei ilume?")




from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import ToolMetadata

# subquestion_query_engine = SubQuestionQueryEngine.from_defaults(
#     [
#         QueryEngineTool(
#             query_engine=sentence_query_engine,
#             metadata=ToolMetadata(
#                 name="climate_report", description="Climate Report on Oceans."
#             ),
#         )
#     ],
#     verbose=False,
# )

# tru_subquestion_query_engine_recorder = TruLlama(
#     subquestion_query_engine,
#     app_name="climate query engine",
#     app_version="sub_question_query_engine",
#     feedbacks=[
#         f_qa_relevance,
#         #f_context_relevance,
#         #f_context_relevance_subquestions,
#         #f_groundedness,
#         f_groundedness_subquestions,
#         f_helpfulness,
#     ],
# )
# with tru_subquestion_query_engine_recorder:
#     subquestion_query_engine.query("Wie viele Mitarbieter arbeiten bei ilume?")


run_dashboard(session)