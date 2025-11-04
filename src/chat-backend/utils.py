#!pip install python-dotenv


import numpy as np
import nest_asyncio

#nest_asyncio.apply()


# def get_openai_api_key():
#     _ = load_dotenv(find_dotenv())

#     return os.getenv("OPENAI_API_KEY")


# def get_hf_api_key():
#     _ = load_dotenv(find_dotenv())

#     return os.getenv("HUGGINGFACE_API_KEY")

from llama_index.core import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core import load_index_from_storage
import os
from llama_index.core import Settings
from llama_index.core.node_parser import HierarchicalNodeParser

from llama_index.core.node_parser import get_leaf_nodes
from llama_index import StorageContext
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine

from trulens.core import Feedback 
from trulens.apps.llamaindex import TruLlama
from trulens.providers.openai import OpenAI
from trulens_eval.feedback import GroundTruthAgreement
from trulens_eval import TruBasicApp, Feedback, Tru, Select
from trulens.core.feedback import Gr 

mini_emb = "sentence-transformers/all-MiniLM-L6-v2"
bge_emb = "local:BAAI/bge-small-en-v1.5"

provider = OpenAI()



qa_relevance = (
    #Feedback(openai.relevance_with_cot_reasons, name="Answer Relevance")
    Feedback(provider.context_relevance_with_cot_reasons, name="Answer Relevance")
    .on_input_output()
)

qs_relevance = (
    #Feedback(openai.relevance_with_cot_reasons, name = "Context Relevance")
    Feedback(provider.context_relevance_with_cot_reasons, name = "Context Relevance") 
    .on_input()
    .on(TruLlama.select_source_nodes().node.text)
    .aggregate(np.mean)
)

#grounded = Groundedness(groundedness_provider=openai, summarize_provider=openai)
# grounded = Groundedness(groundedness_provider=openai)

# groundedness = (
#     Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
#         .on(TruLlama.select_source_nodes().node.text)
#         .on_output()
#         .aggregate(grounded.grounded_statements_aggregator)
# )

groundedness = (
    Feedback(provider.groundedness_with_cot_reasons, name="Groundedness")
    .on(TruLlama.select_source_nodes().node.text)
    .on_output()
    .aggregate(np.mean)
)





feedbacks = [qa_relevance, qs_relevance, groundedness]
def get_prebuilt_trulens_recorder(query_engine, app_id):
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
        )
    return tru_recorder


############################

def build_sentence_window_index(
    document, llm, embed_model=mini_emb, save_dir="sentence_index"
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    # base node parser is a sentence splitter
    text_splitter = SentenceSplitter()
    # sentence_context = ServiceContext.from_defaults(
    #     llm=llm,
    #     embed_model=embed_model,
    #     node_parser=node_parser,
    # )

    Settings.llm = llm
    Settings.embed_model = embed_model

    nodes = node_parser.get_nodes_from_documents(documents=document)
    base_nodes = text_splitter.get_nodes_from_documents(document)
    #Settings.text_splitter = text_splitter


    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            [nodes],
            #transformations=node_parser
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
        base_index = VectorStoreIndex([base_nodes])
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            #service_context=sentence_context,
        )

    return sentence_index, base_index


def get_sentence_window_query_engine(
    sentence_index,
    similarity_top_k=6,
    rerank_top_n=2,
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine





def build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index",
    chunk_sizes=None,
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes, storage_context=storage_context, service_context=merging_context
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=merging_context,
        )
    return automerging_index


def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=12,
    rerank_top_n=6,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine
