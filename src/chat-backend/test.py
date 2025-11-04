from typing import TypedDict
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph
from langchain_chroma import Chroma

class GraphState(TypedDict):
    question: str
    answer: str



loader = PyPDFLoader("praktikumsbericht.pdf")  
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)


embedding = OllamaEmbeddings(model="llama3.2:1b")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="./chroma_store"
)
retriever = vectorstore.as_retriever()

# === Load the LLM ===
llm = OllamaLLM(model="llama3.2")


def ask_node(state):
    question = state["question"]
    # Retrieve relevant chunks
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Construct simple prompt
    prompt = f"""
        You are an expert assistant. Use the following context to answer the question:

        Context:
        {context}

        Question: {question}

        Answer:
        """.strip()
   
    answer = llm.invoke(prompt)
    return {"question": question, "answer": answer}


graph = StateGraph(GraphState)
graph.add_node("ask", ask_node)
graph.set_entry_point("ask")
graph.set_finish_point("ask")
compiled_graph = graph.compile()


while True:
    query = input("Ask a question about the PDF (or type 'exit'): ")
    if query.strip().lower() == "exit":
        break

    result = compiled_graph.invoke({"question": query})
    print("\nAnswer:", result["answer"], "\n")



