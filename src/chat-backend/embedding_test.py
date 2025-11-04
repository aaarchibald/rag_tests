from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, Document
from sentence_transformers import util 
import numpy as np
from llama_index.core.node_parser import SentenceWindowNodeParser, SemanticSplitterNodeParser

#EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # better for very short chunks
#EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # the worst results
#EMBED_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1" # 
#EMBED_MODEL = "sentence-transformers/stsb-mpnet-base-v2"

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2" # the best one 



embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL,
    device="cpu",
    embed_batch_size=8 
)

llm = Ollama(
    model="llama3.2:latest", 
    base_url= "http://localhost:11434",
    request_timeout=120.0
)


s1 = "Wie kann ich mein Passwort zurücksetzen?"
s2 = "Um Ihr Passwort zurückzusetzen, klicken Sie auf „Passwort vergessen?“ auf der Login-Seite."


s3 = "Wie buche ic h eine Reise mit der Bahn"
s4 = """Was mache ich, wenn ich eine Reise buchen möchte?
Schreibe eine Mail mit deiner Reiseanfrage an office@ilume.de
Hier ein paar wichtige Infos, die wir für deine Reisebuchung benötigen:
Reisen werden von der ilume nur aus folgenden Gründen übernommen:
Projektbedingte Reisen
Reisen zum Vorstellungsgespräch
Reisen zu Schulungen / Fortbildungen"""

s5 = """
Buchung eines Bahntickets
Wird ein Bahnticket benötigt, muss dieses bitte im ilume-Office angefragt werden. Bei
Fahrten mit dem Fernverkehr wird das Ticket vom ilume-Office gebucht. Fahrten im
Nahverkehr können bei zu kurzer Entfernung (Mainz – Frankfurt) nicht vom ilume-Office
gebucht werden und müssen vom Mitarbeitenden direkt am Ticketautomaten gebucht
werden. Die Fahrkarte kann im Anschluss im ilume-Office eingereicht werden. Gebucht
werden vom ilume-Office immer Flextickets, sodass eine frühere/spätere An- und Abreise
möglich ist.
"""
s6 = """
Buchung eines Bahntickets
Wird ein Bahnticket benötigt, muss dieses bitte im ilume-Office angefragt werden. Bei
Fahrten mit dem Fernverkehr wird das Ticket vom ilume-Office gebucht. Fahrten im
Nahverkehr können bei zu kurzer Entfernung (Mainz – Frankfurt) nicht vom ilume-Office
gebucht werden und müssen vom Mitarbeitenden direkt am Ticketautomaten gebucht
werden. 
"""

s7 = """
Buchung eines Mietwagens 
Ist die Bahn kein geeignetes Verkehrsmittel für diese Reise, da sich der Zielort an einem 
abgelegenen Ort befindet, so kann nach Genehmigung ein Mietwagen gebucht werden. 
Dieser wird über das ilume-Office gebucht. 
Folgende Daten benötigen wir zum Buchen: 
"""
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=200, chunk_overlap=30)


# semantic_splitter = SemanticSplitterNodeParser(
#         buffer_size=1, 
#         breakpoint_percentile_threshold=95, 
#         embed_model=embed_model,
#         show_progress=True,
#         include_metadata=True,
#     )

docs = [Document(text=s5)]
nodes = splitter.get_nodes_from_documents(docs)
#nodes = semantic_splitter.get_nodes_from_documents(docs)

def cosine_similarity(v1, v2):
   
    v1, v2 = np.array(v1), np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def compare_embeddings(s1, s2):

    emb1 = embed_model.get_text_embedding(s1)
    
    emb2 = embed_model.get_text_embedding(s2)

    similarity = cosine_similarity(emb1, emb2)
    print(f"\nCosine similarity:{similarity:.4f}")

def compare_embeddings2(s1, s2):

    emb1 = embed_model.get_text_embedding(s1)
    for i, n in enumerate(nodes[:3]):
        chunk = n.text
        print(f"Chunk {i+1}: {n.text}\n")
        emb2 = embed_model.get_text_embedding(chunk)

        similarity = cosine_similarity(emb1, emb2)
        print(f"\nCosine similarity:{similarity:.4f}")      

user_query = "Was mache ich wenn ich krank bin?"


# topics_prompt = f"""
# Du bist illume Assistant und muss verstehen, was der Mitarbeiter genau fragt, um die ANtworten in Unterlagen zu finden. Wenn der Kontext nicht ganz klar ist. Frage den noch nach mehrere Details.

# {user_query}
# """
topics_prompt = f"""
Du bist der illume Assistant, der Mitarbeitenden hilft, präzise Antworten in internen Unterlagen zu finden. Analysiere die folgende Frage sorgfältig:

\"\"\"{user_query}\"\"\"

1. Extrahiere die wichtigsten Themen, Schlüsselwörter und möglichen Kontext, die für die Suche relevant sind.
2. Wenn die Frage unklar oder zu allgemein ist, formuliere mindestens eine konkrete Rückfrage, um weitere Details zu erhalten.
3. Gib zuerst eine kurze Liste der relevanten Themen/Keywords aus.
4. Wenn Rückfragen nötig sind, schreibe diese deutlich als separate Frage(n).

Antworte strukturiert im folgenden Format:

Themen: [Thema1, Thema2, ...]  
Rückfrage: [Frage zur Konkretisierung] (falls nötig, sonst "Keine Rückfrage")

---
"""



new_prompt = f"""
Du bist Analyste.  Zu welchem Thema gehört diese Frage:

\"\"\"{user_query}\"\"\"

Optionen: [Reise, Krankheit, IT, HR, Vertrag, Urlaub, etc.]
Return nur ein Wort.
"""

#topics_text = llm.complete(topics_prompt)

#topic = llm.complete(new_prompt)

#print(topic)

compare_embeddings(s1, s2)
compare_embeddings(s3, s5)
compare_embeddings(s3, s4)
compare_embeddings(s3, s6)
compare_embeddings(s3, s7)