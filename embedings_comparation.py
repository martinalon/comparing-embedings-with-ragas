from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA

import json

# Ruta del archivo con las preguntas
file_path = "./docs/questions.json"

# Abrir y cargar el JSON
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)



preguntas = [item["question"] for item in data]
respuestas = [item["ground_truth"] for item in data]


# hasta aqui ya se leyo las preguntas y respuestas del .jason falta corroborar como se tiene que ingresar en ragas 



"""
# defining the retrivals
embeddings__mxbai = OllamaEmbeddings(model="mxbai-embed-large")  # Puedes usar mistral, phi3 o llama2 también
db_mxbai = Chroma(persist_directory="./bases/chroma_mxbai", embedding_function= embeddings__mxbai)
retriever_mxbai = db_mxbai.as_retriever()


embeddings__nomic = OllamaEmbeddings(model="nomic-embed-text")  # Puedes usar mistral, phi3 o llama2 también
db_nomic = Chroma(persist_directory="./bases/chroma_nomic", embedding_function= embeddings__mxbai)
retriever_nomic = db_mxbai.as_retriever()

# 4. Crear el modelo LLM
llm = ChatOllama(model="mistral")  # modelo local



qa_mxbai = RetrievalQA.from_chain_type(llm=llm, retriever= retriever_mxbai)
qa_nomic = RetrievalQA.from_chain_type(llm=llm, retriever= retriever_nomic )

results_a = []
results_b = []

for q in questions:
    res_a = qa_a.invoke(q)
    res_b = qa_b.invoke(q)
    
    results_a.append({
        "question": q,
        "answer": res_a["result"],
        "contexts": [doc.page_content for doc in retriever_a.get_relevant_documents(q)],
    })

    results_b.append({
        "question": q,
        "answer": res_b["result"],
        "contexts": [doc.page_content for doc in retriever_b.get_relevant_documents(q)],
    })
"""
