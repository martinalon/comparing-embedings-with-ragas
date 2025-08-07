from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA

"""embedings
nomic-embed-text
mxbai-embed-large
"""
# defining the document path
text_path = f"./docs/quantum_intro.txt"

# loading the document
loader = TextLoader(text_path)
documents = loader.load()


#spliting the documents
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)  # this takes 500 characters and averlap 50 characters between each element of the list
docs_split = splitter.split_documents(documents)


embedding_models = {
    "nomic": OllamaEmbeddings(model="nomic-embed-text"),
    "mxbai": OllamaEmbeddings(model="mxbai-embed-large")
}


#creating the bases for each embeding model
for name, embedding in embedding_models.items():
    db = Chroma.from_documents(docs_split, embedding=embedding, persist_directory=f"./bases/chroma_{name}")
    
