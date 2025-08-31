from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

CHROMA_DB_PATH = "/chromadb/data"

def create_embedding(material_id, text_content):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    vectordb.add_texts([text_content], metadatas=[{"material_id": material_id}])
    vectordb.persist()
