from langchain_community.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

db = Chroma(
    collection_name="test",
    embedding_function=embeddings
)

texts = ["Hello world", "This is a test"]
ids = ["1", "2"]
db.add_texts(texts=texts, ids=ids)

results = db.similarity_search("test")
print(results)
