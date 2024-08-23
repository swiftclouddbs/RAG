import chromadb
from chromadb.utils import embedding_functions
from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

#command to save and load the database from your local machine

client = chromadb.PersistentClient(path="/Users/reginaldstuart/Documents/IT/ML:AI/myRAG/chroma_db_2")

collection_name="test_col_1",


embedding=HuggingFaceEmbeddings()


collection = client.get_collection(name=collection_name, embedding_function=embedding)

