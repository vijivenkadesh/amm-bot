from langchain_community.vectorstores import FAISS
from embedding_pipeline import EmbeddingManager

embeddings = EmbeddingManager.get_embeddings()

vectorstore = FAISS(embedding_function=embeddings)