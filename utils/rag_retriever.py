import logging
import sys
from pathlib import Path
from langchain_community.vectorstores import FAISS
from utils.embedding_pipeline import EmbeddingManager
from sentence_transformers import CrossEncoder


# Setting up logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setFormatter(fmt=formatter)
logger.addHandler(hdlr=console_handler)

embeddings = EmbeddingManager.get_embeddings()

class RAGRetriever:

    @classmethod
    def load_vector_db(cls):
        folderpath = "database"
        vectorstore = FAISS.load_local(
        folder_path=folderpath,
        embeddings=embeddings,
        allow_dangerous_deserialization=True  # required for newer LangChain
        )
        return vectorstore

    
    @classmethod
    def retrieve_relevant_docs(cls, query: str, k: int = 3):
        vectorstore = cls.load_vector_db()
        docs = vectorstore.similarity_search(query, k=k)
        return docs
    

    @classmethod
    def rerank_docs(cls, query: str, docs):
        cross_encoder = CrossEncoder(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, str(doc)) for doc in docs]
        scores = cross_encoder.predict(sentences=pairs)
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs



if __name__ == "__main__":
    query = "LANDING GEARS AND DOORS OPERATION"
    docs = RAGRetriever.retrieve_relevant_docs(query=query, k=5)
    scored_docs = RAGRetriever.rerank_docs(query=query, docs=docs)
    print("Top relevant documents:")
    for doc, score in scored_docs:  
        print(f"Score: {score:.4f}, Document: {doc.page_content[:200]}...")