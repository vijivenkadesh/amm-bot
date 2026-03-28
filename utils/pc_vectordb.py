from pinecone import Pinecone, Vector
from dotenv import load_dotenv
from utils.embedding_pipeline import EmbeddingManager
import os
from langchain_core.documents import Document
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings
import uuid

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")


class PCVectorDB:

    @classmethod
    def get_pc_db(cls) -> Pinecone:
        pc = Pinecone(api_key=str(api_key))
        return pc
    

    @classmethod
    def define_index(cls, index_name: str):
        pc = cls.get_pc_db()
        index = pc.Index(name=index_name)
        return index
    

    @classmethod
    def get_vectors(cls, chunks: List[Document], embeddings: OpenAIEmbeddings) -> List[Vector]:
        embeddings = EmbeddingManager.get_embeddings()

        text_list = [chunk.page_content for chunk in chunks]
        metadata_list = [chunk.metadata for chunk in chunks]

        embedding_vectors = embeddings.embed_documents(texts=text_list)
        vectors: List[Vector] = []
        for vector, obj in zip(embedding_vectors, metadata_list):
            id = str(uuid.uuid4())
            vectors.append(Vector(id=id, values=vector, metadata=obj))

        return vectors
    

    @classmethod
    def upsert_vectors(cls, vectors: List[Vector], index_name: str, namespace: str) -> None:
        index = cls.define_index(index_name=index_name)
        index.upsert(vectors=vectors, namespace=namespace)
        print(f"Vectors upserted successfully in index: {index_name} and namespace: {namespace}")



if __name__ == "__main__":
    index_name = "amm-bot"
    namespace = "32-XX-XX-Landing-Gear-Docs"
    embeddings = EmbeddingManager.get_embeddings()
    chunks = EmbeddingManager.get_chunks(filepath='doc')
    vectors = PCVectorDB.get_vectors(chunks=chunks, embeddings=embeddings)
    PCVectorDB.upsert_vectors(vectors=vectors, index_name=index_name, namespace=namespace)