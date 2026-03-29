from langchain_openai import ChatOpenAI
# from utils.rag_retriever import docs_with_Score
from core.config import settings
from schema.schemas import OutputSchema
# from utils.documentloader import PdfLoader
from utils.rag_retriever import RAGRetriever
from dotenv import load_dotenv
from utils.pc_vectordb import PCVectorDB
from utils.embedding_pipeline import EmbeddingManager
from typing import List

load_dotenv()


# docs = PdfLoader().load_pdf()

def main(query: List[float]):
    # context = RAGRetriever.retrieve_relevant_docs(query=query, k=5)
    index = PCVectorDB.define_index(index_name="amm-bot")
    context = index.query(vector=query, top_k=5, include_metadata=True, namespace="32-XX-XX-Landing-Gear-Docs")
    llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY, model="gpt-4o-2024-08-06")
    response = llm.invoke(input=f"You are a Aircraft Maintenance Technical Expert, please respond like an AMM manual based on the following context: {context}")
    return response.content


if __name__ == "__main__":
    import json
    query = "How much degreees the steer wheel handles in any direction."
    embedding = EmbeddingManager.get_embeddings()
    query_embedding = embedding.embed_query(text=query)

    result = main(query_embedding)

    print(result)

