from langchain_openai import ChatOpenAI
# from utils.rag_retriever import docs_with_Score
from core.config import settings
from schema.schemas import OutputSchema
# from utils.documentloader import PdfLoader
from utils.rag_retriever import RAGRetriever
from dotenv import load_dotenv

load_dotenv()


# docs = PdfLoader().load_pdf()

def main(query: str):
    context = RAGRetriever.retrieve_relevant_docs(query=query, k=5)
    llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY, model="gpt-4o-2024-08-06")
    response = llm.invoke(input=f"You are a Aircraft Maintenance Technical Expert, please respond like an AMM manual based on the following context: {context}")
    return response.content


if __name__ == "__main__":
    import json
    query = "Summarize the operation of landing gears and doors."
    result = main(query)

    print(result)

