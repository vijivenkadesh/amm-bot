from langchain_openai import ChatOpenAI
from utils.rag_retriever import docs_with_Score
from core.config import settings


def main(context):
    llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY)
    response = llm.invoke(input="Summarise this context from the retiver {context}")
    return response.content


if __name__ == "__main__":
    result = main(docs_with_Score)
    print(result)
