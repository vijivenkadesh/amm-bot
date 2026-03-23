from langchain_openai import ChatOpenAI
# from utils.rag_retriever import docs_with_Score
from core.config import settings
from schema.schemas import OutputSchema
from utils.documentloader import PdfLoader



docs = PdfLoader().load_pdf()

def main(context):
    llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY, model="gpt-4o-2024-08-06")
    llm_with_structure = llm.with_structured_output(schema=OutputSchema)
    response = llm_with_structure.invoke(input=f"Get the candidate data from the provided {context}")
    return response


if __name__ == "__main__":
    import json
    result = main(docs)
    json_result = json.dumps(dict(result), indent=4)
    print(json_result)

