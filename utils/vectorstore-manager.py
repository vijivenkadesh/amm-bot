from langchain_community.vectorstores import FAISS
from embedding_pipeline import EmbeddingManager
from documentloader import PdfLoader
from pathlib import Path

chunks = EmbeddingManager.get_chunks(filepath="doc")

embeddings = EmbeddingManager.get_embeddings()

folder_path = str(Path("database").resolve())

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(folder_path=str(folder_path))




# vectorstore = FAISS.load_local(
#     folder_path=folder_path,
#     embeddings=embeddings,
#     allow_dangerous_deserialization=True  # required for newer LangChain
# )

# query = "What are the causes of hydraulic system failure?"

# # docs = vectorstore.similarity_search(query, k=3)

# query_embedding = embeddings.embed_query(text=query)

# docs_with_Score = vectorstore.similarity_search_with_relevance_scores


# print(docs_with_Score)