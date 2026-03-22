import logging
import sys



# Setting up logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setFormatter(fmt=formatter)
logger.addHandler(hdlr=console_handler)

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