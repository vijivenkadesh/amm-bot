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

folderpath = "database"
database_path = str(Path(f"{folderpath}").resolve())
vectorstore = FAISS.load_local(
folder_path=folderpath,
embeddings=embeddings,
allow_dangerous_deserialization=True  # required for newer LangChain
)

query = "What are the causes of hydraulic system failure?"

# docs = vectorstore.similarity_search(query, k=3)

# query_embedding = embeddings.embed_query(text=query)

docs_with_score = vectorstore.similarity_search(query=query, k=4)

# print(docs_with_score)

cross_encoder = CrossEncoder(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2")

pairs = [(query, str(doc))for doc in docs_with_score]

scores = cross_encoder.predict(sentences=pairs)

print(scores)


scored_docs = list(zip(docs_with_score, scores))
scored_docs.sort(key=lambda x: x[1], reverse=True)

print(scored_docs)