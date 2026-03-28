from pinecone import Pinecone, Vector
from dotenv import load_dotenv
from utils.embedding_pipeline import EmbeddingManager
import os

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=str(api_key))
index_name = "amm-bot"

index = pc.Index(name=index_name)
print(index)

embeddings = EmbeddingManager.get_embeddings()
chunks = EmbeddingManager.get_chunks(filepath='doc')
print(f"Number of chunks: {len(chunks)}")

text_list = [chunk.page_content for chunk in chunks]
metadata_list = [chunk.metadata for chunk in chunks]
sample = text_list[2]
metadata = metadata_list[2]

print(sample)

sample_vector = embeddings.embed_query(text=sample)


# vector = embeddings.embed_query(text="I am testing pinecone")
vectors = [Vector(id="2", values=sample_vector, metadata=metadata)]
print(vectors)

index.upsert(vectors=vectors, namespace="example-namespace")


index.search(namespace="example-namespace", query={} )

index.query()