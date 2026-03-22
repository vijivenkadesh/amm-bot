from langchain_community.vectorstores import FAISS
from embedding_pipeline import EmbeddingManager
from documentloader import PdfLoader
from pathlib import Path
import logging
import sys
from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


# Setting up logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setFormatter(fmt=formatter)
logger.addHandler(hdlr=console_handler)




class VectorStoreManager:

    @classmethod
    def get_vector_Store(cls, chunks: List[Document], embeddings: OpenAIEmbeddings) -> FAISS:
        if not chunks:
            raise ValueError("Chunks can't be empty")
        try:
            vectorstore = FAISS.from_documents(chunks, embeddings)
        except Exception as e:
            logger.error(msg=f"Something went wrong while creating vector store. Please refer the error {e}")
            raise
        return vectorstore
    

    @classmethod
    def save_vectorstore(cls, folderpath: str, vectorstore: FAISS) -> None:
        try:
            folder_path = str(Path(f"{folderpath}").resolve())
        except NotADirectoryError as e:
            logger.error(msg=f"Not a valid directorty provided. Please refer the error {e}")

        vectorstore.save_local(folder_path=str(folder_path))

