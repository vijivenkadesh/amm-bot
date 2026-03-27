from langchain_community.vectorstores import FAISS
from utils.embedding_pipeline import EmbeddingManager
from utils.documentloader import PdfLoader
from pathlib import Path
import logging
import sys
from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

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
            logger.info(msg="Vector store created successfully")
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
            raise

        vectorstore.save_local(folder_path=str(folder_path))
        logger.info(msg=f"Vector store saved successfully at {folder_path}")




    class PineConeManager:

        @classmethod
        def get_pinecone_vectorstore(cls, api_key: str, environment: str) -> Pinecone:
            try:
                pinecone_vectorstore = Pinecone(api_key=api_key, environment=environment)
                logger.info(msg="PineCone vector store created successfully")
            except Exception as e:
                logger.error(msg=f"Something went wrong while creating PineCone vector store. Please refer the error {e}")
                raise
            return pinecone_vectorstore
            

        @classmethod
        def embedd_pinecone_vectorstore(cls, chunks: List[Document], embeddings: OpenAIEmbeddings) -> Pinecone:
            if not chunks:
                raise ValueError("Chunks can't be empty")
            try:
                vectorstore = Pinecone.from_documents(chunks, embeddings)
                logger.info(msg="PineCone vector store created successfully")
            except Exception as e:
                logger.error(msg=f"Something went wrong while creating PineCone vector store. Please refer the error {e}")
                raise
            return vectorstore


if __name__ == "__main__":
    embeddings = EmbeddingManager.get_embeddings()
    chunks = EmbeddingManager.get_chunks(filepath='doc')
    # vectorstore = VectorStoreManager.get_vector_Store(chunks=chunks, embeddings=embeddings)
    # VectorStoreManager.save_vectorstore(folderpath="database", vectorstore=vectorstore)
