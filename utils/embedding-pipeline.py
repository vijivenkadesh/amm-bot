import sys
import os
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from documentloader import PdfLoader
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from dotenv import load_dotenv
from typing import List, Any
from langchain_core.documents import Document

load_dotenv()

# Setting up logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setFormatter(fmt=formatter)
logger.addHandler(hdlr=console_handler)




class EmbeddingManager:
    OPENAI_API_KEY = SecretStr(f"{os.getenv("OPENAI_API_KEY")}")
    # Load the PDF documents and split them into chunks
    @classmethod
    def get_chunks(cls, filepath: str):
        try:
            loader = PdfLoader(filepath=filepath)
        except FileNotFoundError as e:
            logger.error(msg=f"{e}")
        documents  = loader.load_pdf()

        if not documents:
            logger.error(msg="No documents found")
            raise IndexError("No documents found")
        
        splitter = RecursiveCharacterTextSplitter()
        chunks = splitter.split_documents(documents=documents)
        logger.info(msg=f"{len(chunks)} chunks are created")
        return chunks

    @classmethod
    def get_embeddings(cls) -> OpenAIEmbeddings:
        embeddings  = OpenAIEmbeddings(api_key=cls.OPENAI_API_KEY)
        # response = embeddings.embed_documents(texts=chunks)
        return embeddings

    # print(response)

# TODO: add metadata to the chunks
if __name__ == "__main__":
    try:
        test_chunks = EmbeddingManager.get_chunks(filepath="doc") 
        print(test_chunks[0])
    except Exception as e:
        logger.error(e)