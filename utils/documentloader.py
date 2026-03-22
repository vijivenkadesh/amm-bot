from langchain_community.document_loaders import PyMuPDFLoader
import logging
from pathlib import Path
from langchain_core.documents import Document
from typing import List, Iterable
import sys


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setFormatter(fmt=formatter)
logger.addHandler(hdlr=console_handler)

# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")



class PdfLoader:

    def __init__ (self, filepath: str = "doc"):
        self.filepath = Path(filepath).resolve()
    
    # Load the files from the provided directory

    def load_pdf(self) -> List[Document]:
        """
        Load PDF documents from the specified directory.
        Returns:
            List[Document]: A list of loaded documents."""
        
        file_paths = []

        for p in self.filepath.iterdir():
            file_paths.append(p)
        logger.info(f"Found {len(file_paths)} files in the directory: {self.filepath}")

        if not file_paths:
            raise FileNotFoundError(f"No files found in the directory: {self.filepath}")
        
        documments = []

        for path in file_paths:
            loader = PyMuPDFLoader(file_path=path)
            docs = loader.load()
            documments.extend(docs)

        logger.info(f"Loaded {len(documments)} documents from the directory: {self.filepath}")
        return documments