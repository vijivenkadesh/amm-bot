from langchain_community.document_loaders import PyMuPDFLoader
import logging
from pathlib import Path
from langchain_core.documents import Document
from typing import List

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")



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
        logging.debug(f"Found {len(file_paths)} files in the directory: {self.filepath}")

        if not file_paths:
            raise FileNotFoundError(f"No files found in the directory: {self.filepath}")
        
        documments = []

        for path in file_paths:
            loader = PyMuPDFLoader(file_path=path)
            docs = loader.load()
            documments.append(docs)

        logging.debug(f"Loaded {len(documments)} documents from the directory: {self.filepath}")
        return documments


