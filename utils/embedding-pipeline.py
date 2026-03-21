import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from documentloader import PdfLoader
import sys

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setFormatter(fmt=formatter)
logger.addHandler(hdlr=console_handler)

loader = PdfLoader()
documents  = loader.load_pdf()

# print(documents)

splitter = RecursiveCharacterTextSplitter()

chunks = splitter.split_documents(documents=documents)

logger.info(chunks[0].metadata)


# TODO: add metadata to the chunks