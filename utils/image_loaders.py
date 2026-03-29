import logging
import sys
from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
import fitz



# Setting up logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setFormatter(fmt=formatter)
logger.addHandler(hdlr=console_handler)

# This method did not work as expected, it only extracted the text and not the images. I will try to use PyMuPDF directly to extract the images and text separately.    

file_path = Path("doc") / "cmm.pdf"
loader = PyMuPDFLoader(file_path=file_path, extract_images=True)
documents = loader.load()



doc = fitz.open(file_path)
image_list = []
for page_num in range(doc.page_count):

    page = doc[page_num]

    image_info = page.get_images(full=True)

    image_list.append(image_info)
    # print(f"Page {page_num}: {len(image_info)} images")


print(image_list)


class ImageManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.doc = fitz.open(file_path)

    def extract_images(self):
        image_list = []
        for page_num in range(self.doc.page_count):
            page = self.doc[page_num]
            image_info = page.get_images(full=True)
            image_list.append(image_info)
        return image_list

