import logging
import sys
import os
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



# doc = fitz.open(file_path)
# image_list = []
# for page_num in range(doc.page_count):

#     page = doc[page_num]

#     image_info = page.get_images(full=True)

#     image_list.append(image_info)
#     # print(f"Page {page_num}: {len(image_info)} images")


# print(image_list)


class ImageManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.doc = fitz.open(file_path)

    def extract_images_info(self):
        image_list = []
        for page_num in range(self.doc.page_count):
            page = self.doc[page_num]
            image_info = page.get_images(full=True)
            image_list.append(image_info)
        return image_list
    
    def save_images(self, output_dir):
        if not Path(output_dir).exists:
            Path(output_dir).mkdir(exist_ok=True)

        extracted_images = {}
        for page_num in range(self.doc.page_count):
            page = self.doc[page_num]
            image_info = page.get_images(full=True)

            for img in image_info:
                xref = img[0]
                if xref not in extracted_images:
                    base_image = self.doc.extract_image(xref=xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    with open(f"{output_dir}/image_{xref}.{image_ext}", "wb") as f:
                        f.write(image_bytes)
                    
                    extracted_images[xref] = f"{output_dir}/image_{xref}.{image_ext}"
        return extracted_images



if __name__ == "__main__":
    file_path = Path("doc") / "cmm.pdf"
    image_manager = ImageManager(file_path=file_path)
    image_info = image_manager.extract_images_info()
    extracted_images = image_manager.save_images(output_dir="images")
    print(extracted_images)