import streamlit as st
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import tempfile

def pdf_to_images(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(pdf_file.read())
        images = convert_from_path(tmp_file.name)
    return images

def ocr_core(pdf_file):
    images = pdf_to_images(pdf_file)
    text = ""
    
    for image in images:
        text += pytesseract.image_to_string(image)
    
    return text

resumee = st.file_uploader("Veuillez télécharger votre CV", type=["pdf"])

if resumee is not None:
    st.write(ocr_core(resumee))
