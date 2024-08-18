import streamlit as st
import google.generativeai as genai
from PIL import Image
# import cv2
import pdfplumber

genai.configure(api_key="AIzaSyBTUgq4I1dwK6bMg3PXSFC2wdT8jy1khBs")

vision_model = genai.GenerativeModel('gemini-1.5-flash')

resumee=st.file_uploader("Veuillez télécharger le CV",type=["pdf","docx"])

def pdf_to_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def get_text_info_from_pdf(pdf_file, request):
    text = pdf_to_text(pdf_file)
    response = vision_model.generate_content([request, text])
    return response.text

if resumee is not None:
    st.write(get_text_info_from_pdf(resumee,"Veuillez extraire les informations de ce CV"))
            









