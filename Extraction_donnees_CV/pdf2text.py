import streamlit as st
import pdfplumber

resumee=st.file_uploader("Veuillez télécharger votre CV",type=["pdf","docx"])

def pdf_to_text(pdf_file):
    text = ""
    pdf_file.seek(0)  # Réinitialiser le pointeur de fichier
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

if resumee is not None:
    response=pdf_to_text(resumee)
    st.write(response)








