import streamlit as st
import google.generativeai as genai
from PIL import Image
# import cv2
import pdfplumber

genai.configure(api_key="AIzaSyBTUgq4I1dwK6bMg3PXSFC2wdT8jy1khBs")

vision_model = genai.GenerativeModel('gemini-1.5-flash')

resumee=st.file_uploader("Veuillez télécharger les CVs",type=["pdf","docx"],accept_multiple_files=True)
job_description=st.file_uploader("Veuillez télécharger l'offre d'emploie",type=["pdf","docx"],accept_multiple_files=False)

def pdf_to_text(pdf_file):
    text = ""
    pdf_file.seek(0)  # Réinitialiser le pointeur de fichier
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def get_text_info_from_pdf(pdf_file, request):
    text = pdf_to_text(pdf_file)
    response = vision_model.generate_content([request, text])
    return response.text

if resumee is not None and job_description is not None:
    col_a, col_b = st.columns(2)
    resumee_responses=[get_text_info_from_pdf(cv, "Veuillez extraire les informations de ces CVs en les séparant") for cv in resumee]
    job_description_response=get_text_info_from_pdf(job_description,"Veuillez extraire les informations de cet offre d'emploie")

    with col_a:
        st.write("**Informations des CVs**")
        if len(resumee_responses)>1:
            tabs = st.tabs([f"CV {i+1}" for i in range(len(resumee_responses))])
            for i, tab in enumerate(tabs):
                with tab:
                    st.write(resumee_responses[i])
        else:
            st.write(resumee_responses[0])
            
    with col_b:
        st.write("**Informations de l'offre d'emploi**")
        st.write(job_description_response)








