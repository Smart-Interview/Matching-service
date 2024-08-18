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


keywords = {
    'sexe': [],
    'formation': [],
    'localisation': [],
    'compétences': [],
    'expériences': []
}

if resumee is not None and job_description is not None:
    with st.expander(""):
        with st.spinner("En cours..."):
            col_a, col_b = st.columns(2)
            resumee_responses=[get_text_info_from_pdf(cv, "Veuillez extraire juste les mots clés du sexe, la formation, le lieu, les compétences et expériences professionnelles mentionnées dans chaque CV sans écrire le titre (informations...) et sans mentionner d'informations supplémentaires") for cv in resumee]
            resumee_sexes=[get_text_info_from_pdf(cv, "Veuillez extraire juste le sexe (Femme ou Homme ) à partir de chaque CV sans mentionner d'informations supplémentaires, tu peux conclure le sexe à partir du prénom si c'est possible ") for cv in resumee]
            resumee_formations=[get_text_info_from_pdf(cv, "Veuillez extraire juste les mots clés des formations à partir de chaque CV sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)") for cv in resumee]
            resumee_localisations=[get_text_info_from_pdf(cv, "Veuillez extraire juste la localisation actuelle (plus récente) à partir de chaque CV sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)") for cv in resumee]
            resumee_competences=[get_text_info_from_pdf(cv, "Veuillez extraire juste les mots clés des compétences à partir de chaque CV sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)") for cv in resumee]
            resumee_experiences=[get_text_info_from_pdf(cv, "Veuillez extraire juste les mots clés des expériences à partir de chaque CV sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)") for cv in resumee]
            job_description_response=get_text_info_from_pdf(job_description,"Veuillez extraire juste les mots clés du sexe, la formation, le lieu, les compétences et expériences professionnelles demandées dans cet offre d'emploie sans écrire le titre (informations...) sans mentionner d'informations supplémentaires")
            job_description_sexes=get_text_info_from_pdf(job_description, "Veuillez extraire juste le sexe (Femme ou Homme ) à partir de cet offre d'emploie sans mentionner d'informations supplémentaires")
            job_description_formations=get_text_info_from_pdf(job_description, "Veuillez extraire juste les mots clés des formations demandées par cet offre d'emploie sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)") 
            job_description_localisations=get_text_info_from_pdf(job_description, "Veuillez extraire juste la localisation demandée si nécessaire à partir de cet offre d'emploie sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)") 
            job_description_competences=get_text_info_from_pdf(job_description, "Veuillez extraire juste les mots clés des compétences demandées par cet offre d'emploie sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)") 
            job_description_experiences=get_text_info_from_pdf(job_description, "Veuillez extraire juste les mots clés des expériences demandées par cet offre d'emploie sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)") 

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
            
    import re
    with st.expander("Matching"):
        with st.spinner("En cours..."):
            for resumee_sexe in resumee_sexes:
                keywords["sexe"].append(resumee_sexe)
            for resumee_formation in resumee_formations:
                keywords["formation"].append(resumee_formation)
            for resumee_localisation in resumee_localisations:
                keywords["localisation"].append(resumee_localisation)
            for resumee_competence in resumee_competences:
                keywords["compétences"].append(resumee_competence)
            for resumee_experience in resumee_experiences:
                keywords["expériences"].append(resumee_experience)
            
            st.write(keywords)

            







