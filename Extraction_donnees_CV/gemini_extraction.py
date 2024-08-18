import streamlit as st
import google.generativeai as genai
from PIL import Image
import pdfplumber

genai.configure(api_key=st.secrets["api_key"])

vision_model = genai.GenerativeModel('gemini-1.5-flash')

resumee = st.file_uploader("Veuillez télécharger les CVs", type=["pdf", "docx"], accept_multiple_files=True)
job_description = st.file_uploader("Veuillez télécharger l'offre d'emploi", type=["pdf", "docx"], accept_multiple_files=False)

def pdf_to_text(pdf_file):
    text = ""
    pdf_file.seek(0)
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def get_text_info_from_pdf(pdf_file, request):
    text = pdf_to_text(pdf_file)
    response = vision_model.generate_content([request, text])
    
    if response.candidates and response.candidates[0].content:
        return response.candidates[0].content
    else:
        return "Aucun résultat disponible pour cette requête."

def extract_parts(content):
    parts_text = []
    
    # Vérifier si 'content' a des 'parts' et est donc itérable
    if hasattr(content, 'parts'):
        for part in content.parts:
            parts_text.append(part.text)
    else:
        # Si 'content' est une liste d'objets
        for item in content:
            if hasattr(item, 'parts'):
                for part in item.parts:
                    parts_text.append(part.text)
    
    return parts_text


keywords_resumee = {'sexe': [], 'formation': [], 'localisation': [], 'compétences': [], 'expériences': []}
keywords_job_desc = {'sexe': [], 'formation': [], 'localisation': [], 'compétences': [], 'expériences': []}

if resumee and job_description:
    with st.expander(""):
        with st.spinner("En cours..."):
            col_a, col_b = st.columns(2)

            requests_resumee = {
                "sexe": "Veuillez extraire juste le sexe (Femme ou Homme ) à partir de chaque CV sans mentionner d'informations supplémentaires, tu peux conclure le sexe à partir du prénom si c'est possible",
                "formation": "Veuillez extraire juste les mots clés des formations à partir de chaque CV sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)",
                "localisation": "Veuillez extraire juste la localisation actuelle (plus récente) à partir de chaque CV sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)",
                "compétences": "Veuillez extraire juste les mots clés des compétences à partir de chaque CV sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)",
                "expériences": "Veuillez extraire juste les mots clés des expériences à partir de chaque CV sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)"
            }
            requests_job = {
                "sexe": "Veuillez extraire juste le sexe demandé pour cet emploie sans mentionner d'informations supplémentaires",
                "formation": "Veuillez extraire juste les mots clés de la formation demandée pour cet emploie sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)",
                "localisation": "Veuillez extraire juste la localisation demandée pour cet emploie sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)",
                "compétences": "Veuillez extraire juste les mots clés des compétences demandées pour cet emploie  sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)",
                "expériences": "Veuillez extraire juste les mots clés des expériences demandées pour cet emploie  sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)"
            }

            resumee_responses = {key: [get_text_info_from_pdf(cv, req) for cv in resumee] for key, req in requests_resumee.items()}
            resumee_responses["global"] = [get_text_info_from_pdf(cv, "Veuillez extraire juste les mots clés du sexe, la formation, le lieu, les compétences et expériences professionnelles mentionnées dans chaque CV sans écrire le titre (informations...) et sans mentionner d'informations supplémentaires") for cv in resumee]
            
            job_description_responses = {key: get_text_info_from_pdf(job_description, req) for key, req in requests_job.items()}
            job_description_responses["global"] = get_text_info_from_pdf(job_description, "Veuillez extraire juste les mots clés du sexe, la formation, le lieu, les compétences et expériences professionnelles demandées dans cet offre d'emploie sans écrire le titre (informations...) sans mentionner d'informations supplémentaires")

            with col_a:
                st.write("**Informations des CVs**")
                if len(resumee_responses["global"]) > 1:
                    tabs = st.tabs([f"CV {i+1}" for i in range(len(resumee_responses["global"]))])
                    for i, tab in enumerate(tabs):
                        with tab:
                            st.write(resumee_responses["global"][i])
                else:
                    st.write(resumee_responses["global"][0])

            with col_b:
                st.write("**Informations de l'offre d'emploi**")
                st.write(job_description_responses["global"])

    with st.expander(" "):
        with st.spinner("En cours..."):
            tabs = st.tabs(["CVs", "Offre d'emploi"])

            for key in requests_resumee:
                parts = extract_parts(resumee_responses[key])
                keywords_resumee[key].extend(parts)

            for key in requests_job:
                parts_ = extract_parts(job_description_responses[key])
                keywords_job_desc[key].extend(parts_)

            with tabs[0]:
                st.write(keywords_resumee)

            with tabs[1]:
                st.write(keywords_job_desc)
            


