from transformers import BertTokenizer, BertModel
import streamlit as st
import google.generativeai as genai
from PIL import Image
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import re

genai.configure(api_key=st.secrets["api_key"])
vision_model = genai.GenerativeModel('gemini-1.5-flash')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

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
    # Assurez-vous que content est une chaîne de caractères
    if hasattr(content, 'parts'):
        parts_text = [part.text for part in content.parts]
    else:
        parts_text = [content]
    return ', '.join(parts_text)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    last_hidden_state = outputs[0]
    cls_embedding = last_hidden_state[:, 0, :].detach().cpu().numpy()
    return cls_embedding

def get_document_embedding(data):
    combined_text = ' '.join(data['sexe']) + ' ' + ' '.join(data['formation']) + ' ' + ' '.join(data['localisation']) + ' ' + \
                    ' '.join(data['compétences']) + ' ' + ' '.join(data['expériences'])
    return get_embedding(combined_text)

def rank_cvs_with_embeddings(cv, job_description):
    job_embedding = get_document_embedding(job_description)
    cv_embedding = get_document_embedding(cv)
    similarity_score = cosine_similarity(cv_embedding, job_embedding).flatten()[0]
    return similarity_score

def parse_global_info(global_info):
    keywords = {
        'sexe': [],
        'formation': [],
        'localisation': [],
        'compétences': [],
        'expériences': []
    }
    
    # Assurez-vous que global_info est une chaîne de caractères
    global_info = str(global_info)
    
    patterns = {
        'sexe': r'\*\*Sexe:\*\*\s*([^*]*)',
        'formation': r'\*\*Formation:\*\*\s*([^*]*)',
        'localisation': r'\*\*Lieu:\*\*\s*([^*]*)',
        'compétences': r'\*\*Compétences:\*\*\s*([^*]*)',
        'expériences': r'\*\*Expérience professionnelle:\*\*\s*([^*]*)'
    }

    
    for key, pattern in patterns.items():
        match = re.search(pattern, global_info)
        if match:
            extracted = match.group(1).strip()
            keywords[key] = [item.strip() for item in extracted.split(',')]
    
    return keywords



resumee = st.file_uploader("Veuillez télécharger les CVs", type=["pdf", "docx"], accept_multiple_files=False)
job_description = st.file_uploader("Veuillez télécharger l'offre d'emploi", type=["pdf", "docx"], accept_multiple_files=False)

if resumee and job_description:
    with st.expander("Extraction des informations"):
        with st.spinner("En cours..."):
            col_a, col_b = st.columns(2)

            requests_resumee = {
                "sexe": "Veuillez extraire juste le sexe (Femme ou Homme) à partir du CV sans mentionner d'informations supplémentaires, tu peux conclure le sexe à partir du prénom si c'est possible",
                "formation": "Veuillez extraire juste les mots clés des formations à partir du CV sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)",
                "localisation": "Veuillez extraire juste la localisation actuelle (plus récente) à partir du CV sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)",
                "compétences": "Veuillez extraire juste les mots clés des compétences à partir du CV sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)",
                "expériences": "Veuillez extraire juste les mots clés des expériences à partir du CV sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)"
            }
            requests_job = {
                "sexe": "Veuillez extraire juste le sexe demandé pour cet emploi sans mentionner d'informations supplémentaires",
                "formation": "Veuillez extraire juste les mots clés de la formation demandée pour cet emploi sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)",
                "localisation": "Veuillez extraire juste la localisation demandée pour cet emploi sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)",
                "compétences": "Veuillez extraire juste les mots clés des compétences demandées pour cet emploi sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)",
                "expériences": "Veuillez extraire juste les mots clés des expériences demandées pour cet emploi sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)"
            }

            resumee_responses = {key: get_text_info_from_pdf(resumee, req) for key, req in requests_resumee.items()}
            resumee_responses["global"] = get_text_info_from_pdf(resumee, "Veuillez extraire juste les mots clés du sexe (tu peux déduire le sexe), la formation, le lieu, les compétences, expérience professionnelle mentionnées de ce CV sans écrire le titre (informations...) et sans mentionner d'informations supplémentaires, le résultat doit etre sous forme espace **nom:** espace, ne pas afficher de \n") 
            
            job_description_responses = {key: get_text_info_from_pdf(job_description, req) for key, req in requests_job.items()}
            job_description_responses["global"] = get_text_info_from_pdf(job_description, "Veuillez extraire juste les mots clés du sexe, la formation, le lieu, les compétences, expérience professionnelle demandées dans cette offre d'emploi sans écrire le titre (informations...) sans mentionner d'informations supplémentaires, le résultat doit etre sous forme espace **nom:** espace ne pas afficher de \n")

            with col_a:
                st.write("**Informations du CV**")
                st.write(resumee_responses["global"])

            with col_b:
                st.write("**Informations de l'offre d'emploi**")
                st.write(job_description_responses["global"])

    with st.expander("Extraction des mots-clés"):
        with st.spinner("En cours..."):
            tabs = st.tabs(["CVs", "Offre d'emploi"])

            keywords_resumee = parse_global_info(resumee_responses["global"])
            keywords_job_desc = parse_global_info(job_description_responses["global"])

            with tabs[0]:
                st.write(keywords_resumee)

            with tabs[1]:
                st.write(keywords_job_desc)

    with st.expander("Score"):
        with st.spinner("En cours..."):
            score = rank_cvs_with_embeddings(keywords_resumee, keywords_job_desc)
            st.write(f"Le score de compatibilité entre le CV et l'offre d'emploi est : {score}")
