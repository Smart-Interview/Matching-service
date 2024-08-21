import spacy
import streamlit as st
import pdfplumber
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

# Charger le modèle pré-entraîné de spaCy
nlp = spacy.load("en_core_web_sm")

# Fonction pour convertir un PDF en texte
def pdf_to_text(pdf_file):
    text = ""
    pdf_file.seek(0)
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Fonction pour extraire les entités avec spaCy
def extract_entities_spacy(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Fonction pour obtenir les embeddings d'un texte
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    last_hidden_state = outputs[0]
    cls_embedding = last_hidden_state[:, 0, :].detach().cpu().numpy()
    return cls_embedding

# Fonction pour classer les CVs en fonction des embeddings
def rank_cvs_with_embeddings(cvs, job_description, top_n=3):
    job_embedding = get_embedding(' '.join(job_description))
    cv_embeddings = [get_embedding(' '.join(cv)) for cv in cvs]
    similarities = [cosine_similarity(cv_emb, job_embedding).flatten()[0] for cv_emb in cv_embeddings]
    ranked_indices = np.argsort(similarities)[::-1]
    top_cvs = [cvs[i] for i in ranked_indices[:top_n]]
    top_scores = [similarities[i] for i in ranked_indices[:top_n]]
    return top_cvs, top_scores

# Interface utilisateur Streamlit
resumee = st.file_uploader("Veuillez télécharger les CVs", type=["pdf", "docx"], accept_multiple_files=False)
job_description = st.text_area("Veuillez coller la description du poste")

# Extraction des informations
if resumee and job_description:
    with st.expander("Extraction des informations"):
        with st.spinner("En cours..."):
            text = pdf_to_text(resumee)
            entities = extract_entities_spacy(text)
            st.write(entities)

    with st.expander("Score"):
        with st.spinner("En cours..."):
            cvs = [entities]  # Suppose entities is in the desired format
            job_desc = [job_description]
            top_cvs, top_scores = rank_cvs_with_embeddings(cvs, job_desc, top_n=1)

            for i, (cv, score) in enumerate(zip(top_cvs, top_scores), 1):
                st.write(f"Rank {i}: CV: {cv}, Cosine Similarity Score: {score}")
