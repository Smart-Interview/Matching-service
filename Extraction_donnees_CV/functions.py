import pdfplumber
import google.generativeai as genai
import streamlit as st
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
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
                    ' '.join(data['competences']) + ' ' + ' '.join(data['experiences'])
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
        'competences': [],
        'experiences': []
    }
    
    # Assurez-vous que global_info est une chaîne de caractères
    global_info = str(global_info)
    
    patterns = {
        'sexe': r'\*\*Sexe:\*\*\s*([^*]*)',
        'formation': r'\*\*Formation:\*\*\s*([^*]*)',
        'localisation': r'\*\*Lieu:\*\*\s*([^*]*)',
        'competences': r'\*\*Compétences:\*\*\s*([^*]*)',
        'experiences': r'\*\*Expérience professionnelle:\*\*\s*([^*]*)'
    }

    
    for key, pattern in patterns.items():
        match = re.search(pattern, global_info)
        if match:
            extracted = match.group(1).strip()
            keywords[key] = [item.strip() for item in extracted.split(',')]
    
    return keywords