import pdfplumber
import google.generativeai as genai
import streamlit as st
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re

#start
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy

nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

abbreviation_dict = {
    'ml': 'machine learning',
    'f':"femme",
    'h': "homme",
    'dl': 'deep learning',
    'ai': 'artificial intelligence',
    'rh':"ressources humaines",
    'm2': "master 2",
    'ing': "ingénieur"

}

degrees=["deug","deust","licence","master","doctorat","ingénieur"]

years_suffix=["an","année","ans","années"]

def lemmatize_text(text):
    words = text.split()
    return ' '.join([lemmatizer.lemmatize(word) for word in words])

def array_to_text(arr):
    return ' '.join(arr)

def expand_abbreviations(text):
    words = text.split()
    expanded_text = ' '.join([abbreviation_dict.get(word, word) for word in words])
    return expanded_text

def preprocess_text(text):
    text = text.lower()
    # text = re.sub(r'[\\]', '', text)
    text = re.sub(r"[^\w\s']", '', text)

    #Méthode 1:
    text = re.sub(r"\b(le|la|les|de|des|du|d'|l')\b", ' ', text)

    #Méthode 2:
    # Parse the text using spaCy
    # doc = nlp(text)

    # Remove French articles
    # articles = {"le", "la", "les", "de", "des", "du", "d'", "l'", "un", "une"}
    # text = ' '.join([token.text for token in doc if token.text not in articles])

    text = lemmatize_text(text)
    text = expand_abbreviations(text)
    return text

def compute_similarity(text1, text2):
    # Assure que les deux textes ne sont pas vides
    if not text1.strip() or not text2.strip():
        return 0  # ou une autre valeur par défaut si l'un des textes est vide

    # Configuration de TfidfVectorizer pour inclure des chiffres
    vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\b\w+\b')
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

#end

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

# def get_document_embedding(data):
#     combined_text = ' '.join(data['sexe']) + ' ' + ' '.join(data['formation']) + ' ' + ' '.join(data['localisation']) + ' ' + \
#                     ' '.join(data['competences']) + ' ' + ' '.join(data['experiences'])
#     return get_embedding(combined_text)

# def rank_cvs_with_embeddings(cv, job_description):
#     job_embedding = get_document_embedding(job_description)
#     cv_embedding = get_document_embedding(cv)
#     similarity_score = cosine_similarity(cv_embedding, job_embedding).flatten()[0]
#     return similarity_score 


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
        'experiences': r'\*\*Expérience:\*\*\s*([^*]*)'
    }

    
    for key, pattern in patterns.items():
        match = re.search(pattern, global_info)
        if match:
            extracted = match.group(1).strip()
            keywords[key] = [item.strip() for item in extracted.split(',')]
    
    return keywords

#start

# Rendre les éléments des listes en minuscules et les lemmatiser
def similarity(CV,Offre):
    # Rendre les éléments des listes en minuscules et les lemmatiser
    for key in CV:
        CV[key] = [preprocess_text(element) for element in CV[key]]
    for key in Offre:
        Offre[key] = [preprocess_text(element)  for element in Offre[key]]

    # Conversion des dictionnaires en texte
    #sexe
    if CV['sexe']:
        sexe_CV = CV['sexe'][0]
    if Offre['sexe']:
        sexe_offre = Offre['sexe'][0]
    else:
        sexe_offre=""
        
    #formation
    filtered_Offre_formation=([elem for item in Offre['formation'] for elem in item.split() ])
    filtered_CV_formation = ' '.join(set([elem for item in CV['formation'] for elem in item.split() if elem in filtered_Offre_formation]))
    formation_offre = array_to_text(filtered_Offre_formation)

    #competences
    filtered_CV_competence = ' '.join([item for item in CV['competences'] if item in Offre['competences']])
    competences_offre = array_to_text(Offre['competences'])

    #experience
    filtered_Offre_experience = ' '.join([
        str(0) if elem == 'débutant' else
        str(2) if elem == 'junior' else
        str(5) if elem == 'senior' else
        elem
        for item in Offre['experiences']
        for elem in item.split()
        if elem not in years_suffix
    ])

    # filtered_CV_experience = ' '.join(set([elem for item in CV['experiences'] for elem in item.split() if elem in filtered_Offre_experience]))
    filtered_CV_experience = ' '.join(set([elem for item in CV['experiences'] for elem in item.split() if elem not in years_suffix]))
    experience_offre = array_to_text(filtered_Offre_experience)

    # Calcul de la cosine similarity
    if CV['sexe'] and Offre['sexe']:
        cosine_sim_sexe = compute_similarity(sexe_CV,sexe_offre)
    cosine_sim_formation = compute_similarity(filtered_CV_formation,formation_offre)
    cosine_sim_competences = compute_similarity(filtered_CV_competence,competences_offre)
    cosine_sim_experiences = compute_similarity(filtered_CV_experience,experience_offre)

    if cosine_sim_experiences == 1 or (filtered_CV_experience and filtered_Offre_experience and float(filtered_CV_experience) >= float(filtered_Offre_experience)):
        cosine_sim_experiences=1
    else:
        cosine_sim_experiences=0

    if sexe_offre!='femme' and sexe_offre!='homme':
        cosine_sim_sexe=1
    
    if cosine_sim_competences==0:
        cosine_sim_experiences=0

    cosine_sim=(cosine_sim_sexe+cosine_sim_formation+cosine_sim_competences+cosine_sim_experiences)/4

    if cosine_sim_sexe==0:
        cosine_sim=0

    return cosine_sim
#end