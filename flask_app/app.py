from flask import Flask, request, jsonify, send_from_directory
import os
import pdfplumber
import google.generativeai as genai
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy
from dotenv import load_dotenv
import os
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)


#$ curl -X POST -F "file=@static\data-scientist-f-h.pdf" http://127.0.0.1:5000/upload-offre
#$ curl -X POST -F "file=@static\Bouchra-Benghazala-CV-Recent.pdf" http://127.0.0.1:5000/upload-cv

load_dotenv()  # Load environment variables from .env file

api_key = os.getenv("GENAI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    raise ValueError("API key is missing. Please set the GENAI_API_KEY environment variable.")

vision_model = genai.GenerativeModel('gemini-1.5-flash')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')



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



# Route pour recevoir le fichier du CV 
@app.route('/upload-cv', methods=['POST'])
def upload_cv():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Créer le dossier s'il n'existe pas
    upload_folder = './uploads/cv/'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    # Sauvegarder le fichier CV
    file.save(os.path.join(upload_folder, file.filename))
    return jsonify({"message": f"CV uploaded successfully as {file.filename}"}), 200



# Route pour recevoir le fichier de l'offre d'emploi
@app.route('/upload-offre', methods=['POST'])
def upload_offre():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Créer le dossier s'il n'existe pas
    upload_folder = './uploads/offres/'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    # Sauvegarder le fichier offre d'emploi
    file.save(os.path.join(upload_folder, file.filename))
    return jsonify({"message": f"Offre uploaded successfully as {file.filename}"}), 200

# Route pour accéder aux fichiers CV
@app.route('/files/cv/<filename>')
def serve_cv(filename):
    return send_from_directory('./uploads/cv/', filename)


@app.route('/extractInfoCV/cv/<filename>')
def extractInfoCV(filename):
    file_path = os.path.join('./uploads/cv/', filename)
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()

    request = "Veuillez extraire juste les mots clés du sexe (tu peux déduire le sexe), la formation (tous les formations mentionnées), le lieu (donne moi un seul lieu le plus récent), les compétences, expérience (donne moi juste le nombre d'expérience en années pour la partie expérience) mentionnées de ce CV sans écrire le titre (informations...) et sans mentionner d'informations supplémentaires, le résultat doit être sous forme espace **nom:** espace, ne pas afficher de \\n"

    response = vision_model.generate_content([request, text])

    if response.candidates and response.candidates[0].content:
        # Convertir le contenu en chaîne de caractères
        result = str(response.candidates[0].content)
        
        # Retirer les balises 'parts' et 'role'
        # result = result.replace('\\n', '').replace('\n', '').strip()
        
        # Retirer les accolades de début et de fin si présentes
        if result.startswith('{') and result.endswith('}'):
            result = result[1:-1].strip()
        
        return result
    else:
        return "Aucun résultat disponible pour cette requête."
    
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

@app.route('/extractKeyWordsCV/cv/<filename>')
def extractKeyWordsCV(filename):
    keywords_resumee = parse_global_info(extractInfoCV(filename))
    for key in keywords_resumee:
        keywords_resumee[key] = [item.replace('\\n', '').replace('\n', '') for item in keywords_resumee[key]]
        keywords_resumee['experiences'] = [keywords_resumee['experiences'][0].replace('"}role: "model"', '').strip()]

    return keywords_resumee


# Route pour accéder aux fichiers offres
@app.route('/files/offre/<filename>')
def serve_offre(filename):
    return send_from_directory('./uploads/offres/', filename)

@app.route('/extractInfo_Offre/offre/<filename>')
def extractInfo_Offre(filename):
    file_path = os.path.join('./uploads/offres/', filename)
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()

    request = "Veuillez extraire juste les mots clés du sexe (si ce n'est pas affiché écrit rien), la formation (tous les formations mentionnées), le lieu, les compétences, expérience (donne moi juste le nombre d'expérience en mois ou en années) demandées dans cette offre d'emploi sans écrire le titre (informations...) sans mentionner d'informations supplémentaires, le résultat doit etre sous forme espace **nom:** espace ne pas afficher de \n"

    response = vision_model.generate_content([request, text])

    if response.candidates and response.candidates[0].content:
        # Convertir le contenu en chaîne de caractères
        result = str(response.candidates[0].content)
        
        # Retirer les balises 'parts' et 'role'
        # result = result.replace('\\n', '').replace('\n', '').strip()
        
        # Retirer les accolades de début et de fin si présentes
        if result.startswith('{') and result.endswith('}'):
            result = result[1:-1].strip()
        
        return result
    else:
        return "Aucun résultat disponible pour cette requête."
    
@app.route('/extractKeyWordsOffre/offre/<filename>')
def extractKeyWordsOffre(filename):
    keywords_resumee = parse_global_info(extractInfo_Offre(filename))
    for key in keywords_resumee:
        keywords_resumee[key] = [item.replace('\\n', '').replace('\n', '') for item in keywords_resumee[key]]
        keywords_resumee['experiences'] = [keywords_resumee['experiences'][0].replace('"}role: "model"', '').strip()]

    return keywords_resumee


@app.route('/score/<filenameCV>/<filenameOffre>')
def score(filenameCV, filenameOffre):
    try:
        CV = extractKeyWordsCV(filenameCV)
        Offre = extractKeyWordsOffre(filenameOffre)

        # Rendre les éléments des listes en minuscules et les lemmatiser
        for key in CV:
            CV[key] = [preprocess_text(element) for element in CV[key]]
        for key in Offre:
            Offre[key] = [preprocess_text(element) for element in Offre[key]]

        # Conversion des dictionnaires en texte
        # Sexe
        sexe_CV = CV['sexe'][0] if CV['sexe'] else ""
        sexe_offre = Offre['sexe'][0] if Offre['sexe'] else ""

        # Formation
        filtered_Offre_formation = [elem for item in Offre['formation'] for elem in item.split()]
        filtered_CV_formation = ' '.join(set([elem for item in CV['formation'] for elem in item.split() if elem in filtered_Offre_formation]))
        formation_offre = array_to_text(filtered_Offre_formation)

        # Compétences
        filtered_CV_competence = ' '.join([item for item in CV['competences'] if item in Offre['competences']])
        competences_offre = array_to_text(Offre['competences'])

        # Expérience
        filtered_Offre_experience = ' '.join([
            str(0) if elem == 'débutant' else
            str(2) if elem == 'junior' else
            str(5) if elem == 'senior' else
            elem
            for item in Offre['experiences']
            for elem in item.split()
            if elem not in years_suffix
        ])

        filtered_CV_experience = ' '.join(set([elem for item in CV['experiences'] for elem in item.split() if elem not in years_suffix]))
        experience_offre = array_to_text(filtered_Offre_experience)

        # Calcul de la cosine similarity
        cosine_sim_sexe = compute_similarity(sexe_CV, sexe_offre) if sexe_CV and sexe_offre else 1
        cosine_sim_formation = compute_similarity(filtered_CV_formation, formation_offre)
        cosine_sim_competences = compute_similarity(filtered_CV_competence, competences_offre)
        cosine_sim_experiences = compute_similarity(filtered_CV_experience, experience_offre)

        if cosine_sim_experiences == 1 or (filtered_CV_experience and filtered_Offre_experience and float(filtered_CV_experience) >= float(filtered_Offre_experience)):
            cosine_sim_experiences = 1
        else:
            cosine_sim_experiences = 0

        if sexe_offre not in ['femme', 'homme']:
            cosine_sim_sexe = 1

        if cosine_sim_competences == 0:
            cosine_sim_experiences = 0

        cosine_sim = (cosine_sim_sexe + cosine_sim_formation + cosine_sim_competences + cosine_sim_experiences) / 4

        if cosine_sim_sexe == 0:
            cosine_sim = 0

        status = "accepté" if cosine_sim >= 0.5 else "refusé"
        
        return insert_status(status)

    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
        return "échoué"

def create_connection():
    connection = None
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='username',
            password='password',
            database='database_name'
        )
        if connection.is_connected():
            print("Connected to MySQL database")
    except Error as e:
        print(f"Error: '{e}'")

    return connection

def insert_status(status):
    connection = create_connection()

    if connection is None:
        return "error connecting to the database"
    
    cursor = connection.cursor()
    query = "INSERT INTO results (status) VALUES (%s)"          #results is the table name with status as varchar
    cursor.execute(query, (status,))
    connection.commit()

    print(f"Status '{status}' inserted with ID: {cursor.lastrowid}")

    cursor.close()
    connection.close()

    return "inserted successfully"



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


if __name__ == '__main__':
    app.run(debug=True)
