from flask import Flask, request, jsonify, send_from_directory
import os
import pdfplumber
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
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


load_dotenv()  # Load environment variables from .env file

api_key = os.getenv("GENAI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    raise ValueError("API key is missing. Please set the GENAI_API_KEY environment variable.")

vision_model = genai.GenerativeModel('gemini-1.5-flash')




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


def extractKeyWordsCV(filename):
    keywords_resumee = parse_global_info(extractInfoCV(filename))
    for key in keywords_resumee:
        keywords_resumee[key] = [item.replace('\\n', '').replace('\n', '') for item in keywords_resumee[key]]
        keywords_resumee['experiences'] = [keywords_resumee['experiences'][0].replace('"}role: "model"', '').strip()]

    return keywords_resumee



def extractInfo_Offre(filename):
    file_path = os.path.join('./uploads/offer/', filename)
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
    

def extractKeyWordsOffre(filename):
    keywords_resumee = parse_global_info(extractInfo_Offre(filename))
    for key in keywords_resumee:
        keywords_resumee[key] = [item.replace('\\n', '').replace('\n', '') for item in keywords_resumee[key]]
        keywords_resumee['experiences'] = [keywords_resumee['experiences'][0].replace('"}role: "model"', '').strip()]

    return keywords_resumee




@app.route('/analyze_data', methods=['POST'])
def recieve_data():

    if 'cv' not in request.files or 'offerId' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    if 'applicationId' not in request.form:
        return jsonify({"error": "app_id is missing"}), 400

    
    cv_file = request.files['cv']
    offre_file = request.files['offerId']

    try:
        app_id = int(request.form['applicationId'])
    except ValueError:
        return jsonify({"error": "app_id must be an integer"}), 400

    
    if cv_file.filename == '' or offre_file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    
    cv_folder = './uploads/cv'
    offer_folder = './uploads/offer'
    if not os.path.exists(cv_folder):
        os.makedirs(cv_folder)
    
    if not os.path.exists(offer_folder):
        os.mkdir(offer_folder)


    cv_file_path = os.path.join(cv_folder, cv_file.filename)
    offre_file_path = os.path.join(offer_folder, offre_file.filename)

    cv_file.save(cv_file_path)
    offre_file.save(offre_file_path)

 
   
    results, status = score(cv_file.filename, offre_file.filename)

    insert_response = insert(results, status, app_id)

    return jsonify({

        "status": status,
        "applicationId": app_id 
        
    }), 200




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


        status = "ACCEPTED" if cosine_sim >= 0.5 else "REFUSED"    

        score = round(float(cosine_sim), 2)

        return score, status

    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
        return "FAILED"

def create_connection():
    connection = None
    try:
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME')
        )
        if connection.is_connected():
            print("Connected to MySQL database")
    except Error as e:
        print(f"Error: '{e}'")

    return connection

def insert(score, status, application_id):

    if score is None or status is None or application_id is None:
        return "error: all arguments must be provided"
    
    connection = create_connection()
    
    if connection is None:
        return "error connecting to the database"
    
    cursor = connection.cursor()
    query = "INSERT INTO results (score, status, application_id) VALUES (%s, %s, %s)"         #results is the table name with status as varchar
    cursor.execute(query, (score, status, application_id))
    connection.commit()

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
