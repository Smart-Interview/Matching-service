# Importation des modules nécessaires
from flask import Flask, request, jsonify, send_from_directory  # Modules Flask pour créer l'application et gérer les requêtes HTTP
import os  # Module pour interagir avec le système d'exploitation (chemins de fichiers, variables d'environnement, etc.)
import pdfplumber  # Bibliothèque pour extraire du texte à partir de fichiers PDF
import google.generativeai as genai  # Module pour interagir avec les modèles génératifs de Google
from sklearn.metrics.pairwise import cosine_similarity  # Fonction pour calculer la similarité cosinus entre vecteurs
import re  # Module pour les expressions régulières, utilisé dans le traitement de texte
import nltk  # Bibliothèque de traitement du langage naturel
from nltk.stem import WordNetLemmatizer  # Outil de lemmatisation de NLTK
from sklearn.feature_extraction.text import TfidfVectorizer  # Convertit une collection de documents en une matrice TF-IDF
import spacy  # Bibliothèque avancée de traitement du langage naturel (NLP)
from dotenv import load_dotenv  # Charge les variables d'environnement à partir d'un fichier .env
import mysql.connector  # Module pour se connecter et interagir avec une base de données MySQL
from mysql.connector import Error  # Gestion des erreurs pour MySQL

# Initialisation de l'application Flask
app = Flask(__name__)

# Chargement des variables d'environnement depuis un fichier .env
load_dotenv()

# Récupération de la clé API pour Google Generative AI à partir des variables d'environnement
api_key = os.getenv("GENAI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)  # Configuration de l'API avec la clé
else:
    # Lève une erreur si la clé API est manquante
    raise ValueError("API key is missing. Please set the GENAI_API_KEY environment variable.")

# Initialisation du modèle génératif de vision avec le modèle spécifié
vision_model = genai.GenerativeModel('gemini-1.5-flash')

# Téléchargement des ressources nécessaires de NLTK pour la lemmatisation
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialisation du lemmatiseur
lemmatizer = WordNetLemmatizer()

# Dictionnaire pour étendre les abréviations courantes en leurs formes complètes
abbreviation_dict = {
    'ml': 'machine learning',
    'f': "femme",
    'h': "homme",
    'dl': 'deep learning',
    'ai': 'artificial intelligence',
    'rh': "ressources humaines",
    'm2': "master 2",
    'ing': "ingénieur"
}

# Liste des diplômes pour identifier les formations dans le CV et l'offre
degrees = ["deug", "deust", "licence", "master", "doctorat", "ingénieur"]

# Suffixes liés aux années pour interpréter la durée des expériences
years_suffix = ["an", "année", "ans", "années"]

def extractInfoCV(filename):
    """
    Extrait des informations clés d'un CV au format PDF en utilisant le modèle génératif.

    Args:
        filename (str): Nom du fichier PDF du CV.

    Returns:
        str: Chaîne de caractères contenant les mots clés extraits ou un message d'erreur.
    """
    # Construction du chemin complet vers le fichier CV
    file_path = os.path.join('./uploads/cv/', filename)
    text = ""

    # Ouverture et lecture du PDF avec pdfplumber
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text  # Concatenation du texte extrait de chaque page

    # Requête en français pour extraire des mots clés spécifiques du CV
    request = (
        "Veuillez extraire juste les mots clés du sexe (tu peux déduire le sexe), la formation (tous les formations mentionnées), "
        "le lieu (donne moi un seul lieu le plus récent), les compétences, expérience (donne moi juste le nombre d'expérience en années "
        "pour la partie expérience) mentionnées de ce CV sans écrire le titre (informations...) et sans mentionner d'informations supplémentaires, "
        "le résultat doit être sous forme espace **nom:** espace, ne pas afficher de \\n"
    )

    # Utilisation du modèle génératif pour obtenir une réponse basée sur la requête et le texte extrait
    response = vision_model.generate_content([request, text])

    # Vérification et traitement de la réponse du modèle
    if response.candidates and response.candidates[0].content:
        result = str(response.candidates[0].content)  # Conversion du contenu en chaîne de caractères

        # Retrait des accolades de début et de fin si présentes
        if result.startswith('{') and result.endswith('}'):
            result = result[1:-1].strip()

        return result  # Retourne les mots clés extraits
    else:
        return "Aucun résultat disponible pour cette requête."  # Message d'erreur si aucun résultat

def parse_global_info(global_info):
    """
    Analyse les informations globales extraites et les organise dans un dictionnaire.

    Args:
        global_info (str): Chaîne de caractères contenant les informations extraites.

    Returns:
        dict: Dictionnaire avec les mots clés organisés par catégorie.
    """
    # Initialisation du dictionnaire pour stocker les mots clés
    keywords = {
        'sexe': [],
        'formation': [],
        'localisation': [],
        'competences': [],
        'experiences': []
    }

    # S'assurer que global_info est une chaîne de caractères
    global_info = str(global_info)

    # Définition des motifs regex pour extraire chaque catégorie de mots clés
    patterns = {
        'sexe': r'\*\*Sexe:\*\*\s*([^*]*)',
        'formation': r'\*\*Formation:\*\*\s*([^*]*)',
        'localisation': r'\*\*Lieu:\*\*\s*([^*]*)',
        'competences': r'\*\*Compétences:\*\*\s*([^*]*)',
        'experiences': r'\*\*Expérience:\*\*\s*([^*]*)'
    }

    # Parcours de chaque catégorie et extraction des informations correspondantes
    for key, pattern in patterns.items():
        match = re.search(pattern, global_info)
        if match:
            extracted = match.group(1).strip()
            # Séparation des éléments par virgule et suppression des espaces superflus
            keywords[key] = [item.strip() for item in extracted.split(',')]

    return keywords  # Retourne le dictionnaire des mots clés

def extractKeyWordsCV(filename):
    """
    Extrait et nettoie les mots clés d'un CV.

    Args:
        filename (str): Nom du fichier PDF du CV.

    Returns:
        dict: Dictionnaire des mots clés nettoyés.
    """
    # Extraction des mots clés bruts à partir du CV
    keywords_resumee = parse_global_info(extractInfoCV(filename))

    # Nettoyage des mots clés en supprimant les sauts de ligne et autres caractères indésirables
    for key in keywords_resumee:
        keywords_resumee[key] = [item.replace('\\n', '').replace('\n', '') for item in keywords_resumee[key]]
        # Nettoyage spécifique pour les expériences
        if key == 'experiences' and keywords_resumee['experiences']:
            keywords_resumee['experiences'] = [keywords_resumee['experiences'][0].replace('"}role: "model"', '').strip()]

    return keywords_resumee  # Retourne les mots clés nettoyés

def extractInfo_Offre(filename):
    """
    Extrait des informations clés d'une offre d'emploi au format PDF en utilisant le modèle génératif.

    Args:
        filename (str): Nom du fichier PDF de l'offre d'emploi.

    Returns:
        str: Chaîne de caractères contenant les mots clés extraits ou un message d'erreur.
    """
    # Construction du chemin complet vers le fichier d'offre
    file_path = os.path.join('./uploads/offer/', filename)
    text = ""

    # Ouverture et lecture du PDF avec pdfplumber
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text  # Concatenation du texte extrait de chaque page

    # Requête en français pour extraire des mots clés spécifiques de l'offre
    request = (
        "Veuillez extraire juste les mots clés du sexe (si ce n'est pas affiché écrit rien), la formation (tous les formations mentionnées), "
        "le lieu, les compétences, expérience (donne moi juste le nombre d'expérience en mois ou en années) demandées dans cette offre d'emploi "
        "sans écrire le titre (informations...) sans mentionner d'informations supplémentaires, le résultat doit etre sous forme espace **nom:** espace ne pas afficher de \n"
    )

    # Utilisation du modèle génératif pour obtenir une réponse basée sur la requête et le texte extrait
    response = vision_model.generate_content([request, text])

    # Vérification et traitement de la réponse du modèle
    if response.candidates and response.candidates[0].content:
        result = str(response.candidates[0].content)  # Conversion du contenu en chaîne de caractères

        # Retrait des accolades de début et de fin si présentes
        if result.startswith('{') and result.endswith('}'):
            result = result[1:-1].strip()

        return result  # Retourne les mots clés extraits
    else:
        return "Aucun résultat disponible pour cette requête."  # Message d'erreur si aucun résultat

def extractKeyWordsOffre(filename):
    """
    Extrait et nettoie les mots clés d'une offre d'emploi.

    Args:
        filename (str): Nom du fichier PDF de l'offre d'emploi.

    Returns:
        dict: Dictionnaire des mots clés nettoyés.
    """
    # Extraction des mots clés bruts à partir de l'offre d'emploi
    keywords_resumee = parse_global_info(extractInfo_Offre(filename))

    # Nettoyage des mots clés en supprimant les sauts de ligne et autres caractères indésirables
    for key in keywords_resumee:
        keywords_resumee[key] = [item.replace('\\n', '').replace('\n', '') for item in keywords_resumee[key]]
        # Nettoyage spécifique pour les expériences
        if key == 'experiences' and keywords_resumee['experiences']:
            keywords_resumee['experiences'] = [keywords_resumee['experiences'][0].replace('"}role: "model"', '').strip()]

    return keywords_resumee  # Retourne les mots clés nettoyés

@app.route('/analyze_data', methods=['POST'])
def recieve_data():
    """
    Endpoint API pour recevoir les données de CV et d'offre d'emploi, les analyser, scorer et stocker les résultats.

    Méthode: POST
    """
    # Vérification de la présence des fichiers 'cv' et 'offerId' dans la requête
    if 'cv' not in request.files or 'offerId' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400  # Retourne une erreur 400 si les fichiers sont manquants

    # Vérification de la présence du champ 'applicationId' dans les données de formulaire
    if 'applicationId' not in request.form:
        return jsonify({"error": "app_id is missing"}), 400  # Retourne une erreur 400 si 'applicationId' est manquant

    # Récupération des fichiers et de l'ID d'application
    cv_file = request.files['cv']
    offre_file = request.files['offerId']

    try:
        app_id = int(request.form['applicationId'])  # Conversion de 'applicationId' en entier
    except ValueError:
        return jsonify({"error": "app_id must be an integer"}), 400  # Retourne une erreur 400 si la conversion échoue

    # Vérification que les noms des fichiers ne sont pas vides
    if cv_file.filename == '' or offre_file.filename == '':
        return jsonify({"error": "No file selected"}), 400  # Retourne une erreur 400 si aucun fichier n'est sélectionné

    # Définition des dossiers de destination pour les fichiers CV et offres
    cv_folder = './uploads/cv'
    offer_folder = './uploads/offer'

    # Création des dossiers s'ils n'existent pas
    if not os.path.exists(cv_folder):
        os.makedirs(cv_folder)

    if not os.path.exists(offer_folder):
        os.mkdir(offer_folder)

    # Construction des chemins complets pour sauvegarder les fichiers
    cv_file_path = os.path.join(cv_folder, cv_file.filename)
    offre_file_path = os.path.join(offer_folder, offre_file.filename)

    # Sauvegarde des fichiers sur le serveur
    cv_file.save(cv_file_path)
    offre_file.save(offre_file_path)

    # Appel de la fonction de scoring avec les noms de fichiers sauvegardés
    results, status = score(cv_file.filename, offre_file.filename)

    # Insertion des résultats dans la base de données avec l'ID d'application
    insert_response = insert(results, status, app_id)

    # Retourne une réponse JSON avec le statut et l'ID d'application
    return jsonify({
        "status": status,
        "applicationId": app_id
    }), 200  # Retourne un code HTTP 200 pour indiquer le succès

def score(filenameCV, filenameOffre):
    """
    Calcule un score de similarité entre un CV et une offre d'emploi basée sur différents critères.

    Args:
        filenameCV (str): Nom du fichier PDF du CV.
        filenameOffre (str): Nom du fichier PDF de l'offre d'emploi.

    Returns:
        tuple: (score de similarité (float), statut (str)) ou "FAILED" en cas d'erreur.
    """
    try:
        # Extraction des mots clés du CV et de l'offre
        CV = extractKeyWordsCV(filenameCV)
        Offre = extractKeyWordsOffre(filenameOffre)

        # Prétraitement des mots clés : mise en minuscules et lemmatisation
        for key in CV:
            CV[key] = [preprocess_text(element) for element in CV[key]]
        for key in Offre:
            Offre[key] = [preprocess_text(element) for element in Offre[key]]

        # Conversion des listes de mots clés en textes pour la similarité
        # Sexe
        sexe_CV = CV['sexe'][0] if CV['sexe'] else ""
        sexe_offre = Offre['sexe'][0] if Offre['sexe'] else ""

        # Formation
        # Filtrage des formations de l'offre pour n'inclure que celles présentes dans le CV
        filtered_Offre_formation = [elem for item in Offre['formation'] for elem in item.split()]
        filtered_CV_formation = ' '.join(
            set([elem for item in CV['formation'] for elem in item.split() if elem in filtered_Offre_formation])
        )
        formation_offre = array_to_text(filtered_Offre_formation)

        # Compétences
        # Filtrage des compétences du CV pour n'inclure que celles présentes dans l'offre
        filtered_CV_competence = ' '.join([item for item in CV['competences'] if item in Offre['competences']])
        competences_offre = array_to_text(Offre['competences'])

        # Expérience
        # Transformation des niveaux d'expérience en valeurs numériques
        filtered_Offre_experience = ' '.join([
            str(0) if elem == 'débutant' else
            str(2) if elem == 'junior' else
            str(5) if elem == 'senior' else
            elem
            for item in Offre['experiences']
            for elem in item.split()
            if elem not in years_suffix
        ])

        # Filtrage des années d'expérience du CV en excluant les suffixes d'années
        filtered_CV_experience = ' '.join(
            set([elem for item in CV['experiences'] for elem in item.split() if elem not in years_suffix])
        )
        experience_offre = array_to_text(filtered_Offre_experience)

        # Calcul des similarités cosinus pour chaque critère
        cosine_sim_sexe = compute_similarity(sexe_CV, sexe_offre) if sexe_CV and sexe_offre else 1
        cosine_sim_formation = compute_similarity(filtered_CV_formation, formation_offre)
        cosine_sim_competences = compute_similarity(filtered_CV_competence, competences_offre)
        cosine_sim_experiences = compute_similarity(filtered_CV_experience, experience_offre)

        # Ajustement de la similarité pour l'expérience
        if cosine_sim_experiences == 1 or (filtered_CV_experience and filtered_Offre_experience and float(filtered_CV_experience) >= float(filtered_Offre_experience)):
            cosine_sim_experiences = 1
        else:
            cosine_sim_experiences = 0

        # Si le sexe demandé dans l'offre n'est pas spécifié comme 'femme' ou 'homme', on ignore la similarité
        if sexe_offre not in ['femme', 'homme']:
            cosine_sim_sexe = 1

        # Si les compétences ne correspondent pas du tout, l'expérience est également considérée comme non correspondante
        if cosine_sim_competences == 0:
            cosine_sim_experiences = 0

        # Calcul de la similarité globale comme la moyenne des similarités individuelles
        cosine_sim = (cosine_sim_sexe + cosine_sim_formation + cosine_sim_competences + cosine_sim_experiences) / 4

        # Si le sexe ne correspond pas, la similarité globale est annulée
        if cosine_sim_sexe == 0:
            cosine_sim = 0

        # Détermination du statut basé sur le seuil de similarité
        status = "ACCEPTED" if cosine_sim >= 0.5 else "REFUSED"

        # Arrondissement du score à deux décimales
        score = round(float(cosine_sim), 2)

        return score, status  # Retourne le score et le statut

    except Exception as e:
        # En cas d'erreur, affiche le message d'erreur et retourne "FAILED"
        print(f"Une erreur s'est produite : {e}")
        return "FAILED"

def create_connection():
    """
    Crée une connexion à la base de données MySQL en utilisant les variables d'environnement.

    Returns:
        mysql.connector.connection_cext.CMySQLConnection or None: Objet de connexion ou None en cas d'échec.
    """
    connection = None
    try:
        # Tentative de connexion à la base de données avec les informations d'identification
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST'),  # Hôte de la base de données
            user=os.getenv('DB_USER'),  # Utilisateur de la base de données
            password=os.getenv('DB_PASSWORD'),  # Mot de passe de la base de données
            database=os.getenv('DB_NAME')  # Nom de la base de données
        )
        if connection.is_connected():
            print("Connected to MySQL database")  # Confirmation de la connexion réussie
    except Error as e:
        # Affiche le message d'erreur en cas de problème de connexion
        print(f"Error: '{e}'")

    return connection  # Retourne l'objet de connexion ou None

def insert(score, status, application_id):
    """
    Insère les résultats de l'analyse dans la table 'results' de la base de données.

    Args:
        score (float): Score de similarité calculé.
        status (str): Statut basé sur le score ('ACCEPTED' ou 'REFUSED').
        application_id (int): Identifiant de l'application.

    Returns:
        str: Message indiquant le succès ou une erreur.
    """
    # Vérification que tous les arguments sont fournis
    if score is None or status is None or application_id is None:
        return "error: all arguments must be provided"

    # Création de la connexion à la base de données
    connection = create_connection()

    if connection is None:
        return "error connecting to the database"  # Retourne une erreur si la connexion échoue

    # Initialisation du curseur pour exécuter les requêtes SQL
    cursor = connection.cursor()
    query = "INSERT INTO results (score, status, application_id) VALUES (%s, %s, %s)"  # Requête SQL pour l'insertion
    cursor.execute(query, (score, status, application_id))  # Exécution de la requête avec les valeurs fournies
    connection.commit()  # Validation de la transaction

    # Fermeture du curseur et de la connexion
    cursor.close()
    connection.close()
    return "inserted successfully"  # Message de succès

def lemmatize_text(text):
    """
    Lemmatiser un texte en réduisant chaque mot à sa forme de base.

    Args:
        text (str): Texte à lemmatiser.

    Returns:
        str: Texte lemmatisé.
    """
    words = text.split()  # Séparation du texte en mots
    # Application du lemmatiseur à chaque mot et reconstruction du texte
    return ' '.join([lemmatizer.lemmatize(word) for word in words])

def array_to_text(arr):
    """
    Convertit une liste de chaînes en une seule chaîne séparée par des espaces.

    Args:
        arr (list): Liste de chaînes de caractères.

    Returns:
        str: Chaîne unique résultante.
    """
    return ' '.join(arr)

def expand_abbreviations(text):
    """
    Remplace les abréviations dans le texte par leurs formes complètes selon le dictionnaire.

    Args:
        text (str): Texte contenant des abréviations.

    Returns:
        str: Texte avec les abréviations étendues.
    """
    words = text.split()  # Séparation du texte en mots
    # Remplacement des abréviations par leur forme complète si elles existent dans le dictionnaire
    expanded_text = ' '.join([abbreviation_dict.get(word, word) for word in words])
    return expanded_text

def preprocess_text(text):
    """
    Prétraite le texte en le nettoyant, le lemmatisant et en étendant les abréviations.

    Args:
        text (str): Texte brut à prétraiter.

    Returns:
        str: Texte prétraité.
    """
    text = text.lower()  # Conversion en minuscules
    # Suppression des caractères spéciaux sauf les apostrophes
    text = re.sub(r"[^\w\s']", '', text)

    # Méthode 1: Suppression des articles français courants
    text = re.sub(r"\b(le|la|les|de|des|du|d'|l')\b", ' ', text)

    # Méthode 2: (Commentée) Utilisation de spaCy pour supprimer les articles
    # doc = nlp(text)
    # articles = {"le", "la", "les", "de", "des", "du", "d'", "l'", "un", "une"}
    # text = ' '.join([token.text for token in doc if token.text not in articles])

    text = lemmatize_text(text)  # Lemmatisation du texte
    text = expand_abbreviations(text)  # Expansion des abréviations
    return text  # Retourne le texte prétraité

def compute_similarity(text1, text2):
    """
    Calcule la similarité cosinus entre deux textes.

    Args:
        text1 (str): Premier texte.
        text2 (str): Deuxième texte.

    Returns:
        float: Valeur de similarité cosinus entre 0 et 1.
    """
    # Vérifie que les deux textes ne sont pas vides
    if not text1.strip() or not text2.strip():
        return 0  # Retourne 0 si l'un des textes est vide

    # Configuration de TfidfVectorizer pour inclure les chiffres et les mots
    vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\b\w+\b')
    vectors = vectorizer.fit_transform([text1, text2])  # Transformation des textes en vecteurs TF-IDF
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]  # Calcul de la similarité cosinus

    return similarity  # Retourne la similarité

if __name__ == '__main__':
    # Lancement de l'application Flask en mode debug
    app.run(debug=True)
