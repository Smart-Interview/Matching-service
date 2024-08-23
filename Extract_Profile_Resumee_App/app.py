import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import pickle
import re

# def cleanResume(txt):
#     cleanText = re.sub('http\S+\s', ' ', txt)
#     cleanText = re.sub('RT|cc', ' ', cleanText)
#     cleanText = re.sub('#\S+\s', ' ', cleanText)
#     cleanText = re.sub('@\S+', '  ', cleanText)  
#     cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
#     cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
#     cleanText = re.sub('\s+', ' ', cleanText)
#     return cleanText

def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)  
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText


clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd= pickle.load(open('tfidf.pkl','rb'))

Resumee = st.file_uploader("Veuillez télécharger votre CV", type=["txt", "pdf", "docx"])


if Resumee is not None:
    content = Resumee.read().decode("utf-8", errors="ignore")
    cleaned_resume = cleanResume(content)

    input_features = tfidfd.transform([cleaned_resume])

    prediction_id = clf.predict(input_features)[0]

    category_mapping = {
        15: "Java Developer",
        23: "Testing",
        8: "DevOps Engineer",
        20: "Python Developer",
        24: "Web Designing",
        12: "HR",
        13: "Hadoop",
        3: "Blockchain",
        10: "ETL Developer",
        18: "Operations Manager",
        6: "Data Science",
        22: "Sales",
        16: "Mechanical Engineer",
        1: "Arts",
        7: "Database",
        11: "Electrical Engineering",
        14: "Health and fitness",
        19: "PMO",
        4: "Business Analyst",
        9: "DotNet Developer",
        2: "Automation Testing",
        17: "Network Security Engineer",
        21: "SAP Developer",
        5: "Civil Engineer",
        0: "Advocate",
    }

    category_name = category_mapping.get(prediction_id, "Unknown")
    st.write("Predicted Category:", category_name) 
else:
    st.write("Veuillez télécharger un fichier pour la prédiction.")



