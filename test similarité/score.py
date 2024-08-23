from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def tokenize_with_bert(text):
    tokens = tokenizer.tokenize(text)
    return " ".join(tokens)



def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)

    cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return cls_embedding



def get_document_embedding(data):
    combined_text = ' '.join(data['formation']) + ' ' + ' '.join(data['experience']) + ' ' + \
                    ' '.join(data['competences']) + ' ' + ' '.join(data['location'])
    return combined_text



def rank_cvs_with_embeddings(cvs, job_description, top_n=3):

    job_total = tokenize_with_bert(get_document_embedding(job_description))
    cv_total = [tokenize_with_bert(get_document_embedding(cv)) for cv in cvs]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([job_total] + cv_total)

    print(f' len of vocab is {len([job_total] + cv_total)}')


    print(f'these are the vectors: {vectors}')

    print(f'vectorizer info {vectorizer.get_feature_names_out()}')
    print(f'vectorizer length {len(vectorizer.get_feature_names_out())}')


    print(f'cosine similarity {cosine_similarity(vectors[0:1], vectors[1:])}')
    print(f'cosine similarity after flattening {cosine_similarity(vectors[0:1], vectors[1:]).flatten()}')


    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    best_match_index = cosine_similarities.argmax()
    best_match_score = cosine_similarities[best_match_index]



    return best_match_index, best_match_score



cvs = [
    {
        "sexe": ["H"],
        "formation": ["B.Sc. Computer Science"],
        "experience": ["5 years"],
        "competences": ["Python", "Machine Learning", "Data Analysis"],
        "location": ["New York"]
    },
    {
        "sexe": ["F"],
        "formation": ["B.Sc. Computer Science", "M.Sc. Data Science"],
        "experience": ["4 years"],
        "competences": ["Python", "Deep Learning", "Data Science"],
        "location": ["San Francisco"]
    },
    {
        "sexe": ["H"],
        "formation": ["M.Sc. Data Science"],
        "experience": ["two"],
        "competences": ["Python", "Machine Learning", "Data Science"],
        "location": ["New York"]
    },
    {
        "sexe": ["F"],
        "formation": ["M.Sc. Data Science", "B.Sc. Computer Science"],
        "experience": ["3-5 years"],
        "competences": ["Python", "Machine Learning", "Deep Learning", "NLP"],
        "location": ["New York"]
    }
]


job_description = {
    "sexe": ["H"],
    "formation": ["M.Sc. Data Science", "B.Sc. Computer Science"],
    "experience": ["two"],
    "competences": ["Python", "Machine Learning", "Deep Learning", "NLP"],
    "location": ["New York"]
}

best_match_index, best_match_score =  rank_cvs_with_embeddings(cvs, job_description)

best_match_cv = cvs[best_match_index]


"""
for i, (cv, score) in enumerate(zip(top_cvs, top_scores), 1):
    print(f"Rank {i}: CV: {cv}, Cosine Similarity Score: {score}")
"""

print(f"Best matching CV: {best_match_cv}")
print(f"Cosine similarity score: {best_match_score}")
