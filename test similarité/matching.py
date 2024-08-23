from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')



# def get_embedding(text):
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
#     outputs = model(**inputs)

#     cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
#     return cls_embedding
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)

    # outputs is a tuple, the first element contains the last_hidden_state
    last_hidden_state = outputs[0]

    cls_embedding = last_hidden_state[:, 0, :].detach().cpu().numpy()
    return cls_embedding




def get_document_embedding(data):
    combined_text = ' '.join(data['sexe']) + ' '+ ' '.join(data['formation']) + ' ' + ' '.join(data['experience']) + ' ' + \
                    ' '.join(data['competences']) + ' ' + ' '.join(data['location'])
    return get_embedding(combined_text)



def rank_cvs_with_embeddings(cvs, job_description, top_n=3):
    job_embedding = get_document_embedding(job_description)
    cv_embeddings = [get_document_embedding(cv) for cv in cvs]

    similarities = [cosine_similarity(cv_emb, job_embedding).flatten()[0] for cv_emb in cv_embeddings]
    ranked_indices = np.argsort(similarities)[::-1]

    top_cvs = [cvs[i] for i in ranked_indices[:top_n]]
    top_scores = [similarities[i] for i in ranked_indices[:top_n]]

    return top_cvs, top_scores



cvs = [
    {
        "sexe": ["femme"],
        "formation": ["B.Sc. Computer Science"],
        "experience": ["5 years"],
        "competences": ["Python", "Machine Learning", "Data Analysis"],
        "location": ["New York"]
    },
    {
        "sexe": ["homme"],
        "formation": ["B.Sc. Computer Science", "M.Sc. Data Science"],
        "experience": ["4 years"],
        "competences": ["Python", "Deep Learning", "Data Science"],
        "location": ["San Francisco"]
    },
    {
        "sexe": ["homme"],
        "formation": ["M.Sc. Data Science"],
        "experience": ["6 years"],
        "competences": ["Python", "Machine Learning", "Data Science"],
        "location": ["New York"]
    }
]


job_description = {
    "sexe": ["femme"],
    "formation": ["M.Sc. Data Science", "B.Sc. Computer Science"],
    "experience": ["3-5 years"],
    "competences": ["Python", "Machine Learning", "Deep Learning", "NLP"],
    "location": ["San Francisco"]
}

top_cvs, top_scores = rank_cvs_with_embeddings(cvs, job_description, top_n=3)

for i, (cv, score) in enumerate(zip(top_cvs, top_scores), 1):
    print(f"Rank {i}: CV: {cv}, Cosine Similarity Score: {score}")



"""
results
------------------------------------------------------------------
Rank 1: CV: {'sexe': [], 'formation': ['B.Sc. Computer Science', 'M.Sc. Data Science'], 'experience': ['4 years'], 'competences': ['Python', 'Deep Learning', 'Data Science'], 'location': ['San Francisco']}, Cosine Similarity Score: 0.9740996360778809
Rank 2: CV: {'sexe': [], 'formation': ['B.Sc. Computer Science'], 'experience': ['5 years'], 'competences': ['Python', 'Machine Learning', 'Data Analysis'], 'location': ['New York']}, Cosine Similarity Score: 0.9521560072898865
Rank 3: CV: {'sexe': [], 'formation': ['M.Sc. Data Science'], 'experience': ['6 years'], 'competences': ['Python', 'Machine Learning', 'Data Science'], 'location': ['New York']}, Cosine Similarity Score: 0.9332026839256287
-----------------------------------------------------------------
"""
