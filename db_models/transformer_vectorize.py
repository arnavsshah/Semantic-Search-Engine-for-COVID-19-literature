import torch
from sentence_transformers import SentenceTransformer

import nltk
import pandas as pd
import numpy as np
import sqlite3
import dill as pickle

conn = sqlite3.connect("../cord.db")

df = pd.read_sql_query("SELECT title, abstract, authors, body_text FROM cord19", conn)

titles = df['title']
document = df['abstract'] + df['body_text']

# model = SentenceTransformer('monologg/biobert_v1.1_pubmed')

with open('../data/biobert.pickle', 'rb') as file :
    model = pickle.load(file)

if torch.cuda.is_available():
    model = model.to(torch.device("cuda"))

title_embeddings = model.encode(titles, show_progress_bar=True)

document_embeddings = model.encode(document, show_progress_bar=True) # 5 mins

with open('data/title_embeddings.pickle', 'wb') as file :
    pickle.dump(title_embeddings, file)

with open('data/document_embeddings.pickle', 'wb') as file :
    pickle.dump(document_embeddings, file)

#for relevant_factors table
df = pd.read_sql_query("SELECT Study, Factors, Excerpt FROM relevant_factors", conn)

study_title = df['Study'].map(str)
study_text = df['Factors'].map(str) + df['Excerpt'].map(str)

study_title_embeddings = model.encode(study_title, show_progress_bar=True) 
study_text_embeddings = model.encode(study_text, show_progress_bar=True)

with open('../data/study_title_embeddings.pickle', 'wb') as file :
    pickle.dump(study_title_embeddings, file)

with open('../data/study_text_embeddings.pickle', 'wb') as file :
    pickle.dump(study_text_embeddings, file)

#for symptoms table
df = pd.read_sql_query("SELECT Study, Excerpt FROM symptoms", conn)

symptoms_study_title = df['Study'].map(str)
symptoms_study_text = df['Excerpt'].map(str)

symptoms_study_title_embeddings = model.encode(symptoms_study_title, show_progress_bar=True) 
symptoms_study_text_embeddings = model.encode(symptoms_study_text, show_progress_bar=True)

with open('../data/symptoms_study_title_embeddings.pickle', 'wb') as file :
    pickle.dump(symptoms_study_title_embeddings, file)

with open('../data/symptoms_study_text_embeddings.pickle', 'wb') as file :
    pickle.dump(symptoms_study_text_embeddings, file)
