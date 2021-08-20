import torch
from sentence_transformers import SentenceTransformer

import nltk
import pandas as pd
import numpy as np
import sqlite3
import dill as pickle

con = sqlite3.connect("cord.db")
df = pd.read_sql_query("SELECT title, abstract, authors, body_text FROM cord19", con)

titles = df['title']
authors = df['authors']
document = df['abstract'] + df['body_text']

# model = SentenceTransformer('monologg/biobert_v1.1_pubmed')

with open('data/biobert.pickle', 'rb') as file :
    model = pickle.load(file)

if torch.cuda.is_available():
    model = model.to(torch.device("cuda"))


title_embeddings = model.encode(titles, show_progress_bar=True)

document_embeddings = model.encode(document, show_progress_bar=True) # 5 mins


with open('data/title_embeddings.pickle', 'wb') as file :
    pickle.dump(title_embeddings, file)

with open('data/document_embeddings.pickle', 'wb') as file :
    pickle.dump(document_embeddings, file)
