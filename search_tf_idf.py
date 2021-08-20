import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import numpy as np
import dill as pickle
import sqlite3

con = sqlite3.connect("cord.db")
df = pd.read_sql_query("SELECT title, abstract, authors, body_text FROM cord19", con)

with open('data/vectorizer.pickle', 'rb') as file:
    vectorizer = pickle.load(file)
    
with open('data/title_tf_idf.pickle', 'rb') as file:
    title_tf_idf = pickle.load(file)
    
with open('data/document_tf_idf.pickle', 'rb') as file:
    document_tf_idf = pickle.load(file)

title_weight = 0.65
document_weight = 1 - title_weight

query = ['what is the cause of diseases']

query_tf_idf = vectorizer.transform(query)

print(query_tf_idf)

sim = title_weight * cosine_similarity(title_tf_idf, query_tf_idf) + document_weight * cosine_similarity(document_tf_idf, query_tf_idf) 

print(sim.argmax())

df.iloc[sim.argmax()]['title']

df.iloc[sim.argmax()]['abstract']

df.iloc[sim.argmax()]['body_text']