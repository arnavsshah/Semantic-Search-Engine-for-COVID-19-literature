import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

import faiss  
import pandas as pd
import numpy as np
import dill as pickle
import sqlite3

con = sqlite3.connect("cord.db")
df = pd.read_sql_query("SELECT title, abstract, authors, body_text FROM cord19", con)

model = SentenceTransformer('monologg/biobert_v1.1_pubmed')

if torch.cuda.is_available():
    model = model.to(torch.device("cuda"))


with open('data/title_embeddings.pickle', 'rb') as file:
    title_embeddings = pickle.load(file)
    
with open('data/document_embeddings.pickle', 'rb') as file:
    document_embeddings = pickle.load(file)

num_vectors = document_embeddings.shape[0]
dimension = document_embeddings.shape[1]
num_neighbours = 100
title_weight = 0.6
document_weight = 1 - title_weight

title_index = faiss.IndexFlatIP(dimension)
title_index.add(normalize(title_embeddings, norm='l2'))

document_index = faiss.IndexFlatIP(dimension)
document_index.add(normalize(document_embeddings, norm='l2'))

while(True) :
    query = input('Enter search query :\n')
    query_embedding = model.encode([query])
    query_embedding_normalized = normalize(query_embedding, norm='l2')

    papers_dict = {}

    title_distances, title_indices = title_index.search(query_embedding_normalized, num_neighbours)
    document_distances, document_indices = document_index.search(query_embedding_normalized, num_neighbours)

    papers = list((set(title_indices[0]) & set(document_indices[0])) | set(title_indices[0, -5:]) | set(document_indices[0, -5:]))


    for paper in papers :
        title_distance = 0
        document_distance = 0
        
        if paper in title_indices[0]:
            index = np.where(title_indices[0] == paper)
            title_distance = title_distances[0][index][0]
        else : title_distance = cosine_similarity(title_embeddings[paper].reshape(1, -1), query_embedding)[0][0]
        
        if paper in document_indices[0]:
            index = np.where(document_indices[0] == paper)
            document_distance = document_distances[0][index][0]
        else : document_distance = cosine_similarity(document_embeddings[paper].reshape(1, -1), query_embedding)[0][0]
        
        
        papers_dict[paper] = title_weight * title_distance + document_weight * document_distance
