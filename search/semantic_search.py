import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

import faiss  
import pandas as pd
import numpy as np
import dill as pickle
import sqlite3


# model = SentenceTransformer('monologg/biobert_v1.1_pubmed')

with open('data/biobert.pickle', 'rb') as file :
    model = pickle.load(file)

if torch.cuda.is_available():
   model = model.to(torch.device("cuda"))

with open('data/title_embeddings.pickle', 'rb') as file:
    title_embeddings = pickle.load(file)
    
with open('data/document_embeddings.pickle', 'rb') as file:
    document_embeddings = pickle.load(file)

with open('data/study_title_embeddings.pickle', 'rb') as file:
    study_title_embeddings = pickle.load(file)
    
with open('data/study_text_embeddings.pickle', 'rb') as file:
    study_text_embeddings = pickle.load(file)

with open('data/symptoms_study_title_embeddings.pickle', 'rb') as file:
    symptoms_study_title_embeddings = pickle.load(file)
    
with open('data/symptoms_study_text_embeddings.pickle', 'rb') as file:
    symptoms_study_text_embeddings = pickle.load(file)


num_vectors = document_embeddings.shape[0]
dimension = document_embeddings.shape[1]
num_neighbours = 100
alpha = 0.6
beta = 1 - alpha

#papers
title_index = faiss.IndexFlatIP(dimension)
title_index.add(normalize(title_embeddings, norm='l2'))

document_index = faiss.IndexFlatIP(dimension)
document_index.add(normalize(document_embeddings, norm='l2'))


#relevant_factors
study_title_index = faiss.IndexFlatIP(dimension)
study_title_index.add(normalize(study_title_embeddings, norm='l2'))

study_text_index = faiss.IndexFlatIP(dimension)
study_text_index.add(normalize(study_text_embeddings, norm='l2'))


#symptoms
symptoms_study_title_index = faiss.IndexFlatIP(dimension)
symptoms_study_title_index.add(normalize(symptoms_study_title_embeddings, norm='l2'))

symptoms_study_text_index = faiss.IndexFlatIP(dimension)
symptoms_study_text_index.add(normalize(symptoms_study_text_embeddings, norm='l2'))


def search(query, type=None) :
    query = [query]
    query_embedding = model.encode(query)
    query_embedding_normalized = normalize(query_embedding, norm='l2')

    papers_score = {}
    title_distances, title_indices, document_distances, document_indices = None, None, None, None
    
    df = None

    selected_title_embeddings, selected_document_embeddings = None, None
    
    conn = sqlite3.connect("cord.db")

    if type == 'symptoms' :
        title_distances, title_indices = symptoms_study_title_index.search(query_embedding_normalized, num_neighbours)
        document_distances, document_indices = symptoms_study_text_index.search(query_embedding_normalized, num_neighbours)
        
        selected_title_embeddings = symptoms_study_title_embeddings
        selected_document_embeddings = symptoms_study_text_embeddings

        df = pd.read_sql_query("SELECT * FROM symptoms", conn)

    elif type == 'factors' :
        title_distances, title_indices = study_title_index.search(query_embedding_normalized, num_neighbours)
        document_distances, document_indices = study_text_index.search(query_embedding_normalized, num_neighbours)
        
        selected_title_embeddings = study_title_embeddings
        selected_document_embeddings = study_text_embeddings

        df = pd.read_sql_query("SELECT * FROM relevant_factors", conn)

    else :
        title_distances, title_indices = title_index.search(query_embedding_normalized, num_neighbours)
        document_distances, document_indices = document_index.search(query_embedding_normalized, num_neighbours)
        
        selected_title_embeddings = title_embeddings
        selected_document_embeddings = document_embeddings

        df = pd.read_sql_query("SELECT * FROM cord19", conn)

    papers = list((set(title_indices[0]) & set(document_indices[0])) | set(title_indices[0, -5:]) | set(document_indices[0, -5:]))

    for paper in papers :
        title_distance = 0
        document_distance = 0
        
        if paper in title_indices[0]:
            index = np.where(title_indices[0] == paper)
            title_distance = title_distances[0][index][0]
        else : title_distance = cosine_similarity(selected_title_embeddings[paper].reshape(1, -1), query_embedding)[0][0]
        
        if paper in document_indices[0]:
            index = np.where(document_indices[0] == paper)
            document_distance = document_distances[0][index][0]
        else : document_distance = cosine_similarity(selected_document_embeddings[paper].reshape(1, -1), query_embedding)[0][0]
            
        papers_score[paper] = alpha * title_distance + beta * document_distance

    sorted_results = {paper_id: score for paper_id, score in sorted(papers_score.items(), key=lambda item: item[1], reverse=True)}
    return df.iloc[list(sorted_results.keys()), :]