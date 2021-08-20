import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import numpy as np
import sqlite3
import dill as pickle

con = sqlite3.connect("cord.db")
df = pd.read_sql_query("SELECT title, abstract, authors, body_text FROM cord19", con)

# tokenizer = nltk.RegexpTokenizer(r"\w+")
# documents = df['abstract'].str.lower().apply(tokenizer.tokenize)

# nltk_stop_words = nltk.corpus.stopwords.words('english')
# documents_without_stop_words = []
# for document in documents :
#     documents_without_stop_words.append([word for word in document if word not in nltk_stop_words])

# wordnet_lemmatizer = WordNetLemmatizer()
# for i, document in enumerate(documents_without_stop_words) :
#     documents_without_stop_words[i] = [wordnet_lemmatizer.lemmatize(word) for word in document]

titles = df['title']
authors = df['authors']
abstracts = df['abstract']
body_texts = df['body_text']

def preprocess_text(text):
    tokenizer = nltk.RegexpTokenizer(r'[A-Za-z]+')
    tokens = tokenizer.tokenize(text)
    
    nltk_stop_words = nltk.corpus.stopwords.words('english')
    tokens = [token for token in tokens if token not in nltk_stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t.lower()) for t in tokens]

    return tokens

vectorizer = TfidfVectorizer(analyzer=preprocess_text, min_df=40)

document_tf_idf_fit = vectorizer.fit(titles + abstracts + body_texts)

title_tf_idf = vectorizer.transform(titles)
document_tf_idf = vectorizer.transform(abstracts + body_texts)

feature_names = vectorizer.get_feature_names()
dense = document_tf_idf.todense().tolist()
tfidf = pd.DataFrame(dense, columns=feature_names)
tfidf.head()

with open('data/vectorizer.pickle', 'wb') as file:
    pickle.dump(vectorizer, file)

with open('data/title_tf_idf.pickle', 'wb') as file:
    pickle.dump(title_tf_idf, file)
    
with open('data/document_tf_idf.pickle', 'wb') as file:
    pickle.dump(document_tf_idf, file)
