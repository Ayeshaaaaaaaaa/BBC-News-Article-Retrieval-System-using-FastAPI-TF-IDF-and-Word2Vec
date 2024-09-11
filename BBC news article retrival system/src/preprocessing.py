import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    df['pubDate'] = pd.to_datetime(df['pubDate'])
    df['month'] = df['pubDate'].dt.to_period('M')
    return df

def compute_tfidf(df):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_title'])
    return tfidf_vectorizer, tfidf_matrix
