import numpy as np
from nltk.corpus import stopwords
from nltk import pos_tag

def filter_terms(terms, min_length=2, unwanted_terms=None):
    stop_words = set(stopwords.words('english'))
    if unwanted_terms:
        stop_words.update(unwanted_terms)
    return [term for term in terms if len(term) >= min_length and term not in stop_words]

def description_similarity_score(description, query_words, model):
    description_words = set(description.split())
    similarity_scores = []
    for q_word in query_words:
        if q_word in model.wv:
            word_scores = [model.wv.similarity(q_word, d_word) for d_word in description_words if d_word in model.wv]
            if word_scores:
                similarity_scores.append(max(word_scores))
    return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0

def get_top_n_descriptions(model, query_text, df, n=3):
    query_words = query_text.split()
    filtered_query_words = filter_terms(query_words)
    description_scores = []
    for _, row in df.iterrows():
        cleaned_description = row['cleaned_text']
        original_description = row['description']
        score = description_similarity_score(cleaned_description, filtered_query_words, model)
        description_scores.append((original_description, score))
    description_scores.sort(key=lambda x: x[1], reverse=True)
    return description_scores[:n]
