from fastapi import FastAPI
from preprocessing import load_data, preprocess_data
from word2vec_model import train_word2vec_model, save_word2vec_model, load_word2vec_model
from similarity import get_top_n_descriptions
import nltk
import pandas as pd

app = FastAPI()

# Ensure data is loaded and preprocessed on startup
df = load_data('data/cleaned_bbc_news_articles(1).csv')
df = preprocess_data(df)
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Train and save Word2Vec model (if not already saved)
tokenized_text = df['cleaned_text'].apply(lambda x: x.split())
model = train_word2vec_model(tokenized_text)
save_word2vec_model(model, 'models/word2vec_model.model')

# Function to return the root endpoint message
@app.get("/")
def read_root():
    return {"message": "Welcome to the NLP FastAPI app!"}

# Function to handle the search query and return results
@app.get("/search")
def search(query: str):
    # Load the Word2Vec model
    model = load_word2vec_model('models/word2vec_model.model')

    # Get top 3 descriptions with scores
    top_descriptions = get_top_n_descriptions(model, query, df, n=3)

    return {
        "query": query,
        "top_descriptions": [
            {"description": desc, "similarity_score": f"{score:.2%}"}
            for desc, score in top_descriptions
        ],
    }

# Running the code if the file is executed as a script
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
