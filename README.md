# BBC-News-Article-Retrieval-System-using-FastAPI-TF-IDF-and-Word2Vec
This project implements a news article retrieval system using dataset of BBC articles. It utilizes FastAPI for the API layer, TF-IDF for term weighting, and Word2Vec and allows users to query the dataset for top-matching articles with similarity scores.


## Features

- **Data Cleaning:** The dataset of BBC news articles is cleaned to remove noise and irrelevant information.
- **Preprocessing:** Text is preprocessed to standardize and prepare it for analysis.
- **Model Training:** A Word2Vec model is trained on the cleaned dataset and saved for future use.
- **Search Functionality:** Users can query the system to retrieve the top-matching articles from the dataset, with results including similarity scores.

## Project Structure

- `main.py`: The FastAPI application that handles data loading, model management, and search queries.
- `preprocessing.py`: Functions for loading and cleaning the dataset, as well as preprocessing the text.
- `word2vec_model.py`: Functions for training, saving, and loading the Word2Vec model.
- `similarity.py`: Functions for calculating similarity scores and retrieving top matching descriptions.
- `data/cleaned_bbc_news_articles(1).csv`: Cleaned dataset of BBC news articles.
- `models/word2vec_model.model`: Trained Word2Vec model.

## Installation

1. Clone this repository:

   ```bash
   https://github.com/Ayeshaaaaaaaaa/BBC-News-Article-Retrieval-System-using-FastAPI-TF-IDF-and-Word2Vec.git
   cd bbc-news-retrieval-system
