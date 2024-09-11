from gensim.models import Word2Vec

def train_word2vec_model(tokenized_text):
    model = Word2Vec(tokenized_text, vector_size=100, window=5, min_count=1, workers=4)
    return model

def save_word2vec_model(model, filename):
    model.save(filename)
    print(f"Model saved as {filename}")

def load_word2vec_model(filename):
    model = Word2Vec.load(filename)
    return model
