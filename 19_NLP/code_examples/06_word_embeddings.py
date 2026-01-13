"""Word Embeddings - Convert words to dense vectors"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

class WordEmbeddings:
    def __init__(self):
        self.documents = [
            "Python is a great programming language",
            "Machine learning with Python is powerful",
            "Natural language processing is important"
        ]
    
    def tfidf_embedding(self):
        print("=== TF-IDF Embedding ===")
        vectorizer = TfidfVectorizer(max_features=10)
        tfidf_matrix = vectorizer.fit_transform(self.documents)
        print(f"Shape: {tfidf_matrix.shape}")
        print(f"Vocabulary: {vectorizer.get_feature_names_out()}")
        print(f"Matrix:\n{tfidf_matrix.toarray()}")
    
    def bow_embedding(self):
        print("\n=== Bag of Words Embedding ===")
        vectorizer = CountVectorizer()
        bow_matrix = vectorizer.fit_transform(self.documents)
        print(f"Shape: {bow_matrix.shape}")
        print(f"Matrix:\n{bow_matrix.toarray()}")

if __name__ == "__main__":
    embeddings = WordEmbeddings()
    embeddings.tfidf_embedding()
    embeddings.bow_embedding()
