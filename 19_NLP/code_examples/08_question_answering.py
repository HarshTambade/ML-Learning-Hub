"""Question Answering - Basic QA system using TF-IDF similarity"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimpleQA:
    def __init__(self):
        self.passages = [
            "Python is a high-level programming language.",
            "Machine learning enables computers to learn from data.",
            "Natural language processing deals with human language.",
            "Deep learning uses neural networks with multiple layers."
        ]
        self.vectorizer = TfidfVectorizer()
        self.passage_vectors = self.vectorizer.fit_transform(self.passages)
    
    def answer_question(self, question):
        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, self.passage_vectors)
        most_similar_idx = np.argmax(similarities[0])
        return self.passages[most_similar_idx], similarities[0][most_similar_idx]
    
    def run_qa(self):
        print("=== Question Answering System ===")
        questions = [
            "What is Python?",
            "How does machine learning work?",
            "What is NLP?"
        ]
        
        for q in questions:
            answer, score = self.answer_question(q)
            print(f"Q: {q}")
            print(f"A: {answer}")
            print(f"Confidence: {score:.3f}\n")

if __name__ == "__main__":
    qa = SimpleQA()
    qa.run_qa()
