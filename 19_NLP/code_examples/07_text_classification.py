"""Text Classification - Classify documents into categories"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

class TextClassifier:
    def __init__(self):
        self.train_texts = [
            "I love Python programming",
            "Python is great for machine learning",
            "Java is a popular language",
            "Java is used in enterprises"
        ]
        self.train_labels = [0, 0, 1, 1]  # 0 = Python, 1 = Java
    
    def train_classifier(self):
        print("=== Text Classification ===")
        self.classifier = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', MultinomialNB())
        ])
        self.classifier.fit(self.train_texts, self.train_labels)
        return self.classifier
    
    def predict(self, text):
        prediction = self.classifier.predict([text])[0]
        probability = self.classifier.predict_proba([text])
        label = "Python" if prediction == 0 else "Java"
        print(f"Text: {text}")
        print(f"Classification: {label}")
        print(f"Probabilities: {probability}\n")

if __name__ == "__main__":
    classifier = TextClassifier()
    classifier.train_classifier()
    classifier.predict("I prefer Python for ML")
    classifier.predict("Java enterprise applications")
