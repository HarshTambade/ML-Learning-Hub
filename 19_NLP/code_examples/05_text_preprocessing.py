"""Text Preprocessing - Clean and normalize text data"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.text = "Hello! This is a test. It's amazing!!!!"
    
    def lowercase(self):
        return self.text.lower()
    
    def remove_punctuation(self, text):
        return re.sub(r'[^\w\s]', '', text)
    
    def remove_stopwords(self, words):
        return [w for w in words if w not in self.stop_words]
    
    def stemming(self, words):
        return [self.stemmer.stem(w) for w in words]
    
    def lemmatization(self, words):
        return [self.lemmatizer.lemmatize(w) for w in words]
    
    def preprocess(self):
        print("=== Text Preprocessing ===")
        text = self.lowercase()
        print(f"Lowercase: {text}")
        text = self.remove_punctuation(text)
        words = text.split()
        print(f"After removing punctuation: {words}")
        clean_words = self.remove_stopwords(words)
        print(f"After removing stopwords: {clean_words}")
        stemmed = self.stemming(clean_words)
        print(f"Stemmed: {stemmed}")

if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    preprocessor.preprocess()
