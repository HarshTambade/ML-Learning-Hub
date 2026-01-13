"""Sentiment Analysis - Determine sentiment in text"""

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon', quiet=True)

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.texts = [
            "I love this product! It's amazing!",
            "This is terrible and disappointing.",
            "The weather is nice today."
        ]
    
    def analyze_sentiment(self):
        print("\n=== Sentiment Analysis ===")
        for text in self.texts:
            scores = self.sia.polarity_scores(text)
            print(f"Text: {text}")
            print(f"Scores: {scores}")
            sentiment = 'positive' if scores['compound'] > 0.05 else ('negative' if scores['compound'] < -0.05 else 'neutral')
            print(f"Sentiment: {sentiment}\n")

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    analyzer.analyze_sentiment()
