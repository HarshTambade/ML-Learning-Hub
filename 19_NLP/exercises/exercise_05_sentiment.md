# Exercise 5: Sentiment Analysis

## Problem
Build a sentiment analyzer that:
1. Analyzes sentiment polarity
2. Classifies text as positive, negative, or neutral
3. Calculates sentiment scores

## Sample Texts
```
Texts:
- "I love this amazing product!"
- "This is terrible and disappointing."
- "The weather is nice today."
```

## Solution
```python
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()

texts = [
    "I love this amazing product!",
    "This is terrible and disappointing.",
    "The weather is nice today."
]

for text in texts:
    scores = sia.polarity_scores(text)
    sentiment = 'positive' if scores['compound'] > 0.05 else ('negative' if scores['compound'] < -0.05 else 'neutral')
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment} ({scores['compound']:.3f})\n")
```

## Expected Output
```
Text: I love this amazing product!
Sentiment: positive (0.851)

Text: This is terrible and disappointing.
Sentiment: negative (-0.788)

Text: The weather is nice today.
Sentiment: neutral (0.320)
```
