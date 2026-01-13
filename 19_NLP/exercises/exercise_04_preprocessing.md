# Exercise 4: Text Preprocessing

## Problem
Implement text preprocessing that:
1. Converts text to lowercase
2. Removes punctuation
3. Removes stopwords
4. Applies stemming

## Sample Text
```
Natural Language Processing (NLP) is amazing! We can process text efficiently.
```

## Solution
```python
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

text = "Natural Language Processing (NLP) is amazing! We can process text efficiently."
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Lowercase
text = text.lower()

# Remove punctuation
text = re.sub(r'[^\w\s]', '', text)
words = text.split()

# Remove stopwords and stem
processed = [stemmer.stem(w) for w in words if w not in stop_words]
print(f"Processed: {processed}")
```

## Expected Output
```
Processed: ['natur', 'languag', 'process', 'amaz', 'process', 'text', 'efficientli']
```
