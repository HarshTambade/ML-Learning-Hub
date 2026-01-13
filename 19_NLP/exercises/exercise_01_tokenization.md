# Exercise 1: Tokenization

## Problem
Write a program that tokenizes the following sentence and:
1. Counts the number of word tokens
2. Counts the number of sentence tokens
3. Identifies sentences with more than 5 words

## Sample Text
```
Natural Language Processing is fascinating. It helps machines understand text. Python is great for NLP!
```

## Solution
```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt', quiet=True)

text = "Natural Language Processing is fascinating. It helps machines understand text. Python is great for NLP!"

# Word tokenization
words = word_tokenize(text)
print(f"Word tokens: {len(words)}")
print(f"Words: {words}")

# Sentence tokenization
sentences = sent_tokenize(text)
print(f"\nSentence tokens: {len(sentences)}")

# Find sentences with more than 5 words
for sent in sentences:
    word_count = len(word_tokenize(sent))
    if word_count > 5:
        print(f"Sentence with {word_count} words: {sent}")
```

## Expected Output
```
Word tokens: 21
Sentence tokens: 3
Sentences with >5 words: Natural Language Processing is fascinating.
Python is great for NLP!
```

## Key Concepts
- Word tokenization breaks text into individual words/tokens
- Sentence tokenization identifies sentence boundaries
- NLTK provides ready-to-use tokenizers for NLP tasks
