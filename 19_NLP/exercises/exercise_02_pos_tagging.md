# Exercise 2: POS Tagging

## Problem
Implement a POS tagging system that:
1. Tags words with their grammatical roles
2. Extracts all nouns from a sentence
3. Counts different POS tag types

## Sample Text
```
Python programming is a fundamental skill for data scientists.
```

## Solution
```python
import nltk
from nltk import pos_tag, word_tokenize

nltk.download('averaged_perceptron_tagger', quiet=True)
text = "Python programming is a fundamental skill for data scientists."

tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)

print("POS Tags:", pos_tags)

# Extract nouns
nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
print(f"Nouns: {nouns}")

# Count tag types
from collections import Counter
tag_types = Counter(tag for word, tag in pos_tags)
print(f"Tag distribution: {dict(tag_types)}")
```

## Expected Output
```
POS Tags: [('Python', 'NNP'), ('programming', 'NN'), ...]
Nouns: ['Python', 'programming', 'skill', 'scientists']
Tag distribution: {'NN': 3, 'NNP': 1, ...}
```
