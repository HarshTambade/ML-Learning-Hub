# Exercise 3: Named Entity Recognition

## Problem
Build an NER system that:
1. Identifies named entities
2. Classifies entities by type (PERSON, LOCATION, ORGANIZATION)
3. Extracts organization names

## Sample Text
```
Apple Inc. was founded by Steve Jobs in California. Bill Gates leads Microsoft.
```

## Solution
```python
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

nltk.download('maxent_ne_chunker', quiet=True)

text = "Apple Inc. was founded by Steve Jobs in California. Bill Gates leads Microsoft."
tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)
ner_tags = ne_chunk(pos_tags)

print("Named Entities:")
for subtree in ner_tags:
    if isinstance(subtree, Tree):
        print(f"{subtree.label()}: {' '.join(w for w,t in subtree.leaves())}")
```

## Expected Output
```
Named Entities:
PERSON: Steve Jobs
ORGANIZATION: Apple Inc.
LOCATION: California
PERSON: Bill Gates
ORGANIZATION: Microsoft
```
