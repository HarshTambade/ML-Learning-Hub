# Tokenization - Breaking Text into Units

## Definition
Tokenization breaks text into tokens (words/sentences). Essential first step in NLP.

## Types

### Word Tokenization
- Splits text into words
- Example: "Python is great!" -> ["Python", "is", "great", "!"]

### Sentence Tokenization
- Splits text into sentences
- Handles: Dr., Mr., abbreviations

### Subword Tokenization (BPE)
- For transformer models
- Handles OOV (Out-of-Vocabulary) words
- Used in: GPT-2, BERT, RoBERTa

## Challenges
1. Contractions: "don't" -> ["do", "n't"] or ["donot"]?
2. Hyphenated: "state-of-the-art"
3. URLs/Emails: "user@example.com"
4. Numbers: Separate or attached?
5. Emojis: How to tokenize?

## Algorithms

| Algorithm | Use Case |
|---|---|
| BPE (Byte-Pair Encoding) | GPT-2, XLM |
| WordPiece | BERT, RoBERTa |
| SentencePiece | XLNet, mBART |
| Whitespace | Simple baseline |

## Code Example
```python
from nltk.tokenize import word_tokenize, sent_tokenize

text = "Dr. Smith is here. How are you?"
words = word_tokenize(text)
sents = sent_tokenize(text)
```

## Evaluation Metrics
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: Harmonic mean

## References
- Sennrich et al. (2016) - Byte Pair Encoding
- Devlin et al. (2019) - BERT Paper
- Radford et al. (2019) - GPT-2
