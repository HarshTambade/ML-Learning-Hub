# Word Embeddings - Dense Vector Representations

## What are Word Embeddings?
Word embeddings convert words into continuous dense vectors that capture semantic meaning. Words with similar meanings have similar vectors.

## Types of Embeddings

### 1. Static Embeddings
- Fixed vectors for each word
- Techniques: Word2Vec, GloVe, FastText
- Limitation: Polysemy (same word, different meanings)

### 2. Contextual Embeddings
- Dynamic vectors based on context
- Techniques: ELMo, BERT, GPT
- Advantage: Handles polysemy

## Popular Embedding Models

| Model | Year | Characteristics |
|---|---|---|
| Word2Vec | 2013 | Fast, efficient, 300D |
| GloVe | 2014 | Combines local & global info |
| FastText | 2016 | Handles OOV, character ngrams |
| ELMo | 2018 | Bidirectional, context-aware |
| BERT | 2018 | Transformer-based, 768D |
| GPT-3 | 2020 | Large language model |

## Word2Vec Architecture

### CBOW (Continuous Bag of Words)
- Predicts word from context
- Input: surrounding words
- Output: target word

### Skip-gram
- Predicts context from word
- Input: target word  
- Output: surrounding words
- Better for rare words

## GloVe (Global Vectors)
- Combines matrix factorization & local context
- Uses word co-occurrence matrix
- Formula: log(P_ij) = w_i * w_j + b_i + b_j

## FastText
- Extension of Word2Vec
- Uses character n-grams
- Handles misspellings & rare words

## Applications
- Text similarity
- Document classification
- Machine translation
- Question answering

## References
- Mikolov et al. (2013) - Word2Vec
- Pennington et al. (2014) - GloVe
- Bojanowski et al. (2017) - FastText
