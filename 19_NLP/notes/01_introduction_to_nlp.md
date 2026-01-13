# Introduction to Natural Language Processing (NLP)

## Definition & Overview
Natural Language Processing (NLP) is a subfield of Artificial Intelligence that focuses on the interaction between computers and human language. It enables machines to understand, interpret, analyze, and generate human language in a meaningful and contextually appropriate way.

## NLP Pipeline Architecture
```
┌─────────────┐
│  Raw Text   │
└──────┬──────┘
       │
       v
┌─────────────────┐
│ Tokenization    │ (Breaking into tokens)
└──────┬──────────┘
       │
       v
┌─────────────────┐
│  Normalization  │ (Lowercase, remove punct.)
└──────┬──────────┘
       │
       v
┌─────────────────────────┐
│ Feature Extraction      │ (TF-IDF, embeddings)
└──────┬──────────────────┘
       │
       v
┌─────────────────┐
│  ML/DL Model    │
└──────┬──────────┘
       │
       v
┌─────────────────┐
│  Output/Labels  │
└─────────────────┘
```

## Core NLP Tasks

### 1. Tokenization
- Breaks text into tokens (words/sentences)
- Types: Word tokenization, sentence tokenization
- Challenges: Contractions (don't), special characters

### 2. Part-of-Speech (POS) Tagging
- Assigns grammatical tags: NN(Noun), VB(Verb), JJ(Adjective)
- Example: "The cat sat on mat" → [(The, DT), (cat, NN), (sat, VBD)...]

### 3. Named Entity Recognition (NER)
- Identifies and classifies entities: PERSON, ORG, LOC, DATE
- Example: "Steve Jobs founded Apple Inc." → [PERSON: Steve Jobs, ORG: Apple Inc.]

### 4. Sentiment Analysis
- Determines emotional tone: Positive, Negative, Neutral
- Applications: Review analysis, social media monitoring

### 5. Machine Translation
- Converts text between languages
- Uses sequence-to-sequence models
- Challenges: Idioms, cultural context

### 6. Text Classification
- Categorizes documents into predefined classes
- Examples: Spam detection, topic classification

## NLP Applications

| Application | Use Case |
|---|---|
| Chatbots | Customer support automation |
| Machine Translation | Google Translate |
| Information Extraction | Resume parsing |
| Recommendation Systems | Personalized suggestions |
| Spam Detection | Email filtering |
| Speech Recognition | Voice assistants |
| Text Summarization | News summarization |

## Key Challenges in NLP

1. **Ambiguity**: Words with multiple meanings (bank - financial vs river)
2. **Context Dependency**: Same words, different meanings in different contexts
3. **Sarcasm & Idioms**: "That's just great!" (negative sarcasm)
4. **Domain-Specific Language**: Technical terms vary by domain
5. **Language Evolution**: New words, slang, abbreviations
6. **Negation Handling**: "not good" vs "good"
7. **Coreference Resolution**: Identifying pronoun references

## NLP vs NLU vs NLG

- **NLP**: General processing of natural language
- **NLU** (Natural Language Understanding): Comprehension aspect
- **NLG** (Natural Language Generation): Creating human-like text

## Python Libraries for NLP

```
NLTK      - Traditional NLP tasks
SpaCy     - Industrial NLP
Transformers - Modern deep learning models
Gensim    - Topic modeling, embeddings
TextBlob  - Simple NLP operations
```

## References
- Jurafsky, D., & Martin, J. H. (2009). Speech and Language Processing
- Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python
