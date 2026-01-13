# Named Entity Recognition (NER)

## Definition
NER identifies and classifies named entities in text. Entities are real-world objects like persons, locations, organizations, dates.

## Entity Types

| Entity Type | Examples |
|---|---|
| PERSON | John, Steve Jobs, Einstein |
| ORG | Google, Microsoft, NASA |
| LOC | USA, California, London |
| DATE | Monday, 2023, March 15 |
| TIME | 3:00 PM, noon |
| MONEY | $100, €50 |
| PERCENT | 10%, 5.5% |
| PRODUCT | iPhone, Windows 10 |
| EVENT | Olympics, WWII |

## NER Approaches

### 1. Rule-Based
- Uses hand-crafted rules & patterns
- Regex-based extraction
- Limitation: Low coverage, high maintenance

### 2. Statistical (CRF)
- Conditional Random Fields
- Sequence labeling
- Better generalization than rules

### 3. Deep Learning
- BiLSTM + CRF
- BERT-based NER
- State-of-the-art performance

## NER Pipeline
```
Raw Text → Tokenization → Feature Extraction → Sequence Labeling → Entity Extraction
```

## Popular NER Models

| Model | Framework | Languages |
|---|---|---|
| spaCy | spaCy | 17 languages |
| StanfordNER | Java | Multi-language |
| BERT-NER | PyTorch | 104 languages |
| Flair | PyTorch | Multi-language |

## Applications
- Resume parsing
- Information extraction
- Question answering systems
- Knowledge graph construction
- Content recommendation

## Evaluation Metrics
- Precision, Recall, F1-Score
- Seqeval (for sequence labeling)

## References
- Finkel et al. (2005) - CRF
- Lample et al. (2016) - BiLSTM-CRF for NER
