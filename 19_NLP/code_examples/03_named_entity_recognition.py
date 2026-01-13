"""Named Entity Recognition (NER) - Identify and classify named entities

NER identifies and classifies named entities (persons, locations, organizations, etc.)
It's crucial for information extraction and knowledge graph creation.
"""

import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

class NERExtractor:
    """Named Entity Recognition demonstration"""
    
    def __init__(self):
        self.sentences = [
            "Steve Jobs founded Apple in Cupertino, California.",
            "Elon Musk is the CEO of Tesla Inc.",
            "Facebook, now Meta, was founded by Mark Zuckerberg."
        ]
    
    def basic_ner(self):
        """Perform basic NER"""
        print("\n=== Named Entity Recognition ===")
        for sent in self.sentences:
            tokens = word_tokenize(sent)
            pos_tags = pos_tag(tokens)
            ner_tags = ne_chunk(pos_tags)
            print(f"Sentence: {sent}")
            ner_tags.pprint()
    
    def extract_entities(self):
        """Extract named entities"""
        print("\n=== Extracted Entities ===")
        for sent in self.sentences:
            tokens = word_tokenize(sent)
            pos_tags = pos_tag(tokens)
            ner_tags = ne_chunk(pos_tags)
            
            for subtree in ner_tags:
                if isinstance(subtree, Tree):
                    entity_type = subtree.label()
                    entity_name = " ".join([word for word, tag in subtree.leaves()])
                    print(f"{entity_type}: {entity_name}")
    
    def entity_classification(self):
        """Classify entity types"""
        print("\n=== Entity Classification ===")
        text = "Barack Obama was President of USA. Microsoft is in Redmond."
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        ner_tags = ne_chunk(pos_tags)
        print(ner_tags)

if __name__ == "__main__":
    extractor = NERExtractor()
    extractor.basic_ner()
    extractor.extract_entities()
    extractor.entity_classification()
