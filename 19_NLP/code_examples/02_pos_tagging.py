"""POS (Part-of-Speech) Tagging - Identify word types in sentences

POS tagging assigns grammatical tags to words (noun, verb, adjective, etc.)
It's essential for parsing, named entity recognition, and other NLP tasks.
"""

import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet

nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class POSTagger:
    """Demonstrate Part-of-Speech tagging"""
    
    def __init__(self):
        self.sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is revolutionizing artificial intelligence.",
            "She sells seashells by the seashore."
        ]
    
    def basic_pos_tagging(self):
        """Basic POS tagging using NLTK"""
        print("\n=== Basic POS Tagging ===")
        for sent in self.sentences:
            tokens = word_tokenize(sent)
            pos_tags = pos_tag(tokens)
            print(f"Sentence: {sent}")
            for word, tag in pos_tags:
                print(f"  {word:15} -> {tag}")
            print()
    
    def analyze_pos_tags(self):
        """Analyze POS tags in detail"""
        print("\n=== POS Tag Analysis ===")
        text = "The beautiful flowers bloom beautifully in spring."
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        
        tag_counts = {}
        for word, tag in pos_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        print(f"Total tokens: {len(pos_tags)}")
        print("\nTag distribution:")
        for tag, count in sorted(tag_counts.items()):
            print(f"  {tag}: {count}")
    
    def extract_by_pos(self):
        """Extract words by their POS tags"""
        print("\n=== Extract by POS Tag ===")
        text = "The quick brown fox jumps over lazy sleeping dogs."
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        
        nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
        verbs = [word for word, pos in pos_tags if pos.startswith('VB')]
        adjectives = [word for word, pos in pos_tags if pos.startswith('JJ')]
        
        print(f"Nouns: {nouns}")
        print(f"Verbs: {verbs}")
        print(f"Adjectives: {adjectives}")

if __name__ == "__main__":
    tagger = POSTagger()
    tagger.basic_pos_tagging()
    tagger.analyze_pos_tags()
    tagger.extract_by_pos()
