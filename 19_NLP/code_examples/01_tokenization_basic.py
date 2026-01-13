"""Tokenization Basic - Introduction to NLP Tokenization

Tokenization is the process of breaking text into individual words or phrases (tokens).
It's a fundamental preprocessing step in NLP.
"""

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import TreebankWordTokenizer

# Download required NLTK data
nltk.download('punkt', quiet=True)

class BasicTokenizer:
    """A class to demonstrate basic tokenization techniques"""
    
    def __init__(self):
        self.text = """
        Natural Language Processing (NLP) is a fascinating field of AI.
        It helps machines understand human language. Tokenization is crucial!
        """
    
    def word_tokenization(self):
        """Demonstrate word tokenization"""
        print("\n=== Word Tokenization ===")
        tokens = word_tokenize(self.text)
        print(f"Tokens: {tokens[:15]}...")  # Show first 15 tokens
        print(f"Total tokens: {len(tokens)}")
        return tokens
    
    def sentence_tokenization(self):
        """Demonstrate sentence tokenization"""
        print("\n=== Sentence Tokenization ===")
        sentences = sent_tokenize(self.text)
        for i, sent in enumerate(sentences, 1):
            print(f"Sentence {i}: {sent.strip()}")
        return sentences
    
    def custom_tokenization(self):
        """Custom tokenization using TreebankWordTokenizer"""
        print("\n=== Custom Tokenization ===")
        tokenizer = TreebankWordTokenizer()
        tokens = tokenizer.tokenize(self.text.strip())
        print(f"Tokens: {tokens[:15]}...")
        return tokens
    
    def whitespace_tokenization(self):
        """Simple whitespace tokenization"""
        print("\n=== Whitespace Tokenization ===")
        tokens = self.text.split()
        print(f"Tokens: {tokens[:10]}...")
        return tokens

if __name__ == "__main__":
    tokenizer = BasicTokenizer()
    tokenizer.word_tokenization()
    tokenizer.sentence_tokenization()
    tokenizer.custom_tokenization()
    tokenizer.whitespace_tokenization()
    print("\n=== Tokenization Complete ===")
