import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class GPT2TextGenerator:
    """GPT-2 based text generation"""
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def generate(self, prompt, max_length=100, temperature=0.7):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=1,
                do_sample=True
            )
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    generator = GPT2TextGenerator()
    prompt = "The future of AI is"
    result = generator.generate(prompt, max_length=50)
    print(f"Prompt: {prompt}")
    print(f"Generated: {result}")
