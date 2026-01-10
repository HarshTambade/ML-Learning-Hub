import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BERTClassifier(nn.Module):
    """BERT-based Text Classification Model"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

# Example usage
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BERTClassifier(num_classes=2)
    
    text = "This movie is great!"
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
    
    print(f"Input: {text}")
    print(f"Logits shape: {outputs.shape}")
    print(f"Predicted class: {torch.argmax(outputs, dim=1).item()}")
