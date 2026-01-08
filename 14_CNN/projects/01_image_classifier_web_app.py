"""Image Classifier Web Application
Project: Real-time CNN-based image classification using Flask and PyTorch
Use case: Deploy a trained CNN model as a web API for image classification
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

# Load pre-trained ResNet18 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18(pretrained=True)
model.eval()
model.to(device)

# ImageNet class labels
CLASS_LABELS = [
    'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead',
    'electric ray', 'stingray', 'cock', 'hen', 'ostrich'
]

# Preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.route('/classify', methods=['POST'])
def classify_image():
    """Endpoint to classify uploaded image"""
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Read image from request
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
        
        # Prepare response
        return jsonify({
            'predicted_class': CLASS_LABELS[predicted_idx.item()] if predicted_idx < len(CLASS_LABELS) else 'Unknown',
            'confidence': float(confidence),
            'all_predictions': {
                CLASS_LABELS[i]: float(probabilities[i])
                for i in range(min(5, len(CLASS_LABELS)))
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'Model is running', 'device': str(device)})

if __name__ == '__main__':
    app.run(debug=False, port=5000)

# Usage:
# python 01_image_classifier_web_app.py
# curl -X POST -F "image=@test_image.jpg" http://localhost:5000/classify
