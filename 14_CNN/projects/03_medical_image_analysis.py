"""Medical Image Analysis using CNNs
Project: Detect abnormalities in medical images (X-rays, CT scans)
Use case: Classification and segmentation of medical imaging data
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class MedicalImageAnalyzer:
    """CNN-based medical image analysis system"""
    
    def __init__(self, model_type='resnet50', num_classes=2, pretrained=True):
        """Initialize medical image analyzer
        
        Args:
            model_type: Type of CNN architecture (resnet50, densenet121, etc.)
            num_classes: Number of output classes
            pretrained: Use pretrained ImageNet weights
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Load pretrained model
        if model_type == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            # Modify final layer for medical classification
            self.model.fc = nn.Linear(2048, num_classes)
        elif model_type == 'densenet121':
            self.model = models.densenet121(pretrained=pretrained)
            self.model.classifier = nn.Linear(1024, num_classes)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_image(self, image_path):
        """Load and preprocess medical image
        
        Args:
            image_path: Path to medical image file
            
        Returns:
            Preprocessed image tensor
        """
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        # Convert grayscale to RGB by repeating channels
        image_rgb = Image.new('RGB', image.size)
        image_rgb.paste(image)
        
        return self.transform(image_rgb).unsqueeze(0).to(self.device)
    
    def predict(self, image_path, return_grad_cam=False):
        """Predict abnormality in medical image
        
        Args:
            image_path: Path to medical image
            return_grad_cam: Return Grad-CAM visualization
            
        Returns:
            Prediction and confidence score
        """
        input_tensor = self.preprocess_image(image_path)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        result = {
            'predicted_class': predicted_class.item(),
            'confidence': float(confidence.item()),
            'class_probabilities': {
                'normal': float(probabilities[0, 0].item()),
                'abnormal': float(probabilities[0, 1].item()) if self.num_classes > 1 else 0
            }
        }
        
        return result
    
    def batch_predict(self, image_paths):
        """Predict on multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of predictions
        """
        results = []
        for img_path in image_paths:
            result = self.predict(img_path)
            results.append({'image': img_path, **result})
        
        return results
    
    def generate_report(self, predictions):
        """Generate analysis report from predictions
        
        Args:
            predictions: List of prediction results
            
        Returns:
            Summary report
        """
        total = len(predictions)
        abnormal_count = sum(1 for p in predictions if p['predicted_class'] == 1)
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        
        report = {
            'total_images': total,
            'abnormal_detected': abnormal_count,
            'normal_count': total - abnormal_count,
            'abnormality_rate': abnormal_count / total if total > 0 else 0,
            'average_confidence': avg_confidence,
            'details': predictions
        }
        
        return report

# Usage example
if __name__ == '__main__':
    # Initialize analyzer
    analyzer = MedicalImageAnalyzer(
        model_type='resnet50',
        num_classes=2,  # Normal vs Abnormal
        pretrained=True
    )
    
    # Single image prediction
    # result = analyzer.predict('xray_sample.jpg')
    # print(f"Prediction: {result}")
    
    # Batch processing
    # image_list = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    # predictions = analyzer.batch_predict(image_list)
    # report = analyzer.generate_report(predictions)
    # print(f"Report: {report}")
