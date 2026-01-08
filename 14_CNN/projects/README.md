# CNN Real-World Projects

This folder contains practical, production-ready projects demonstrating CNN applications in real-world scenarios.

## Projects Overview

### 1. Image Classifier Web App (`01_image_classifier_web_app.py`)
**Description:** A Flask-based web application for real-time image classification using a pre-trained ResNet18 model.

**Key Features:**
- RESTful API endpoints for image classification
- Support for pre-trained ImageNet models
- Health check endpoint for monitoring
- Error handling and validation
- Inference optimization with no_grad context

**Dependencies:**
```bash
pip install torch torchvision flask pillow
```

**Usage:**
```bash
python 01_image_classifier_web_app.py
curl -X POST -F "image=@test_image.jpg" http://localhost:5000/classify
```

**Output:**
```json
{
  "predicted_class": "golden_retriever",
  "confidence": 0.95,
  "all_predictions": {
    "golden_retriever": 0.95,
    "labrador": 0.04
  }
}
```

---

### 2. Real-time Object Detection (`02_object_detection_real_time.py`)
**Description:** Real-time object detection using YOLOv8, supporting video files and webcam feeds.

**Key Features:**
- YOLOv8 model support (nano, small, medium, large variants)
- Video file processing with output saving
- Live webcam detection with FPS counter
- Configurable confidence threshold
- GPU acceleration support

**Dependencies:**
```bash
pip install torch ultralytics opencv-python numpy
```

**Usage:**
```python
from 02_object_detection_real_time import ObjectDetector

# Initialize detector
detector = ObjectDetector(model_name='yolov8n.pt')

# Option 1: Process video file
detector.process_video('video.mp4', output_path='output.mp4')

# Option 2: Real-time webcam detection
detector.process_webcam(duration_seconds=30)
```

**Features:**
- Multiple object detection
- Bounding box annotations
- Confidence scoring
- FPS monitoring for performance

---

### 3. Medical Image Analysis (`03_medical_image_analysis.py`)
**Description:** CNN-based medical image analysis for detecting abnormalities in X-rays, CT scans, and other medical imaging data.

**Key Features:**
- Pre-trained ResNet50 and DenseNet121 support
- Grayscale medical image preprocessing
- Batch prediction capabilities
- Comprehensive reporting functionality
- Classification and confidence scoring

**Dependencies:**
```bash
pip install torch torchvision pillow numpy matplotlib
```

**Usage:**
```python
from 03_medical_image_analysis import MedicalImageAnalyzer

# Initialize analyzer
analyzer = MedicalImageAnalyzer(
    model_type='resnet50',
    num_classes=2,  # Normal vs Abnormal
    pretrained=True
)

# Single image prediction
result = analyzer.predict('xray_sample.jpg')
print(result)

# Batch processing
predictions = analyzer.batch_predict(['img1.jpg', 'img2.jpg'])
report = analyzer.generate_report(predictions)
print(report)
```

**Output:**
```json
{
  "predicted_class": 1,
  "confidence": 0.92,
  "class_probabilities": {
    "normal": 0.08,
    "abnormal": 0.92
  }
}
```

---

## Project Structure

```
projects/
├── README.md                          # This file
├── 01_image_classifier_web_app.py    # Web-based image classification
├── 02_object_detection_real_time.py  # Real-time object detection
└── 03_medical_image_analysis.py      # Medical image classification
```

## Requirements

**Common Requirements:**
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (optional, for GPU acceleration)

**All Dependencies:**
```bash
pip install torch torchvision flask pillow ultralytics opencv-python numpy matplotlib
```

## Performance Considerations

1. **Image Classifier:** ~100ms inference time on CPU, ~20ms on GPU
2. **Object Detection:** Real-time processing at 30 FPS with YOLOv8n on GPU
3. **Medical Image Analysis:** Batch processing for multiple images recommended

## Deployment Tips

1. **API Deployment:** Use Gunicorn or uWSGI for the Flask app
2. **Model Optimization:** Export to ONNX for faster inference
3. **Containerization:** Use Docker for easy deployment
4. **Monitoring:** Implement logging and metrics collection

## Further Reading

- [PyTorch Documentation](https://pytorch.org/docs/)
- [YOLOv8 Guide](https://docs.ultralytics.com/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Medical Image Analysis Research Papers](https://arxiv.org/)

## Contributing

Feel free to extend these projects with:
- Additional model architectures
- Custom training scripts
- Enhanced preprocessing pipelines
- Performance optimizations
