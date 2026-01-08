"""Real-time Object Detection using YOLOv8
Project: Detect objects in video streams using CNN-based YOLO model
Use case: Real-time detection of multiple objects with bounding boxes
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_name='yolov8n.pt', conf_threshold=0.5):
        """Initialize YOLOv8 object detector
        
        Args:
            model_name: Pre-trained YOLO model name
            conf_threshold: Confidence threshold for detections
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_name)
        self.model.to(self.device)
        self.conf_threshold = conf_threshold
    
    def detect_frame(self, frame):
        """Detect objects in a single frame
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Annotated frame with bounding boxes
        """
        # Run inference
        results = self.model(frame, conf=self.conf_threshold)
        
        # Extract detections
        annotated_frame = results[0].plot()
        
        return annotated_frame, results[0]
    
    def process_video(self, video_path, output_path=None, show=True):
        """Process video file for object detection
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video
            show: Whether to display results
        """
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            annotated_frame, detections = self.detect_frame(frame)
            
            # Display results
            if show:
                cv2.imshow('Object Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write to output video
            if output_path:
                out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        if output_path:
            out.release()
        if show:
            cv2.destroyAllWindows()
        
        print(f"Processing complete! Total frames: {frame_count}")
    
    def process_webcam(self, duration_seconds=30):
        """Real-time detection from webcam
        
        Args:
            duration_seconds: Duration to capture from webcam
        """
        cap = cv2.VideoCapture(0)
        start_time = cv2.getTickCount()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            annotated_frame, detections = self.detect_frame(frame)
            
            # Add FPS counter
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - start_time)
            cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Real-time Object Detection', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Usage example
if __name__ == '__main__':
    # Initialize detector
    detector = ObjectDetector(model_name='yolov8n.pt')
    
    # Option 1: Process video file
    # detector.process_video('video.mp4', output_path='output.mp4')
    
    # Option 2: Real-time webcam detection
    detector.process_webcam(duration_seconds=30)
