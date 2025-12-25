#!/usr/bin/env python3
"""
Setup script to create ML-Learning-Hub directory structure.
This script creates all chapter directories with necessary subdirectories.
"""

import os
from pathlib import Path

# Define all chapters with their structure
CHAPTERS = {
    "01_Introduction_to_ML": "Introduction to Machine Learning",
    "02_Python_Basics": "Python for ML & Data Analysis",
    "03_Statistics_Probability": "Statistics & Probability",
    "04_Data_Preprocessing": "Data Preprocessing & Feature Engineering",
    "05_Linear_Regression": "Linear Regression",
    "06_Logistic_Regression": "Logistic Regression & Classification",
    "07_Decision_Trees": "Decision Trees & Ensemble Methods",
    "08_SVM": "Support Vector Machines (SVM)",
    "09_KNN": "K-Nearest Neighbors (KNN)",
    "10_Clustering": "Clustering Algorithms",
    "11_Dimensionality_Reduction": "Dimensionality Reduction",
    "12_Anomaly_Detection": "Anomaly Detection",
    "13_Neural_Networks": "Neural Networks Fundamentals",
    "14_CNN": "Convolutional Neural Networks (CNN)",
    "15_RNN": "Recurrent Neural Networks (RNN)",
    "16_Transformers": "Transformers & Attention Mechanisms",
    "17_Reinforcement_Learning": "Reinforcement Learning",
    "18_Generative_Models": "Generative Models",
    "19_NLP": "Natural Language Processing (NLP)",
    "20_Time_Series": "Time Series Analysis",
    "21_Model_Evaluation": "Model Evaluation & Selection",
    "22_ML_Pipelines": "ML Pipelines & Automation",
    "23_Model_Deployment": "Model Deployment",
    "24_MLOps": "MLOps & Model Monitoring",
}

# Subdirectories within each chapter
SUBDIRES = ["notes", "code", "notebooks", "datasets", "exercises", "projects"]

def create_structure():
    """Create the complete ML-Learning-Hub directory structure."""
    
    for chapter_dir, chapter_name in CHAPTERS.items():
        chapter_path = Path(chapter_dir)
        
        # Create chapter directory
        chapter_path.mkdir(exist_ok=True)
        print(f"Created directory: {chapter_dir}")
        
        # Create subdirectories
        for subdir in SUBDIRES:
            subdir_path = chapter_path / subdir
            subdir_path.mkdir(exist_ok=True)
            print(f"  Created subdirectory: {subdir}")
        
        # Create a basic README.md for each chapter
        readme_path = chapter_path / "README.md"
        if not readme_path.exists():
            with open(readme_path, 'w') as f:
                f.write(f"# {chapter_name}\n\n")
                f.write(f"## Overview\n")
                f.write(f"This chapter covers {chapter_name.lower()}.\n\n")
                f.write(f"## Contents\n")
                f.write(f"- **notes/**: Theoretical explanations and concepts\n")
                f.write(f"- **code/**: Implementation examples in Python\n")
                f.write(f"- **notebooks/**: Jupyter notebooks with experiments\n")
                f.write(f"- **datasets/**: Sample datasets for practice\n")
                f.write(f"- **exercises/**: Practice problems with solutions\n")
                f.write(f"- **projects/**: Real-world mini-projects\n\n")
                f.write(f"## Learning Objectives\n")
                f.write(f"- Objective 1\n")
                f.write(f"- Objective 2\n")
                f.write(f"- Objective 3\n\n")
                f.write(f"## Key Topics\n")
                f.write(f"- Topic 1\n")
                f.write(f"- Topic 2\n")
                f.write(f"- Topic 3\n\n")
            print(f"  Created README.md")

if __name__ == "__main__":
    print("Setting up ML-Learning-Hub structure...")
    create_structure()
    print("\n✓ All directories created successfully!")
    print(f"Total chapters: {len(CHAPTERS)}")
    print(f"Subdirectories per chapter: {len(SUBDIRES)}")
