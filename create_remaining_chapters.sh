#!/bin/bash

# Script to create remaining ML-Learning-Hub chapters (08-24)
# Run this script locally to create all remaining directory structures

echo "Creating remaining ML Learning Hub chapters (08-24)..."

chapters=(
  "08_SVM"
  "09_KNN"
  "10_Clustering"
  "11_Dimensionality_Reduction"
  "12_Anomaly_Detection"
  "13_Neural_Networks"
  "14_CNN"
  "15_RNN"
  "16_Transformers"
  "17_Reinforcement_Learning"
  "18_Generative_Models"
  "19_NLP"
  "20_Time_Series"
  "21_Model_Evaluation"
  "22_ML_Pipelines"
  "23_Model_Deployment"
  "24_MLOps"
)

for chapter in "${chapters[@]}"; do
  mkdir -p "${chapter}"
  touch "${chapter}/.gitkeep"
  echo "Created: ${chapter}"
done

echo "\nAll chapters created successfully!"
echo "Next: Add content to each chapter's subdirectories (notes/, code/, notebooks/, etc.)"
