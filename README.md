# ML-Learning-Hub

A comprehensive Machine Learning learning resource with chapter-wise code examples, explanations, and hands-on projects. This repository contains everything you need to master Machine Learning from fundamentals to advanced concepts.

## 📚 Course Structure

### Part 1: Fundamentals
- **Chapter 1:** [Introduction to Machine Learning](./01_Introduction_to_ML/)
  - What is ML?
  - Types of ML (Supervised, Unsupervised, Reinforcement)
  - ML Workflow
  - Applications

- **Chapter 2:** [Python for ML & Data Analysis](./02_Python_Basics/)
  - NumPy Fundamentals
  - Pandas Data Manipulation
  - Matplotlib & Visualization
  - Data Loading & Preprocessing

- **Chapter 3:** [Statistics & Probability](./03_Statistics_Probability/)
  - Descriptive Statistics
  - Probability Distributions
  - Hypothesis Testing
  - Correlation & Causation

- **Chapter 4:** [Data Preprocessing & Feature Engineering](./04_Data_Preprocessing/)
  - Data Cleaning
  - Handling Missing Values
  - Feature Scaling
  - Feature Selection
  - Encoding Categorical Variables

### Part 2: Supervised Learning
- **Chapter 5:** [Linear Regression](./05_Linear_Regression/)
  - Simple Linear Regression
  - Multiple Linear Regression
  - Polynomial Regression
  - Regularization (Ridge, Lasso, ElasticNet)

- **Chapter 6:** [Logistic Regression & Classification](./06_Logistic_Regression/)
  - Binary Classification
  - Multi-class Classification
  - Classification Metrics
  - ROC & AUC

- **Chapter 7:** [Decision Trees & Ensemble Methods](./07_Decision_Trees/)
  - Decision Tree Basics
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - AdaBoost

- **Chapter 8:** [Support Vector Machines (SVM)](./08_SVM/)
  - Linear SVM
  - Non-linear Kernels
  - Multi-class SVM
  - Hyperparameter Tuning

- **Chapter 9:** [K-Nearest Neighbors (KNN)](./09_KNN/)
  - KNN Algorithm
  - Distance Metrics
  - Choosing K
  - KNN Variations

### Part 3: Unsupervised Learning
- **Chapter 10:** [Clustering Algorithms](./10_Clustering/)
  - K-Means Clustering
  - Hierarchical Clustering
  - DBSCAN
  - Gaussian Mixture Models

- **Chapter 11:** [Dimensionality Reduction](./11_Dimensionality_Reduction/)
  - Principal Component Analysis (PCA)
  - t-SNE
  - UMAP
  - Feature Extraction

- **Chapter 12:** [Anomaly Detection](./12_Anomaly_Detection/)
  - Statistical Methods
  - Isolation Forest
  - One-Class SVM
  - Autoencoder-based Methods

### Part 4: Deep Learning
- **Chapter 13:** [Neural Networks Fundamentals](./13_Neural_Networks/)
  - Perceptron
  - Feedforward Neural Networks
  - Activation Functions
  - Backpropagation
  - Gradient Descent Optimization

- **Chapter 14:** [Convolutional Neural Networks (CNN)](./14_CNN/)
  - CNN Architecture
  - Convolution & Pooling
  - Image Classification
  - Transfer Learning

- **Chapter 15:** [Recurrent Neural Networks (RNN)](./15_RNN/)
  - RNN Basics
  - LSTM & GRU
  - Sequence Modeling
  - Time Series Prediction

- **Chapter 16:** [Transformers & Attention Mechanisms](./16_Transformers/)
  - Attention Mechanism
  - Transformer Architecture
  - BERT & GPT
  - Fine-tuning Pretrained Models

### Part 5: Advanced Topics
- **Chapter 17:** [Reinforcement Learning](./17_Reinforcement_Learning/)
  - Markov Decision Process
  - Q-Learning
  - Policy Gradient Methods
  - Deep Q-Networks

- **Chapter 18:** [Generative Models](./18_Generative_Models/)
  - Variational Autoencoders (VAE)
  - Generative Adversarial Networks (GAN)
  - Diffusion Models

- **Chapter 19:** [Natural Language Processing (NLP)](./19_NLP/)
  - Text Preprocessing
  - Word Embeddings
  - Sentiment Analysis
  - Machine Translation

- **Chapter 20:** [Time Series Analysis](./20_Time_Series/)
  - ARIMA
  - Prophet
  - LSTM for Time Series
  - Forecasting Techniques

### Part 6: Production & Deployment
- **Chapter 21:** [Model Evaluation & Selection](./21_Model_Evaluation/)
  - Cross-Validation
  - Hyperparameter Tuning
  - Model Comparison
  - Statistical Tests

- **Chapter 22:** [ML Pipelines & Automation](./22_ML_Pipelines/)
  - Scikit-learn Pipelines
  - Feature Engineering Pipelines
  - Automated ML (AutoML)

- **Chapter 23:** [Model Deployment](./23_Model_Deployment/)
  - Model Serialization
  - REST APIs
  - Docker Containerization
  - Cloud Deployment (AWS, GCP, Azure)

- **Chapter 24:** [MLOps & Model Monitoring](./24_MLOps/)
  - Model Versioning
  - Data Drift Detection
  - Model Monitoring
  - Continuous Integration/Deployment

## 🛠️ Prerequisites

- Python 3.8+
- Basic understanding of programming
- Linear algebra and calculus basics
- Git & GitHub

## 📦 Dependencies

```bash
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
torch>=1.10.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/HarshTambade/ML-Learning-Hub.git
cd ML-Learning-Hub
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Navigate to any chapter folder and explore the notebooks:
```bash
cd 01_Introduction_to_ML
jupyter notebook
```

## 📖 How to Use This Repository

1. **For Beginners:** Start with Chapter 1 and follow the sequence
2. **Theory & Concepts:** Each chapter has detailed notes in `.md` files
3. **Code Examples:** Check `.ipynb` (Jupyter Notebooks) for hands-on examples
4. **Projects:** Each chapter includes mini-projects to apply concepts
5. **Exercises:** Practice problems with solutions at the end of each chapter

## 📋 Repository Structure

Each chapter folder contains:
```
├── README.md           # Chapter overview & learning objectives
├── notes/             # Detailed theoretical explanations
├── code/              # Python implementation examples
├── notebooks/         # Jupyter notebooks with experiments
├── datasets/          # Sample datasets for practice
├── exercises/         # Practice problems
└── projects/          # Real-world projects
```

## 🎯 Learning Path

**Weeks 1-4:** Fundamentals (Chapters 1-4)
**Weeks 5-8:** Supervised Learning (Chapters 5-9)
**Weeks 9-10:** Unsupervised Learning (Chapters 10-12)
**Weeks 11-16:** Deep Learning (Chapters 13-16)
**Weeks 17-20:** Advanced Topics (Chapters 17-20)
**Weeks 21-24:** Production & MLOps (Chapters 21-24)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Submit a pull request

Please ensure all code follows PEP 8 standards and includes proper documentation.

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 💬 Feedback & Questions

Feel free to open issues for:
- Questions about any topic
- Suggestions for improvements
- Bug reports
- Topic requests

## 🔗 Resources

- [Fast.ai](https://www.fast.ai/)
- [Coursera ML Course](https://www.coursera.org/learn/machine-learning)
- [Deep Learning Book](http://www.deeplearningbook.org/)
- [Papers with Code](https://paperswithcode.com/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)

## 📧 Contact

Harsh Tambade
- GitHub: [@HarshTambade](https://github.com/HarshTambade)
- Email: [Your Email]

---

⭐ If you find this helpful, please consider giving it a star!

Happy Learning! 🚀
