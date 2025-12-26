# Chapter 01: Exercises & Practice Problems

## Conceptual Questions

### Easy

1. **What is Machine Learning?**
   - Define machine learning in your own words
   - How is it different from traditional programming?

2. **Types of Learning**
   - Name three types of machine learning
   - Give an example for each type

3. **Why is data quality important in ML?**
   - Explain with examples

### Medium

4. **Training vs Testing Data**
   - Why do we split data into training and testing sets?
   - What happens if we test on training data?

5. **Features and Labels**
   - What are features in a machine learning context?
   - What is a label/target?
   - Give examples from different domains

6. **Model Evaluation**
   - What is accuracy and what are its limitations?
   - Name other metrics used to evaluate models

### Hard

7. **Overfitting vs Underfitting**
   - Explain both concepts
   - How can you detect each?
   - How do you solve them?

8. **The ML Workflow**
   - Create a detailed flowchart of the ML development process
   - Identify decision points where you might go back to previous steps

## Coding Challenges

### Challenge 1: Load and Explore Data
```python
# Load the Iris dataset
# Print the shape of the data
# Print the first 5 samples
# Print unique classes
# Print feature names
```

**Objective**: Familiarize yourself with data exploration

### Challenge 2: Train Your First Model
```python
# Split the iris data 80-20
# Train a Decision Tree classifier
# Print training accuracy
# Print testing accuracy
# Compare the two accuracies - what does it tell you?
```

**Objective**: Understand training vs testing performance

### Challenge 3: Model Comparison
```python
# Train at least 3 different models on iris:
#   - Decision Tree
#   - Logistic Regression
#   - K-Nearest Neighbors
# Compare their accuracies
# Which performs best?
```

**Objective**: Understand that different algorithms have different performance

### Challenge 4: Hyperparameter Tuning
```python
# Train Decision Tree with different max_depth values
# Plot accuracy vs max_depth
# Find the depth that gives best test accuracy
```

**Objective**: Learn about hyperparameter importance

### Challenge 5: Custom Dataset
```python
# Find a small dataset online (e.g., housing, wine, cancer)
# Load it
# Train a classifier
# Evaluate its performance
# Document your findings
```

**Objective**: Apply ML pipeline to a new problem

## Discussion Questions

1. How would you approach solving a real-world problem using ML?
2. What are the ethical considerations in ML?
3. Why is reproducibility important in ML?
4. What role does randomness play in ML?

## Mini Projects

### Project: ML Comparison Report

Write a report comparing 3 different ML algorithms on the Iris dataset:

1. **Introduction**: Explain what you're comparing and why
2. **Data Description**: Describe the iris dataset
3. **Methodology**: Explain your experimental setup
4. **Results**: Compare accuracy, precision, recall
5. **Conclusion**: Which algorithm performs best and why?
6. **Future Work**: How could you improve this?

## Answers & Solutions

### Conceptual Answers

**Q1: What is ML?**
ML is a subset of AI that enables systems to learn from data and improve performance without being explicitly programmed.

**Q2: Types of Learning**
- Supervised: Labeled data, examples: classification, regression
- Unsupervised: Unlabeled data, examples: clustering, dimensionality reduction  
- Reinforcement: Learn from environment, examples: game AI, robotics

**Q3: Data Quality**
Poor quality data (missing values, duplicates, outliers, errors) leads to poor model performance. "Garbage in, garbage out."

### Coding Challenge Solutions

Solutions are provided in the `code_solutions/` folder.

## Resources

- Scikit-learn documentation
- Kaggle datasets
- Papers with Code
- Fast.ai course

## Submission Guidelines

1. Complete at least 3 challenges
2. Submit your code with comments
3. Include a brief explanation of your findings
4. Peer review another student's work

---

**Difficulty Levels**: ⭐ Easy | ⭐⭐ Medium | ⭐⭐⭐ Hard
