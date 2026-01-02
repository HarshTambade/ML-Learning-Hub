# SVM Exercises

## Overview
This folder contains exercises to practice and master Support Vector Machines concepts and implementation.

## Exercise Structure

Each exercise includes:
1. **Problem Statement**: Clear description of what to implement
2. **Datasets**: Links or instructions for data
3. **Starter Code**: Boilerplate to get started
4. **Solution**: Complete working solution
5. **Expected Output**: Expected results

## Exercise List

### Exercise 1: Basic SVM Classification
**File**: `01_basic_classification.ipynb`

**Objective**: Implement SVM on the Iris dataset

**Tasks**:
- Load the Iris dataset
- Split into train/test sets
- Train SVM with linear kernel
- Evaluate accuracy and confusion matrix
- Plot decision boundaries

**Difficulty**: Beginner
**Time**: 30 minutes

### Exercise 2: Kernel Comparison
**File**: `02_kernel_comparison.ipynb`

**Objective**: Compare different kernels (linear, RBF, polynomial)

**Tasks**:
- Create a non-linear dataset
- Train SVM with different kernels
- Compare accuracy scores
- Visualize decision boundaries for each kernel
- Analyze pros and cons

**Difficulty**: Beginner-Intermediate
**Time**: 45 minutes

### Exercise 3: Hyperparameter Tuning
**File**: `03_hyperparameter_tuning.ipynb`

**Objective**: Find optimal C and gamma parameters

**Tasks**:
- Use GridSearchCV for parameter tuning
- Plot performance vs. parameters
- Identify overfitting/underfitting
- Implement early stopping
- Analyze sensitivity to hyperparameters

**Difficulty**: Intermediate
**Time**: 60 minutes

### Exercise 4: Multi-class Classification
**File**: `04_multiclass_svm.ipynb`

**Objective**: Implement one-vs-rest multi-class SVM

**Tasks**:
- Load MNIST or fashion dataset
- Implement one-vs-rest strategy
- Compare with one-vs-one
- Evaluate on 3+ classes
- Analyze per-class performance

**Difficulty**: Intermediate
**Time**: 60 minutes

### Exercise 5: Real-World Classification
**File**: `05_real_world_project.ipynb`

**Objective**: Apply SVM to a real dataset

**Tasks**:
- Choose a classification dataset (UCI ML, Kaggle)
- Perform exploratory data analysis
- Handle missing values and outliers
- Feature engineering and scaling
- Full SVM pipeline with validation
- Interpret results

**Difficulty**: Intermediate-Advanced
**Time**: 90 minutes

## Quick Start

### Environment Setup
```bash
# Install required packages
pip install scikit-learn numpy pandas matplotlib jupyter
```

### Running Exercises
```bash
# Start Jupyter notebook
jupyter notebook

# Open desired exercise notebook
# Work through the problems
# Compare with solution
```

## Learning Outcomes

After completing these exercises, you will:
1. ✓ Understand SVM fundamentals
2. ✓ Know when to use different kernels
3. ✓ Master hyperparameter tuning
4. ✓ Handle multi-class problems
5. ✓ Apply SVM to real datasets

## Dataset Resources

- **Iris**: Included in scikit-learn
- **MNIST**: Available from scikit-learn or Keras
- **Fashion MNIST**: From Keras
- **Breast Cancer**: From scikit-learn
- **Kaggle Datasets**: www.kaggle.com/datasets
- **UCI ML Repository**: www.archive.ics.uci.edu/ml

## Tips for Success

1. **Start Simple**: Begin with linear kernel before trying RBF
2. **Scale Features**: Always use StandardScaler
3. **Cross-Validate**: Use k-fold CV for evaluation
4. **Visualize**: Plot decision boundaries when possible
5. **Iterate**: Test multiple approaches
6. **Document**: Write comments explaining your code
7. **Compare**: Check solutions to learn best practices

## Common Mistakes to Avoid

- ❌ Forgetting to scale features
- ❌ Not splitting train/test data
- ❌ Tuning hyperparameters on test set
- ❌ Using too large C (overfitting)
- ❌ Not checking class balance
- ❌ Ignoring cross-validation
- ❌ Using inappropriate kernel for data type

## Additional Resources

### Books
- "Kernel Methods for Pattern Analysis" by Schölkopf & Smola
- "Pattern Recognition and Machine Learning" by Bishop

### Websites
- scikit-learn SVM documentation
- StatQuest with Josh Starmer (SVM videos)
- Coursera ML course

### Practice
- Kaggle competitions
- LeetCode ML problems
- GitHub ML projects

## Troubleshooting

**Issue**: SVM takes too long to train
**Solution**: Use linear kernel or reduce data size

**Issue**: Low accuracy
**Solution**: Check if features are scaled, try different kernels

**Issue**: Overfitting
**Solution**: Reduce C parameter or use larger margin

## Next Steps

After mastering SVM:
1. Learn ensemble methods (Random Forest, Gradient Boosting)
2. Explore neural networks for comparison
3. Study transfer learning
4. Apply to real production systems

---

**Happy Learning!** 🚀📚🙋
