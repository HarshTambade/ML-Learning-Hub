# Chapter 05: Linear Regression

## 📚 Overview

Linear Regression is a fundamental supervised learning algorithm for predicting continuous numerical values. It fits a straight line or hyperplane through data points to minimize prediction errors.

## 🎯 Topics Covered

1. **Fundamentals**: Simple & Multiple Linear Regression
2. **Mathematics**: Cost functions, Gradient Descent, Normal Equation
3. **Optimization**: Batch GD, SGD, Mini-batch GD
4. **Evaluation**: R², MSE, RMSE, MAE
5. **Advanced**: Polynomial, Ridge, Lasso, Elastic Net
6. **Applications**: House prices, Stock forecasts, Sales prediction

## 📁 Structure

```
05_Linear_Regression/
├── README.md (this file)
├── code_examples/
│   ├── 01_simple_linear_regression.py
│   ├── 02_multiple_linear_regression.py
│   ├── 03_polynomial_regression.py
│   ├── 04_regularization.py
│   └── 05_gradient_descent.py
├── exercises/README.md
├── notes/
│   ├── 01_mathematical_foundations.md
│   ├── 02_gradient_descent_explained.md
│   ├── 03_regularization_techniques.md
│   ├── 04_model_evaluation_metrics.md
│   └── 05_assumptions_diagnostics.md
├── projects/README.md
├── datasets/
│   ├── housing.csv
│   ├── auto_mpg.csv
│   └── insurance.csv
└── .gitkeep
```

## 🚀 Learning Path

**Beginner** → Read notes/01 → Run 01_simple_regression.py → Basic Exercises
**Intermediate** → Learn 02_multiple_regression.py → Study gradient descent → Medium Exercises  
**Advanced** → Study 03_polynomial, 04_regularization → Read all notes → Advanced Exercises & Projects

## 📖 Key Concepts

| Variant | Features | Use Case |
|---------|----------|----------|
| Simple | 1 | Single predictor |
| Multiple | 2+ | Multiple predictors |
| Polynomial | Transformed | Non-linear data |
| Ridge | 2+ + L2 | Multicollinearity |
| Lasso | 2+ + L1 | Feature selection |

| Metric | Range | Meaning |
|--------|-------|----------|
| R² | 0-1 | Variance explained |
| RMSE | 0-∞ | Error in original units |
| MAE | 0-∞ | Average absolute error |

## 🎓 Why It Matters

- **Foundation**: Essential for advanced ML algorithms
- **Interpretability**: Easy to explain coefficients
- **Efficiency**: Fast training & prediction
- **Baseline**: Perfect baseline for regression
- **Real-world**: Used in production systems everywhere

## 🚀 Quick Start Workflow

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Load data
data = pd.read_csv('data.csv')
X = data[['features']]
y = data['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)
print(f"R² = {r2_score(y_test, y_pred)}")
print(f"RMSE = {np.sqrt(mean_squared_error(y_test, y_pred))}")
```

## 🛠 Tools & Libraries

- **scikit-learn**: ML library
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **SciPy/StatsModels**: Statistics

## 💡 Tips for Success

1. **Visualize**: Plot data, residuals, relationships
2. **Check Assumptions**: Linearity, normality, homoscedasticity
3. **Scale Features**: Important for gradient descent
4. **Regularize**: Use Ridge/Lasso when needed
5. **Validate Properly**: Train/test split, cross-validation
6. **Interpret**: Understand what coefficients mean
7. **Analyze Residuals**: Look for patterns
8. **Document**: Track all preprocessing & choices

## 📋 Content

- **Code Examples**: 5 complete Python scripts with runnable code
- **Guides**: 5 detailed markdown files (100+ lines each)
- **Datasets**: 3 real-world datasets for practice
- **Exercises**: 6 problems with solutions (Basic → Advanced)
- **Projects**: 4 real-world application scenarios
- **Total**: 1000+ lines of code and documentation

## 💡 Learning Outcomes

✅ Understand mathematical foundations of linear regression
✅ Implement simple & multiple linear regression
✅ Implement gradient descent optimization
✅ Apply regularization techniques
✅ Evaluate using multiple metrics
✅ Check regression assumptions
✅ Handle polynomial and advanced techniques
✅ Build complete pipelines
✅ Solve real-world prediction problems

## 🔗 Next Chapters

- **Chapter 06**: Logistic Regression (Classification)
- **Chapter 07**: Decision Trees
- **Chapter 08**: Support Vector Machines

---

**Ready to master Linear Regression? 🚀 Start with the code examples!**
