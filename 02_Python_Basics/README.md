# Chapter 02: Python Basics for Machine Learning

## 📚 Learning Objectives

By the end of this chapter, you will be able to:

- **Python Fundamentals** - Variables, data types, control flow, functions
- **NumPy Essentials** - Arrays, operations, broadcasting, indexing
- **Pandas for Data** - DataFrames, Series, data manipulation, cleaning
- **Data Visualization** - Matplotlib, Seaborn, plotting techniques
- **Working with Data** - Loading, saving, exploring datasets

## 🎯 Key Concepts

### Python Programming Basics

#### 1. **Variables & Data Types**
- Integers, floats, strings, booleans
- Dynamic typing in Python
- Type checking and conversion

#### 2. **Control Flow**
- If-else statements
- For and while loops
- List comprehensions
- Exception handling

#### 3. **Functions**
- Defining and calling functions
- Parameters and return values
- Default arguments and *args, **kwargs
- Lambda functions

#### 4. **Data Structures**
- Lists, tuples, dictionaries, sets
- Operations on collections
- Comprehensions for efficient code

### NumPy for Numerical Computing

```python
import numpy as np

# Creating arrays
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Array operations
arr * 2      # Element-wise multiplication
arr + arr    # Element-wise addition
np.dot(a, b) # Matrix multiplication

# Useful functions
np.mean(), np.std(), np.sum()
np.reshape(), np.flatten()
np.where(), np.unique()
```

### Pandas for Data Manipulation

```python
import pandas as pd

# Creating DataFrames
df = pd.read_csv('data.csv')
df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})

# Data exploration
df.head(), df.info(), df.describe()
df.shape, df.columns, df.dtypes

# Data manipulation
df['new_column'] = df['A'] * 2
df.loc[0], df.iloc[0]
df.groupby('category').mean()
df.fillna(0), df.dropna()
```

### Visualization with Matplotlib & Seaborn

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Basic plots
plt.plot(x, y)
plt.scatter(x, y)
plt.hist(data, bins=30)
plt.bar(categories, values)

# Seaborn plots
sns.boxplot(data)
sns.heatmap(correlation_matrix)
sns.scatterplot(data=df, x='x', y='y', hue='category')
```

## 📁 Folder Structure

```
02_Python_Basics/
├── README.md                 # This file
├── notes/                    # Theory and concepts
│   ├── 01_python_basics.md
│   ├── 02_numpy_tutorial.md
│   ├── 03_pandas_guide.md
│   └── 04_visualization.md
├── code_examples/            # Practical examples
│   ├── 01_python_fundamentals.py
│   ├── 02_numpy_operations.py
│   ├── 03_pandas_dataframes.py
│   └── 04_data_visualization.py
├── projects/                 # Mini-projects
│   └── data_analysis_project.md
└── exercises/                # Practice problems
    └── README.md
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher installed
- Basic understanding of programming concepts
- Jupyter Notebook installed

### Installation
```bash
# Install required libraries
pip install numpy pandas matplotlib seaborn jupyter

# Or use requirements.txt
pip install -r requirements.txt
```

## 📖 Topics Covered

### 1. Python Programming Fundamentals
   - Variables, data types, and operators
   - Control structures (if/else, loops)
   - Functions and modules
   - File I/O operations
   - Working with strings and regular expressions

### 2. NumPy: Numerical Computing
   - Creating and manipulating arrays
   - Mathematical operations
   - Broadcasting and vectorization
   - Linear algebra operations
   - Random number generation

### 3. Pandas: Data Manipulation
   - Series and DataFrame objects
   - Reading and writing data
   - Data cleaning and preprocessing
   - Grouping and aggregation
   - Merging and joining datasets

### 4. Data Visualization
   - Matplotlib basics
   - Seaborn for statistical plots
   - Creating publication-quality figures
   - Interactive visualizations

## 💡 Key Takeaways

- Python is essential for ML development
- NumPy provides efficient numerical operations
- Pandas simplifies data manipulation
- Visualization helps understand data patterns
- Practice is crucial for mastery

## 🔗 Resources

- [Python Official Documentation](https://docs.python.org/3/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Tutorial](https://matplotlib.org/stable/tutorials/)
- [DataCamp Python Courses](https://www.datacamp.com/)

## ✅ Chapter Checklist

- [ ] Master Python fundamentals
- [ ] Understand NumPy array operations
- [ ] Learn DataFrame manipulation
- [ ] Create effective visualizations
- [ ] Complete all code examples
- [ ] Solve practice exercises
- [ ] Finish mini-project

---

**Next Chapter:** Chapter 03 - Statistics & Probability

*Last Updated: December 27, 2025*
