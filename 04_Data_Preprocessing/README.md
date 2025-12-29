# Chapter 04: Data Preprocessing

## 📚 Overview

Data Preprocessing is a critical phase in machine learning pipelines. It involves cleaning, transforming, and preparing raw data for model training. Poor data quality can severely impact model performance, making preprocessing essential.

## 🎯 Topics Covered

### 1. Data Loading
- Loading from CSV, JSON, Excel files
- Handling different file formats
- Data import best practices
- Database connections and APIs

### 2. Data Exploration & EDA
- Descriptive statistics
- Data shape and structure analysis
- Data type identification
- Basic data visualization

### 3. Data Cleaning
- **Missing Value Handling**
  - Detection and analysis
  - Deletion strategies
  - Imputation techniques (mean, median, mode, forward fill)
  - Advanced imputation (KNN, Iterative)
- **Duplicate Removal**
  - Identifying duplicates
  - Handling duplicate records

### 4. Data Transformation
- **Outlier Detection & Handling**
  - Z-score method
  - IQR method
  - Isolation Forest
  - Capping and removal strategies
- **Feature Scaling**
  - StandardScaler (Z-score normalization)
  - MinMaxScaler (0-1 range)
  - RobustScaler (IQR-based)
  - Normalizer (vector normalization)
- **Categorical Encoding**
  - Label Encoding
  - One-Hot Encoding
  - Ordinal Encoding
  - Target Encoding
  - Frequency Encoding

### 5. Data Validation
- Data quality metrics
- Schema validation
- Statistical validation
- Cross-field validation

### 6. Data Integration
- Merging multiple sources
- Joining datasets
- Aggregation strategies

## 📁 Folder Structure

```
04_Data_Preprocessing/
├── README.md                          # Chapter overview
├── code_examples/
│   ├── 01_data_loading_exploration.py    # Data loading and EDA
│   ├── 02_missing_value_handling.py      # Missing value techniques
│   ├── 03_outlier_detection.py           # Outlier detection methods
│   ├── 04_feature_scaling.py             # Feature scaling techniques
│   └── 05_categorical_encoding.py        # Categorical encoding methods
├── exercises/
│   └── README.md                         # Practice problems with solutions
├── notes/
│   ├── 01_data_loading_guide.md          # Detailed guide on data loading
│   ├── 02_missing_value_imputation.md    # Imputation techniques guide
│   ├── 03_feature_scaling_normalization.md # Scaling techniques guide
│   ├── 04_categorical_encoding.md        # Encoding techniques guide
│   └── 05_data_quality_validation.md     # Data quality validation guide
├── projects/
│   └── README.md                         # Real-world preprocessing projects
└── .gitkeep
```

## 🚀 Learning Path

### Beginner Level
1. Start with **code_examples/01_data_loading_exploration.py** - Learn how to load and explore data
2. Read **notes/01_data_loading_guide.md** - Understand different data sources
3. Complete **exercises** (Basic level) - Practice with simple datasets

### Intermediate Level
1. Study **code_examples/02_missing_value_handling.py** - Learn missing value techniques
2. Read **notes/02_missing_value_imputation.md** - Understand imputation methods
3. Study **code_examples/03_outlier_detection.py** - Learn outlier detection
4. Practice **exercises** (Intermediate level) - Combined preprocessing techniques

### Advanced Level
1. Study **code_examples/04_feature_scaling.py** - Scaling techniques
2. Study **code_examples/05_categorical_encoding.py** - Encoding methods
3. Read **notes/03_feature_scaling_normalization.md** - Advanced scaling
4. Read **notes/04_categorical_encoding.md** - Advanced encoding
5. Read **notes/05_data_quality_validation.md** - Quality validation
6. Complete **exercises** (Advanced level) - Real-world scenarios
7. Work on **projects** - End-to-end preprocessing pipelines

## 📖 Key Concepts

### Missing Values
- **MCAR** (Missing Completely At Random): No pattern
- **MAR** (Missing At Random): Depends on observed variables
- **MNAR** (Missing Not At Random): Depends on unobserved variables

### Feature Scaling Methods
| Method | Range | Use Case |
|--------|-------|----------|
| StandardScaler | (-∞, +∞) | Normal distribution |
| MinMaxScaler | [0, 1] | Bounded values needed |
| RobustScaler | Varies | Outliers present |
| Normalizer | [0, 1] | Vector normalization |

### Categorical Encoding Methods
| Method | Best For | Pros |
|--------|----------|------|
| Label Encoding | Ordinal data | Memory efficient |
| One-Hot Encoding | Nominal data | No ordinal assumption |
| Ordinal Encoding | Ordered categories | Preserves order |
| Target Encoding | High cardinality | Reduces dimensions |
| Frequency Encoding | Categorical data | Simple and fast |

## 🎓 Why Preprocessing Matters

1. **Model Performance**: 80% of ML success depends on data quality
2. **Training Speed**: Clean data trains faster
3. **Convergence**: Scaled features help gradient descent converge better
4. **Interpretability**: Transformed data is easier to understand
5. **Robustness**: Proper preprocessing makes models more robust

## 🔄 Common Preprocessing Workflow

```python
1. Load data → pd.read_csv()
2. Explore data → df.info(), df.describe()
3. Handle missing values → imputation/deletion
4. Detect outliers → Z-score, IQR, Isolation Forest
5. Handle outliers → removal, capping, transformation
6. Scale features → StandardScaler, MinMaxScaler
7. Encode categorical → Label, One-Hot, Ordinal
8. Validate quality → schema validation, statistics
9. Split data → train_test_split()
10. Ready for modeling!
```

## 📚 Code Examples

### Quick Start: Complete Preprocessing Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# 1. Load data
df = pd.read_csv('data.csv')

# 2. Handle missing values
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(
    imputer.fit_transform(df.select_dtypes(include=[np.number])),
    columns=df.select_dtypes(include=[np.number]).columns
)

# 3. Scale features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_imputed)

# 4. Encode categorical
df_encoded = pd.get_dummies(df, columns=['category_column'])

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    df_encoded.drop('target', axis=1),
    df_encoded['target'],
    test_size=0.2
)

print("Preprocessing complete!")
```

## 🛠 Tools & Libraries

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Preprocessing, imputation, scaling, encoding
- **scipy**: Statistical functions
- **matplotlib/seaborn**: Visualization

## 📚 Useful Links

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Data Cleaning Best Practices](https://en.wikipedia.org/wiki/Data_cleansing)

## 💡 Quick Start

### To Run Code Examples
```bash
# Install dependencies
pip install pandas numpy scikit-learn scipy matplotlib seaborn

# Run any code example
python code_examples/01_data_loading_exploration.py
python code_examples/02_missing_value_handling.py
python code_examples/03_outlier_detection.py
python code_examples/04_feature_scaling.py
python code_examples/05_categorical_encoding.py
```

### To Read Guides
- Check the `notes/` folder for detailed markdown guides on each topic
- Each guide includes theory, examples, and best practices

### To Practice
- Work through exercises in `exercises/README.md`
- Start with Basic exercises, progress to Advanced
- Each exercise includes complete solutions

### To Build Projects
- Real-world projects in `projects/` folder
- Apply all preprocessing techniques together
- Get hands-on experience with complete pipelines

## 📊 Learning Outcomes

After completing this chapter, you will be able to:

✅ Load and explore data from multiple sources
✅ Identify and handle missing values appropriately
✅ Detect and manage outliers using statistical methods
✅ Apply appropriate feature scaling techniques
✅ Encode categorical variables correctly
✅ Validate data quality using metrics
✅ Build complete preprocessing pipelines
✅ Handle real-world data challenges
✅ Optimize data for machine learning models
✅ Document preprocessing steps and decisions

## 📋 Exercise & Project Directory

### Exercises
- **Basic (⭐)**: Introduction to each technique
- **Intermediate (⭐⭐)**: Combining multiple techniques
- **Advanced (⭐⭐⭐)**: Real-world scenarios

Each exercise includes:
- Problem statement
- Tasks to complete
- Complete solution with explanation

### Projects
- Real-world datasets
- End-to-end preprocessing pipelines
- Performance metrics and validation

## 💡 Tips for Success

1. **Understand Your Data**: Always explore before preprocessing
2. **Don't Lose Information**: Document what you change and why
3. **Fit on Training Data**: Always fit scalers/encoders on train data only
4. **Validate Results**: Check distributions before and after
5. **Document Decisions**: Keep track of preprocessing choices
6. **Test Gradually**: Apply techniques one at a time
7. **Monitor Performance**: Compare model performance with and without preprocessing
8. **Handle Test Data**: Apply same preprocessing to test data

## 🔗 Next Steps

After mastering data preprocessing, move to:
- **Chapter 05**: Linear Regression (Predictive Modeling)
- **Chapter 03**: Statistics & Probability (Theory Foundations)

## 📝 Resources

- Code Examples: 5 Python scripts with runnable code
- Guides: 5 detailed markdown files (150+ lines each)
- Exercises: 6 problems with complete solutions
- Projects: Real-world preprocessing scenarios
- Total Content: 1000+ lines of code and documentation

---

**Happy Learning! 🚀**

Master data preprocessing and unlock the potential of your ML models!
