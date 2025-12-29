# Chapter 04 Exercises: Data Preprocessing

## Overview

These exercises provide hands-on practice with data preprocessing techniques. Start with basic exercises and progress to more advanced challenges.

## Difficulty Levels

- **Basic** (⭐): Fundamental concepts
- **Intermediate** (⭐⭐): Combined techniques
- **Advanced** (⭐⭐⭐): Real-world scenarios

---

## BASIC EXERCISES

### Exercise 1: Missing Value Imputation (⭐)

**Problem**: A dataset has missing values in the 'Age' and 'Salary' columns. Implement different imputation strategies.

**Tasks**:
1. Load the dataset and identify missing values
2. Apply mean imputation for 'Age'
3. Apply forward fill for 'Salary'
4. Compare the results

**Solution**:

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Load sample dataset
data = pd.DataFrame({
    'Age': [25, np.nan, 30, np.nan, 45],
    'Salary': [50000, 60000, np.nan, 80000, np.nan],
    'Department': ['HR', 'IT', 'IT', 'Sales', 'HR']
})

print("Original Data:")
print(data)
print(f"\nMissing values:\n{data.isnull().sum()}")

# Mean imputation for Age
imputer_age = SimpleImputer(strategy='mean')
data['Age'] = imputer_age.fit_transform(data[['Age']])

print(f"\nAfter Mean Imputation for Age:\n{data}")

# Forward fill for Salary
data['Salary'] = data['Salary'].fillna(method='ffill')

print(f"\nAfter Forward Fill for Salary:\n{data}")
```

---

### Exercise 2: Outlier Detection and Handling (⭐)

**Problem**: Identify and handle outliers in a numerical column using statistical methods.

**Tasks**:
1. Calculate mean and standard deviation
2. Identify values beyond 3 standard deviations
3. Remove or cap outliers
4. Visualize the before and after

**Solution**:

```python
import pandas as pd
import numpy as np

# Create sample data with outliers
data = pd.DataFrame({
    'Score': [10, 12, 15, 14, 13, 100, 11, 12, 150, 13]
})

print("Original Data:")
print(data)

# Calculate statistics
mean = data['Score'].mean()
std = data['Score'].std()
threshold = 3

print(f"\nMean: {mean}, Std: {std}")

# Method 1: Identify outliers
outliers_mask = np.abs((data['Score'] - mean) / std) > threshold
print(f"\nOutliers detected: {data[outliers_mask].index.tolist()}")

# Method 2: Remove outliers
data_cleaned = data[~outliers_mask].reset_index(drop=True)
print(f"\nData after removing outliers:\n{data_cleaned}")

# Method 3: Cap outliers
data_capped = data.copy()
lower_bound = mean - threshold * std
upper_bound = mean + threshold * std
data_capped['Score'] = data_capped['Score'].clip(lower_bound, upper_bound)
print(f"\nData after capping outliers:\n{data_capped}")
```

---

### Exercise 3: Feature Scaling (⭐)

**Problem**: Scale features to a standard range for machine learning.

**Tasks**:
1. Apply StandardScaler (z-score normalization)
2. Apply MinMaxScaler (range [0, 1])
3. Compare scaled outputs
4. Understand when to use each

**Solution**:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Create sample data
data = pd.DataFrame({
    'Income': [30000, 50000, 80000, 120000],
    'Age': [25, 35, 45, 55]
})

print("Original Data:")
print(data)

# StandardScaler (z-score normalization)
scaler_std = StandardScaler()
data_std = scaler_std.fit_transform(data)
data_std_df = pd.DataFrame(data_std, columns=['Income', 'Age'])

print("\nAfter StandardScaler:")
print(data_std_df)
print(f"Mean: {data_std_df.mean().round(4).tolist()}")
print(f"Std: {data_std_df.std().round(4).tolist()}")

# MinMaxScaler (range [0, 1])
scaler_minmax = MinMaxScaler()
data_minmax = scaler_minmax.fit_transform(data)
data_minmax_df = pd.DataFrame(data_minmax, columns=['Income', 'Age'])

print("\nAfter MinMaxScaler:")
print(data_minmax_df)
print(f"Min: {data_minmax_df.min().tolist()}")
print(f"Max: {data_minmax_df.max().tolist()}")
```

---

### Exercise 4: Categorical Encoding (⭐)

**Problem**: Convert categorical variables to numerical format.

**Tasks**:
1. Apply Label Encoding for ordinal variables
2. Apply One-Hot Encoding for nominal variables
3. Handle unknown categories
4. Compare encoding methods

**Solution**:

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Create sample data
data = pd.DataFrame({
    'Education': ['High School', 'Bachelor', 'Master', 'PhD', 'High School'],
    'Department': ['HR', 'IT', 'Sales', 'IT', 'HR']
})

print("Original Data:")
print(data)

# Method 1: Label Encoding (for ordinal variables)
le = LabelEncoder()
data['Education_Encoded'] = le.fit_transform(data['Education'])

print("\nAfter Label Encoding (Education):")
print(data[['Education', 'Education_Encoded']])
print(f"Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Method 2: One-Hot Encoding (for nominal variables)
data_onehot = pd.get_dummies(data, columns=['Department'], prefix='Department')

print("\nAfter One-Hot Encoding (Department):")
print(data_onehot)

# Method 3: Ordinal Encoding with custom order
from sklearn.preprocessing import OrdinalEncoder

data_copy = data.copy()
oe = OrdinalEncoder(categories=[['High School', 'Bachelor', 'Master', 'PhD']])
data_copy['Education_Ordinal'] = oe.fit_transform(data_copy[['Education']])

print("\nAfter Ordinal Encoding (Education):")
print(data_copy[['Education', 'Education_Ordinal']])
```

---

## INTERMEDIATE EXERCISES

### Exercise 5: Combined Preprocessing Pipeline (⭐⭐)

**Problem**: Build a complete preprocessing pipeline combining multiple techniques.

**Solution**:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

# Create mixed-type data
data = pd.DataFrame({
    'Age': [25, np.nan, 35, 45],
    'Income': [50000, 60000, np.nan, 100000],
    'Department': ['HR', 'IT', 'IT', 'Sales']
})

print("Original Data:")
print(data)
print(f"\nMissing values:\n{data.isnull().sum()}")

# Define preprocessing pipelines
numeric_features = ['Age', 'Income']
categorical_features = ['Department']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing
data_processed = preprocessor.fit_transform(data)

print("\nProcessed Data (array format):")
print(data_processed)

# Convert to DataFrame for better visualization
feature_names = numeric_features + [f'Department_{cat}' for cat in ['IT', 'Sales']]
data_processed_df = pd.DataFrame(data_processed, columns=feature_names)
print("\nProcessed Data (DataFrame format):")
print(data_processed_df)
```

---

## ADVANCED EXERCISES

### Exercise 6: Real-world Preprocessing with Validation (⭐⭐⭐)

**Problem**: Apply preprocessing with train-test split and proper validation.

**Solution**:

```python
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

# Create sample dataset
np.random.seed(42)
data = pd.DataFrame({
    'Age': np.random.randint(20, 60, 100),
    'Income': np.random.randint(30000, 150000, 100),
    'Department': np.random.choice(['HR', 'IT', 'Sales', 'Marketing'], 100),
    'YearsExperience': np.random.randint(0, 30, 100),
    'Target': np.random.choice([0, 1], 100)
})

# Introduce missing values
mask = np.random.rand(len(data)) < 0.1
data.loc[mask, 'Age'] = np.nan
mask = np.random.rand(len(data)) < 0.1
data.loc[mask, 'Income'] = np.nan

print(f"Original dataset shape: {data.shape}")
print(f"Missing values:\n{data.isnull().sum()}")

# Separate features and target
X = data.drop('Target', axis=1)
y = data['Target']

# Split data BEFORE preprocessing (important!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Define preprocessing (fit on training data only)
numeric_features = ['Age', 'Income', 'YearsExperience']
categorical_features = ['Department']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit preprocessor on training data
preprocessor.fit(X_train)

# Transform both train and test
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"\nProcessed train shape: {X_train_processed.shape}")
print(f"Processed test shape: {X_test_processed.shape}")
print(f"\nFirst row of processed training data:\n{X_train_processed[0]}")
```

---

## Common Pitfalls to Avoid

❌ Not handling edge cases
❌ Modifying original data in place
❌ Ignoring data type issues
❌ Using same imputation for all columns
❌ Forgetting to handle unknown categories
❌ Not validating results
❌ Using same scaling for all features

## Evaluation Criteria

- **Correctness** (40%): Does solution work?
- **Efficiency** (20%): Is code optimized?
- **Documentation** (20%): Is code explained?
- **Visualization** (20%): Are results clear?

## Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Data Cleaning Best Practices](https://en.wikipedia.org/wiki/Data_cleansing)
