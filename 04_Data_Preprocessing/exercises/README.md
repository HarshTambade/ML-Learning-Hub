# Chapter 04 Exercises: Data Preprocessing

## Overview
These exercises provide hands-on practice with data preprocessing techniques. Start with basic exercises and progress to more advanced challenges.

## Difficulty Levels
- **Basic** (⭐): Fundamental concepts
- **Intermediate** (⭐⭐): Combined techniques
- **Advanced** (⭐⭐⭐): Real-world scenarios

---

## BASIC EXERCISES

### Exercise 1: Data Loading (⭐)
**Problem**: Load different file formats and explore their structure.

**Tasks**:
1. Load a CSV file and display shape
2. Load a JSON file and convert to DataFrame
3. Load an Excel file with specific sheet
4. Display info() and describe() for each

**Solution Guide**:
```python
import pandas as pd

# CSV
df_csv = pd.read_csv('data.csv')
print(df_csv.info())

# JSON
df_json = pd.read_json('data.json')

# Excel
df_excel = pd.read_excel('data.xlsx', sheet_name=0)
```

---

### Exercise 2: Missing Value Detection (⭐)
**Problem**: Identify and analyze missing values in a dataset.

**Tasks**:
1. Count missing values per column
2. Calculate percentage of missing data
3. Visualize missing data patterns
4. Identify columns with >50% missing

**Dataset**: Create with `np.random.choice` and None values

---

### Exercise 3: Data Types & Conversion (⭐)
**Problem**: Identify and convert incorrect data types.

**Tasks**:
1. Identify current data types
2. Convert string dates to datetime
3. Convert object to numeric where appropriate
4. Handle conversion errors

---

## INTERMEDIATE EXERCISES

### Exercise 4: Missing Value Imputation (⭐⭐)
**Problem**: Handle missing values using multiple strategies.

**Tasks**:
1. Implement mean imputation for numeric columns
2. Implement mode imputation for categorical
3. Forward fill for time-series data
4. Compare imputation impacts on statistics

---

### Exercise 5: Outlier Detection & Treatment (⭐⭐)
**Problem**: Detect and handle outliers using IQR method.

**Tasks**:
1. Calculate IQR for numeric columns
2. Identify outlier values
3. Remove outliers
4. Cap outliers at bounds
5. Compare datasets before/after

---

### Exercise 6: Duplicate Handling (⭐⭐)
**Problem**: Identify and remove duplicate records.

**Tasks**:
1. Find complete duplicates
2. Find duplicates in specific columns
3. Remove duplicates keeping first/last
4. Analyze duplicate patterns

---

### Exercise 7: Feature Scaling (⭐⭐)
**Problem**: Scale features using different methods.

**Tasks**:
1. Implement StandardScaler
2. Implement MinMaxScaler
3. Compare before/after distributions
4. Apply scaling appropriately

---

## ADVANCED EXERCISES

### Exercise 8: Categorical Encoding (⭐⭐⭐)
**Problem**: Encode categorical variables appropriately.

**Tasks**:
1. One-hot encode nominal categories
2. Label encode ordinal categories
3. Target encode high-cardinality features
4. Analyze encoding impact

---

### Exercise 9: End-to-End Pipeline (⭐⭐⭐)
**Problem**: Build complete preprocessing pipeline.

**Tasks**:
1. Load and explore data
2. Handle missing values
3. Detect and treat outliers
4. Scale numerical features
5. Encode categorical features
6. Validate final dataset

---

### Exercise 10: Data Quality Assessment (⭐⭐⭐)
**Problem**: Evaluate data quality and create report.

**Tasks**:
1. Check for missing values
2. Detect outliers
3. Validate data types
4. Check for inconsistencies
5. Generate quality report

---

## Submission Requirements

For each exercise, submit:
- **Code file**: Well-commented Python script
- **Output**: Print statements showing results
- **Analysis**: Brief explanation of approach
- **Visualization**: Plots where applicable

## Evaluation Criteria

- **Correctness** (40%): Does solution work?
- **Efficiency** (20%): Is code optimized?
- **Documentation** (20%): Is code explained?
- **Visualization** (20%): Are results clear?

## Common Pitfalls

❌ Not handling edge cases
❌ Modifying original data
❌ Ignoring data type issues
❌ Not validating results
❌ Using same imputation for all columns

## Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Data Cleaning Best Practices](https://en.wikipedia.org/wiki/Data_cleansing)

## Tips for Success

1. **Always explore first**: Use `info()`, `describe()`, `head()`
2. **Validate changes**: Compare before/after statistics
3. **Document decisions**: Explain why you chose each approach
4. **Test edge cases**: Handle empty files, all nulls, etc.
5. **Keep original data**: Work on copies, preserve source
