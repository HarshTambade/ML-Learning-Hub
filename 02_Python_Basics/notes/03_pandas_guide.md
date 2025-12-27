# Pandas Guide for Data Manipulation

## Overview
Pandas is a powerful Python library for data manipulation and analysis. It provides data structures like DataFrames and Series that are essential for machine learning and data science workflows.

## Key Concepts

### 1. Series
A one-dimensional array-like object:
```python
import pandas as pd

# Create a Series
s = pd.Series([1, 2, 3, 4])
print(s)  # Index and values

# Series with custom index
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
print(s['a'])  # Access by index
```

### 2. DataFrame
A two-dimensional table-like structure:
```python
# Create DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
})

print(df.head())  # First 5 rows
print(df.shape)   # (rows, columns)
print(df.dtypes)  # Data types
```

### 3. Indexing and Selection
```python
# Accessing columns
df['name']  # Column as Series
df[['name', 'age']]  # Multiple columns as DataFrame

# Row access
df.loc[0]  # By label
df.iloc[0]  # By position

# Conditional selection
df[df['age'] > 25]
```

### 4. Data Cleaning
```python
# Handle missing values
df.isnull()  # Detect NaN
df.dropna()  # Remove NaN rows
df.fillna(0)  # Fill with value

# Duplicates
df.drop_duplicates()

# Data type conversion
df['age'] = df['age'].astype(int)
```

### 5. Grouping and Aggregation
```python
# Group by operation
df.groupby('city')['age'].mean()

# Multiple aggregations
df.groupby('city').agg({'age': 'mean', 'name': 'count'})
```

### 6. Merging and Joining
```python
# Merge DataFrames
pd.merge(df1, df2, on='key', how='inner')

# Concatenate
pd.concat([df1, df2])
```

### 7. Statistical Operations
```python
df.describe()  # Summary statistics
df.mean()      # Mean of each column
df.corr()      # Correlation matrix
df.std()       # Standard deviation
```

### 8. Reading and Writing Data
```python
# CSV
df = pd.read_csv('data.csv')
df.to_csv('output.csv', index=False)

# Excel
df = pd.read_excel('data.xlsx')
df.to_excel('output.xlsx')

# JSON
df = pd.read_json('data.json')
df.to_json('output.json')
```

### 9. Data Transformation
```python
# Apply custom function
df['new_col'] = df['age'].apply(lambda x: 'Adult' if x >= 18 else 'Minor')

# String operations
df['name'] = df['name'].str.upper()
df['name'].str.contains('a')

# Sorting
df.sort_values(by='age', ascending=False)
```

### 10. Pivot Tables
```python
# Create pivot table
pd.pivot_table(df, values='age', index='city', aggfunc='mean')
```

## Real-World Example
```python
# Load and clean data
df = pd.read_csv('sales_data.csv')

# Remove missing values
df = df.dropna()

# Create derived columns
df['total'] = df['quantity'] * df['price']

# Group by region
regional_sales = df.groupby('region')['total'].sum()

# Filter high-value transactions
high_value = df[df['total'] > 1000]

# Export results
high_value.to_csv('high_value_sales.csv')
```

## Performance Tips
- Use `df.loc[]` and `df.iloc[]` for faster indexing
- Avoid loops; vectorize operations when possible
- Use `copy()` when creating DataFrame copies
- Use `inplace=True` for memory efficiency
- Consider using `categorical` for repeated string values

## Common Issues
- **SettingWithCopyWarning**: Use `copy()` explicitly
- **Memory issues with large datasets**: Use `chunking` or `dask`
- **Performance**: Profile with `cProfile` for bottlenecks

## Resources
- Official Documentation: https://pandas.pydata.org/docs/
- Pandas Cheat Sheet: Commonly used functions reference
- Real datasets: Kaggle, UCI Machine Learning Repository
