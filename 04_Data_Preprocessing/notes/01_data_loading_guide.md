# Data Loading Guide

## Introduction

Data loading is the first critical step in any data preprocessing pipeline. It involves reading data from various sources and loading it into memory for processing.

## Common Data Sources

### 1. CSV Files

CSV (Comma-Separated Values) is one of the most common data formats.

```python
import pandas as pd

# Basic CSV loading
df = pd.read_csv('data.csv')

# With parameters
df = pd.read_csv('data.csv', 
                  sep=',',           # delimiter
                  header=0,          # row number for column names
                  encoding='utf-8',  # file encoding
                  nrows=1000)        # limit rows
```

### 2. Excel Files

```python
df = pd.read_excel('data.xlsx', sheet_name=0)
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
```

### 3. JSON Files

```python
df = pd.read_json('data.json')
df = pd.read_json('data.json', orient='records')
```

### 4. Database Connections

```python
import sqlite3

conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM table_name', conn)
```

### 5. APIs

```python
import requests

response = requests.get('https://api.example.com/data')
data = response.json()
df = pd.DataFrame(data)
```

## Data Loading Best Practices

1. **Specify Data Types**: Reduce memory usage and improve performance
   ```python
   dtypes = {'age': int, 'salary': float, 'name': str}
   df = pd.read_csv('data.csv', dtype=dtypes)
   ```

2. **Handle Missing Values**: Specify how to handle missing data
   ```python
   df = pd.read_csv('data.csv', na_values=['NA', 'N/A', ''])
   ```

3. **Verify Data Integrity**: Always check the loaded data
   ```python
   print(df.shape)         # dimensions
   print(df.info())        # data types and nulls
   print(df.describe())    # summary statistics
   print(df.head())        # first few rows
   ```

4. **Handle Large Files**: Load in chunks
   ```python
   for chunk in pd.read_csv('large_file.csv', chunksize=10000):
       process_chunk(chunk)
   ```

## Common Issues and Solutions

### Issue 1: Encoding Problems

**Solution**: Specify the correct encoding
```python
df = pd.read_csv('data.csv', encoding='latin-1')
```

### Issue 2: Wrong Delimiters

**Solution**: Specify the correct delimiter
```python
df = pd.read_csv('data.txt', sep='\t')  # for tab-separated
```

### Issue 3: Memory Issues with Large Files

**Solution**: Load specific columns only
```python
df = pd.read_csv('large_file.csv', usecols=['col1', 'col2'])
```

## Performance Tips

- Use `dtype` parameter to specify data types
- Load only necessary columns with `usecols`
- Use `chunksize` for very large files
- Consider using `dask` for out-of-memory datasets

## Summary

Proper data loading is crucial for ensuring data quality and processing efficiency. Always verify your data after loading to catch issues early in the pipeline.
