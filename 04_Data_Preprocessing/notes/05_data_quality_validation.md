05_data_quality_validation.md  # Data Quality and Validation

## Introduction

Data quality validation ensures that data preprocessing is performed correctly and the data is suitable for analysis and model training.

## Key Data Quality Dimensions

### 1. Completeness

Extent to which required data is present

```python
# Check missing values
missing_percentage = df.isnull().sum() / len(df) * 100
print(missing_percentage[missing_percentage > 0])

# Accept threshold
if missing_percentage > 50:
    print("Warning: Column has >50% missing values")
```

### 2. Accuracy

Data conforms to the correct format and values

```python
# Validate data types
for col in df.columns:
    if df[col].dtype != expected_types[col]:
        print(f"Type mismatch in {col}")

# Check value ranges
if (df['age'] < 0).any() or (df['age'] > 150).any():
    print("Invalid age values")
```

### 3. Consistency

Data is consistent across the dataset

```python
# Check for duplicates
duplicates = df[df.duplicated(subset=['id'], keep=False)]
print(f"Duplicate records: {len(duplicates)}")

# Standardize categorical values
df['category'] = df['category'].str.lower().str.strip()
```

### 4. Uniformity

Data is stored in consistent units and formats

```python
# Validate date formats
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Check for consistent units
if any(df['weight'] > 500):  # Assume kg
    print("Potential unit conversion issue")
```

### 5. Validity

Data conforms to required formats

```python
import re

# Validate email format
email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
invalid_emails = ~df['email'].apply(lambda x: bool(re.match(email_pattern, x)))
print(f"Invalid emails: {invalid_emails.sum()}")
```

## Data Validation Techniques

### 1. Schema Validation

```python
from pandera import Column, DataFrameSchema

schema = DataFrameSchema({
    'age': Column(int, checks=[(lambda x: (x >= 0) & (x <= 150))]),
    'salary': Column(float, checks=[(lambda x: x > 0)]),
    'email': Column(str)
})

schema.validate(df)
```

### 2. Statistical Validation

```python
# Check distribution anomalies
from scipy import stats

for col in df.select_dtypes(include=[np.number]):
    skewness = stats.skew(df[col].dropna())
    if abs(skewness) > 2:
        print(f"{col} is highly skewed: {skewness}")
```

### 3. Cross-field Validation

```python
# Validate relationships
invalid = df[df['end_date'] < df['start_date']]
if len(invalid) > 0:
    print(f"Invalid date ranges: {len(invalid)}")
```

## Data Quality Metrics

```python
def assess_data_quality(df):
    """Comprehensive data quality assessment"""
    quality_report = {}
    
    # Completeness
    quality_report['Completeness'] = 1 - (df.isnull().sum().sum() / df.size)
    
    # Duplicate rate
    quality_report['Uniqueness'] = 1 - (df.duplicated().sum() / len(df))
    
    # Column count vs expected
    quality_report['Columns_Count'] = len(df.columns)
    
    # Row count
    quality_report['Row_Count'] = len(df)
    
    return quality_report
```

## Common Data Quality Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Missing Values | Data not collected | Imputation or deletion |
| Duplicates | Collection error | Remove or merge |
| Inconsistent | Different formats | Standardize |
| Outliers | Measurement error | Cap or remove |
| Type mismatch | Input error | Convert or validate |

## Best Practices

1. **Validate Early**: Check data immediately after loading
2. **Document Rules**: Define acceptance criteria
3. **Automate Checks**: Use validation frameworks
4. **Track Quality**: Monitor over time
5. **Quarantine**: Set aside invalid data
6. **Investigate**: Understand root causes

## Data Quality Checklist

- [ ] All required columns present
- [ ] No unexpected missing values
- [ ] Data types correct
- [ ] Value ranges valid
- [ ] No duplicate records
- [ ] Consistent formatting
- [ ] No obvious outliers
- [ ] Data relationships valid
- [ ] Sample data reviewed
- [ ] Quality metrics acceptable

## Summary

Robust data quality validation ensures reliable model performance and trustworthy analysis. Implement early, comprehensive validation to catch issues before modeling.
