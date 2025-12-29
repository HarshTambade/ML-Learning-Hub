"""Data Loading and Exploration - Chapter 04

Demonstrates techniques for loading data from various sources
and performing exploratory data analysis (EDA).
"""

import pandas as pd
import numpy as np
from scipy import stats


def load_csv_data(filepath):
    """Load data from CSV file."""
    return pd.read_csv(filepath)


def load_json_data(filepath):
    """Load data from JSON file."""
    return pd.read_json(filepath)


def load_excel_data(filepath, sheet_name=0):
    """Load data from Excel file."""
    return pd.read_excel(filepath, sheet_name=sheet_name)


def create_sample_data():
    """Create sample dataset for demonstration."""
    data = {
        'ID': range(1, 101),
        'Age': np.random.randint(20, 80, 100),
        'Income': np.random.normal(50000, 15000, 100),
        'Experience': np.random.randint(0, 40, 100),
        'Department': np.random.choice(['Sales', 'HR', 'IT', 'Finance'], 100),
        'Score': np.random.uniform(0, 100, 100)
    }
    return pd.DataFrame(data)


def explore_data(df):
    """Perform basic EDA on dataset."""
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    print("\n1. Dataset Shape:")
    print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    print("\n2. Column Information:")
    print(df.info())
    
    print("\n3. Statistical Summary:")
    print(df.describe())
    
    print("\n4. Data Types:")
    print(df.dtypes)


def check_missing_values(df):
    """Analyze missing values in dataset."""
    print("\n" + "="*70)
    print("MISSING VALUE ANALYSIS")
    print("="*70)
    
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    
    print("\nMissing Values:")
    for col in df.columns:
        if missing_count[col] > 0:
            print(f"  {col}: {missing_count[col]} ({missing_percent[col]:.2f}%)")
    
    if missing_count.sum() == 0:
        print("  No missing values found!")
    
    return missing_count


def detect_outliers(df, column, method='iqr'):
    """Detect outliers using IQR method."""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers, lower_bound, upper_bound


def analyze_duplicates(df):
    """Check for duplicate rows."""
    print("\n" + "="*70)
    print("DUPLICATE ANALYSIS")
    print("="*70)
    
    dup_count = df.duplicated().sum()
    print(f"\nTotal duplicate rows: {dup_count}")
    
    if dup_count > 0:
        print("\nDuplicate rows:")
        print(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)))
    
    return dup_count


def analyze_distributions(df):
    """Analyze data distributions."""
    print("\n" + "="*70)
    print("DISTRIBUTION ANALYSIS")
    print("="*70)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        print(f"\n{col}:")
        print(f"  Skewness: {stats.skew(df[col]):.4f}")
        print(f"  Kurtosis: {stats.kurtosis(df[col]):.4f}")


def analyze_correlations(df):
    """Analyze correlations between variables."""
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)
    
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    print("\nHighly Correlated Features (|r| > 0.7):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                print(f"  {corr_matrix.columns[i]} - {corr_matrix.columns[j]}: "
                      f"{corr_matrix.iloc[i, j]:.4f}")


if __name__ == '__main__':
    # Create sample data
    df = create_sample_data()
    print("Sample data created successfully!")
    
    # Perform EDA
    explore_data(df)
    
    # Check missing values
    check_missing_values(df)
    
    # Check for duplicates
    analyze_duplicates(df)
    
    # Analyze distributions
    analyze_distributions(df)
    
    # Analyze correlations
    analyze_correlations(df)
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
