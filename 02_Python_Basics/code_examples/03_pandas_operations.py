"""Pandas Operations Examples

Comprehensive examples of data manipulation and analysis using Pandas.
"""

import pandas as pd
import numpy as np

# ===== SERIES CREATION AND OPERATIONS =====
def series_operations():
    """Demonstrate Series creation and basic operations."""
    print("\n=== Series Operations ===")
    
    # Create Series
    s1 = pd.Series([1, 2, 3, 4, 5])
    s2 = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
    
    # Series from dictionary
    data_dict = {'Apple': 2.5, 'Banana': 1.2, 'Orange': 3.0}
    s3 = pd.Series(data_dict)
    
    print(f"\nSeries 1:\n{s1}")
    print(f"\nSeries 2:\n{s2}")
    print(f"\nSeries 3 (from dict):\n{s3}")
    
    # Series operations
    print(f"\nMean: {s1.mean()}")
    print(f"Std: {s1.std()}")
    print(f"Min: {s1.min()}, Max: {s1.max()}")
    print(f"Access element 'a': {s2['a']}")
    print(f"Slice [1:4]: {s1[1:4].tolist()}")

# ===== DATAFRAME CREATION =====
def dataframe_creation():
    """Create DataFrames from various sources."""
    print("\n=== DataFrame Creation ===")
    
    # From dictionary
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 28],
        'Salary': [50000, 60000, 75000, 55000],
        'City': ['NYC', 'LA', 'Chicago', 'Boston']
    }
    df = pd.DataFrame(data)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nFirst 3 rows:\n{df.head(3)}")
    print(f"\nColumn names: {df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    
    return df

# ===== DATA SELECTION AND INDEXING =====
def data_selection(df):
    """Demonstrate different ways to select data."""
    print("\n=== Data Selection ===")
    
    # Column selection
    print(f"\nAccess single column:\n{df['Name']}")
    print(f"\nAccess multiple columns:\n{df[['Name', 'Salary']]}")
    
    # Row selection
    print(f"\nAccess first row (iloc):\n{df.iloc[0]}")
    print(f"\nAccess row by condition (Age > 28):\n{df[df['Age'] > 28]}")
    
    # Boolean indexing
    high_salary = df[df['Salary'] > 55000]
    print(f"\nHigh earners (>55k):\n{high_salary}")

# ===== DATA CLEANING =====
def data_cleaning():
    """Demonstrate data cleaning operations."""
    print("\n=== Data Cleaning ===")
    
    # Create data with missing values
    data = {
        'ID': [1, 2, 3, 4, 5],
        'Value': [10, np.nan, 30, 40, np.nan],
        'Category': ['A', 'B', 'A', np.nan, 'B']
    }
    df = pd.DataFrame(data)
    
    print(f"\nOriginal DataFrame:\n{df}")
    print(f"\nMissing values:\n{df.isnull()}")
    print(f"\nMissing count per column:\n{df.isnull().sum()}")
    
    # Handle missing values
    df_filled = df.fillna(0)
    df_dropped = df.dropna()
    
    print(f"\nAfter fillna(0):\n{df_filled}")
    print(f"\nAfter dropna():\n{df_dropped}")
    
    # Remove duplicates
    df_dup = pd.DataFrame({
        'A': [1, 2, 2, 3],
        'B': ['x', 'y', 'y', 'z']
    })
    print(f"\nWith duplicates:\n{df_dup}")
    print(f"\nAfter drop_duplicates():\n{df_dup.drop_duplicates()}")

# ===== GROUPING AND AGGREGATION =====
def grouping_aggregation():
    """Demonstrate groupby and aggregation operations."""
    print("\n=== Grouping and Aggregation ===")
    
    data = {
        'Department': ['Sales', 'Sales', 'HR', 'HR', 'IT', 'IT'],
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
        'Salary': [50000, 55000, 45000, 48000, 60000, 65000]
    }
    df = pd.DataFrame(data)
    
    print(f"\nOriginal DataFrame:\n{df}")
    
    # Group by department and calculate mean salary
    avg_salary = df.groupby('Department')['Salary'].mean()
    print(f"\nAverage salary by department:\n{avg_salary}")
    
    # Multiple aggregations
    agg_result = df.groupby('Department').agg({
        'Salary': ['mean', 'sum', 'count']
    })
    print(f"\nMultiple aggregations:\n{agg_result}")
    
    # Custom aggregation
    agg_custom = df.groupby('Department').agg({
        'Salary': ['min', 'max']
    })
    print(f"\nMin and Max salary by department:\n{agg_custom}")

# ===== DATA TRANSFORMATION =====
def data_transformation():
    """Demonstrate data transformation operations."""
    print("\n=== Data Transformation ===")
    
    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Score': [85, 92, 78]
    })
    
    print(f"\nOriginal DataFrame:\n{df}")
    
    # Apply function
    df['Grade'] = df['Score'].apply(lambda x: 'A' if x >= 90 else 'B' if x >= 80 else 'C')
    print(f"\nAfter apply():\n{df}")
    
    # String operations
    df['Name_Upper'] = df['Name'].str.upper()
    df['Name_Length'] = df['Name'].str.len()
    print(f"\nAfter string operations:\n{df}")

# ===== MERGING AND JOINING =====
def merging_joining():
    """Demonstrate merge and join operations."""
    print("\n=== Merging and Joining ===")
    
    df1 = pd.DataFrame({
        'ID': [1, 2, 3],
        'Name': ['Alice', 'Bob', 'Charlie']
    })
    
    df2 = pd.DataFrame({
        'ID': [1, 2, 3],
        'Salary': [50000, 60000, 75000]
    })
    
    # Merge
    merged = pd.merge(df1, df2, on='ID', how='inner')
    print(f"\nMerged DataFrames:\n{merged}")
    
    # Concatenate
    df3 = pd.DataFrame({'ID': [4], 'Name': ['David']})
    concatenated = pd.concat([df1, df3], ignore_index=True)
    print(f"\nConcatenated DataFrames:\n{concatenated}")

# ===== PIVOT TABLES =====
def pivot_operations():
    """Demonstrate pivot table operations."""
    print("\n=== Pivot Tables ===")
    
    data = {
        'Month': ['Jan', 'Jan', 'Feb', 'Feb'],
        'Region': ['North', 'South', 'North', 'South'],
        'Sales': [100, 150, 120, 180]
    }
    df = pd.DataFrame(data)
    
    print(f"\nOriginal DataFrame:\n{df}")
    
    # Create pivot table
    pivot = pd.pivot_table(df, values='Sales', index='Month', columns='Region')
    print(f"\nPivot Table:\n{pivot}")

# ===== SORTING AND FILTERING =====
def sorting_filtering():
    """Demonstrate sorting and filtering operations."""
    print("\n=== Sorting and Filtering ===")
    
    df = pd.DataFrame({
        'Name': ['Charlie', 'Alice', 'Bob'],
        'Score': [78, 85, 92]
    })
    
    print(f"\nOriginal DataFrame:\n{df}")
    
    # Sort by column
    sorted_df = df.sort_values('Score', ascending=False)
    print(f"\nSorted by Score (descending):\n{sorted_df}")
    
    # Filter
    filtered = df[df['Score'] >= 85]
    print(f"\nFiltered (Score >= 85):\n{filtered}")

# ===== STATISTICAL OPERATIONS =====
def statistical_operations():
    """Demonstrate statistical operations."""
    print("\n=== Statistical Operations ===")
    
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50]
    })
    
    print(f"\nOriginal DataFrame:\n{df}")
    print(f"\nDescribe:\n{df.describe()}")
    print(f"\nCorrelation Matrix:\n{df.corr()}")
    print(f"\nMean:\n{df.mean()}")
    print(f"\nStandard Deviation:\n{df.std()}")

# ===== MAIN EXECUTION =====
if __name__ == '__main__':
    print("Pandas Operations Examples")
    print("=" * 50)
    
    series_operations()
    df = dataframe_creation()
    data_selection(df)
    data_cleaning()
    grouping_aggregation()
    data_transformation()
    merging_joining()
    pivot_operations()
    sorting_filtering()
    statistical_operations()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
