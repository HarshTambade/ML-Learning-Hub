03_outlier_detection.py  """
Chapter 04: Outlier Detection and Handling
Demonstrates techniques for identifying and handling outliers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def create_sample_data_with_outliers():
    """Create sample data with outliers"""
    np.random.seed(42)
    data = {
        'Score': [10, 12, 14, 15, 13, 11, 12, 100, 150, 13, 12, 14],
        'Value': [100, 105, 98, 102, 99, 101, 100, 500, 450, 103, 97, 99]
    }
    return pd.DataFrame(data)

def method_1_zscore():
    """Method 1: Z-score based outlier detection"""
    print("\n=== METHOD 1: Z-SCORE OUTLIER DETECTION ===")
    df = create_sample_data_with_outliers()
    print("Original data:")
    print(df['Score'].values)
    
    # Calculate Z-scores
    z_scores = np.abs(stats.zscore(df['Score']))
    threshold = 3
    
    outliers = df[z_scores > threshold]
    print(f"\nOutliers (Z-score > {threshold}):")
    print(outliers)
    print(f"Indices: {outliers.index.tolist()}")
    
    # Remove outliers
    df_cleaned = df[z_scores <= threshold]
    print(f"\nData after removing outliers: {len(df_cleaned)} rows remaining")
    print(df_cleaned['Score'].values)

def method_2_iqr():
    """Method 2: Interquartile Range (IQR) method"""
    print("\n=== METHOD 2: IQR-BASED OUTLIER DETECTION ===")
    df = create_sample_data_with_outliers()
    print("Original data:")
    print(df['Score'].values)
    
    Q1 = df['Score'].quantile(0.25)
    Q3 = df['Score'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"\nQ1: {Q1}, Q3: {Q3}, IQR: {IQR}")
    print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
    
    outliers = df[(df['Score'] < lower_bound) | (df['Score'] > upper_bound)]
    print(f"\nOutliers found: {len(outliers)}")
    print(outliers)
    
    # Remove outliers
    df_cleaned = df[(df['Score'] >= lower_bound) & (df['Score'] <= upper_bound)]
    print(f"\nData after removing outliers: {len(df_cleaned)} rows remaining")
    print(df_cleaned['Score'].values)

def method_3_isolation_forest():
    """Method 3: Isolation Forest Algorithm"""
    print("\n=== METHOD 3: ISOLATION FOREST ===")
    from sklearn.ensemble import IsolationForest
    
    df = create_sample_data_with_outliers()
    print("Original data:")
    print(df['Score'].values)
    
    # Train Isolation Forest
    iso_forest = IsolationForest(contamination=0.2, random_state=42)
    outlier_labels = iso_forest.fit_predict(df[['Score']])
    
    outliers = df[outlier_labels == -1]
    print(f"\nOutliers detected by Isolation Forest:")
    print(outliers)
    
    # Remove outliers
    df_cleaned = df[outlier_labels == 1]
    print(f"\nData after removing outliers: {len(df_cleaned)} rows remaining")

def method_4_capping():
    """Method 4: Outlier Capping"""
    print("\n=== METHOD 4: OUTLIER CAPPING ===")
    df = create_sample_data_with_outliers()
    print("Original data:")
    print(df['Score'].values)
    
    mean = df['Score'].mean()
    std = df['Score'].std()
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    
    print(f"\nMean: {mean:.2f}, Std: {std:.2f}")
    print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    df_capped = df.copy()
    df_capped['Score'] = df_capped['Score'].clip(lower_bound, upper_bound)
    
    print("\nData after capping:")
    print(df_capped['Score'].values)

def method_5_log_transformation():
    """Method 5: Log Transformation for skewed data"""
    print("\n=== METHOD 5: LOG TRANSFORMATION ===")
    df = create_sample_data_with_outliers()
    print("Original data (skewed):")
    print(df['Value'].values)
    
    # Apply log transformation
    df_log = df.copy()
    df_log['Value_log'] = np.log1p(df_log['Value'])
    
    print("\nAfter log transformation:")
    print(df_log['Value_log'].values)
    
    print(f"\nOriginal variance: {df['Value'].var():.2f}")
    print(f"Log-transformed variance: {df_log['Value_log'].var():.2f}")

def outlier_statistics():
    """Print detailed outlier statistics"""
    print("\n=== OUTLIER STATISTICS ===")
    df = create_sample_data_with_outliers()
    
    print("Score column statistics:")
    print(f"Count: {df['Score'].count()}")
    print(f"Mean: {df['Score'].mean():.2f}")
    print(f"Median: {df['Score'].median():.2f}")
    print(f"Std: {df['Score'].std():.2f}")
    print(f"Min: {df['Score'].min()}")
    print(f"Max: {df['Score'].max()}")
    print(f"Q1: {df['Score'].quantile(0.25):.2f}")
    print(f"Q3: {df['Score'].quantile(0.75):.2f}")
    print(f"Skewness: {df['Score'].skew():.2f}")
    print(f"Kurtosis: {df['Score'].kurtosis():.2f}")

def main():
    """Run all outlier detection examples"""
    print("="*60)
    print("OUTLIER DETECTION AND HANDLING TECHNIQUES")
    print("="*60)
    
    outlier_statistics()
    method_1_zscore()
    method_2_iqr()
    method_3_isolation_forest()
    method_4_capping()
    method_5_log_transformation()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
