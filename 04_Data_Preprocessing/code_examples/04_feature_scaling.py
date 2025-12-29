04_feature_scaling.py  """
Chapter 04: Feature Scaling and Normalization
Demonstrates various feature scaling techniques
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

def create_sample_data():
    data = {'Income': [30000, 50000, 80000, 120000, 35000, 95000],
            'Age': [25, 35, 45, 55, 28, 50]}
    return pd.DataFrame(data)

def method_1_standardscaler():
    print("\n=== METHOD 1: STANDARDSCALER (Z-score) ===")
    df = create_sample_data()
    print("Original:")
    print(df)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    print("\nScaled:")
    print(pd.DataFrame(scaled, columns=df.columns).round(3))

def method_2_minmax():
    print("\n=== METHOD 2: MINMAXSCALER ===")
    df = create_sample_data()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df)
    print("Scaled (0-1):")
    print(pd.DataFrame(scaled, columns=df.columns).round(3))

def method_3_robust():
    print("\n=== METHOD 3: ROBUSTSCALER ===")
    df = create_sample_data()
    scaler = RobustScaler()
    scaled = scaler.fit_transform(df)
    print("Scaled (using IQR):")
    print(pd.DataFrame(scaled, columns=df.columns).round(3))

def method_4_normalizer():
    print("\n=== METHOD 4: NORMALIZER ===")
    df = create_sample_data()
    normalizer = Normalizer(norm='l2')
    scaled = normalizer.fit_transform(df)
    print("L2 Normalized:")
    print(pd.DataFrame(scaled, columns=df.columns).round(3))

def main():
    print("="*60)
    print("FEATURE SCALING TECHNIQUES")
    print("="*60)
    method_1_standardscaler()
    method_2_minmax()
    method_3_robust()
    method_4_normalizer()
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
