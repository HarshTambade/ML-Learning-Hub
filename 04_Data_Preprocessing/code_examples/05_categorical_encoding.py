05_categorical_encoding.py  """
Chapter 04: Categorical Encoding
Demonstrates various categorical encoding techniques
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

def create_sample_data():
    data = {'Color': ['Red', 'Blue', 'Red', 'Green', 'Blue', 'Green'],
            'Department': ['HR', 'IT', 'Sales', 'IT', 'HR', 'Sales']}
    return pd.DataFrame(data)

def method_1_label_encoding():
    print("\n=== METHOD 1: LABEL ENCODING ===")
    df = create_sample_data()
    print("Original:")
    print(df)
    le = LabelEncoder()
    df['Color_encoded'] = le.fit_transform(df['Color'])
    print("\nAfter Label Encoding:")
    print(df[['Color', 'Color_encoded']])
    print(f"Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

def method_2_onehot_encoding():
    print("\n=== METHOD 2: ONE-HOT ENCODING ===")
    df = create_sample_data()
    df_encoded = pd.get_dummies(df, columns=['Department'], prefix='Dept')
    print("\nAfter One-Hot Encoding:")
    print(df_encoded)

def method_3_ordinal_encoding():
    print("\n=== METHOD 3: ORDINAL ENCODING ===")
    data = {'Grade': ['A', 'B', 'C', 'A', 'B', 'C']}
    df = pd.DataFrame(data)
    print("Original:")
    print(df)
    enc = OrdinalEncoder(categories=[['A', 'B', 'C']])
    df['Grade_encoded'] = enc.fit_transform(df[['Grade']])
    print("\nAfter Ordinal Encoding:")
    print(df)

def method_4_frequency_encoding():
    print("\n=== METHOD 4: FREQUENCY ENCODING ===")
    df = create_sample_data()
    freq = df['Color'].value_counts(normalize=True).to_dict()
    df['Color_freq'] = df['Color'].map(freq)
    print("\nAfter Frequency Encoding:")
    print(df[['Color', 'Color_freq']])

def main():
    print("="*60)
    print("CATEGORICAL ENCODING TECHNIQUES")
    print("="*60)
    method_1_label_encoding()
    method_2_onehot_encoding()
    method_3_ordinal_encoding()
    method_4_frequency_encoding()
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
