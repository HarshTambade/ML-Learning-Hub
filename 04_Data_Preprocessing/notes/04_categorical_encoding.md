04_categorical_encoding.md  # Categorical Encoding

## Introduction

Categorical variables are non-numerical features that need to be converted to numerical format for machine learning algorithms.

## Common Categorical Encoding Techniques

### 1. Label Encoding

Converts categories to integers (0, 1, 2, ...)

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['color_encoded'] = le.fit_transform(df['color'])

# Mapping
print(dict(zip(le.classes_, le.transform(le.classes_))))
```

**Use When**: Ordinal categories or binary classification
**Pros**: Simple, memory efficient
**Cons**: Implies ordinal relationship

### 2. One-Hot Encoding

Creates binary columns for each category

```python
df_encoded = pd.get_dummies(df, columns=['color'])

# Or with sklearn
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse=False, drop='first')
colors_encoded = enc.fit_transform(df[['color']])
```

**Use When**: Nominal categories, tree-based models
**Pros**: No ordinal relationship
**Cons**: Curse of dimensionality with many categories

### 3. Ordinal Encoding

For ordinal categorical data with meaningful order

```python
from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder(categories=[['low', 'medium', 'high']])
df_encoded = enc.fit_transform(df[['quality']])
```

### 4. Target Encoding

Encodes categories by target mean

```python
target_means = df.groupby('color')['target'].mean()
df['color_target_encoded'] = df['color'].map(target_means)
```

**Use When**: High cardinality categories
**Cons**: Risk of overfitting

### 5. Frequency Encoding

Replaces categories with their frequency

```python
frequency = df['color'].value_counts(normalize=True).to_dict()
df['color_freq'] = df['color'].map(frequency)
```

### 6. Embedding

Learned representations (neural networks)

```python
# Using embedding layer in Keras
from tensorflow.keras.layers import Embedding

embedding = Embedding(input_dim=num_categories, output_dim=8)
```

## Handling Unseen Categories

```python
# Handle unseen categories
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X_train)
X_test = enc.transform(X_test)  # Won't error on new categories
```

## High Cardinality Features

```python
# Group rare categories
threshold = 0.05
values = df['category'].value_counts(normalize=True)
rare_categories = values[values < threshold].index
df['category_grouped'] = df['category'].apply(
    lambda x: 'other' if x in rare_categories else x
)
```

## Performance Considerations

| Encoding | Memory | Speed | Works With |
|----------|--------|-------|------------|
| Label | Low | Fast | All |
| One-Hot | High | Fast | All |
| Ordinal | Low | Fast | Tree-based |
| Target | Low | Medium | All |
| Frequency | Low | Fast | All |
| Embedding | Medium | Slow | Neural Nets |

## Best Practices

1. **Fit on training data**: Prevent data leakage
2. **Handle missing values**: Before encoding
3. **Rare categories**: Group or use target encoding
4. **Cardinality**: Be aware of one-hot explosion
5. **Validation**: Check encoded values

## Summary

Choose encoding based on:
- Data type (ordinal vs nominal)
- Algorithm type
- Cardinality
- Model interpretation needs
