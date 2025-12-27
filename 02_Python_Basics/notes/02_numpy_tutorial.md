# NumPy: Numerical Python

## Introduction

NumPy is the fundamental package for numerical computing in Python. It provides support for arrays and matrices, along with a large collection of mathematical functions.

## Why NumPy?

1. **Performance**: NumPy arrays are much faster than Python lists
2. **Convenience**: Easy-to-use syntax for mathematical operations
3. **Integration**: Works seamlessly with other scientific packages
4. **Memory Efficiency**: More compact than native Python data structures

## NumPy Arrays

### Creating Arrays

```python
import numpy as np

# From list
arr = np.array([1, 2, 3, 4, 5])

# Special arrays
arr_zeros = np.zeros((3, 4))      # All zeros
arr_ones = np.ones((3, 4))        # All ones
arr_range = np.arange(0, 10, 2)   # Like range()
arr_linspace = np.linspace(0, 1, 5)  # Evenly spaced

# Random arrays
arr_random = np.random.rand(3, 4)     # Random [0, 1)
arr_normal = np.random.randn(3, 4)    # Normal distribution
arr_randint = np.random.randint(0, 10, (3, 4))  # Random integers
```

### Array Properties

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.shape        # (2, 3)
arr.size         # 6
arr.ndim         # 2
arr.dtype        # Data type
arr.itemsize     # Bytes per element
```

## Array Indexing and Slicing

### 1D Indexing

```python
arr = np.array([0, 1, 2, 3, 4, 5])

arr[2]       # Element at index 2
arr[-1]       # Last element
arr[1:4]      # Elements 1 to 3
arr[::2]      # Every 2nd element
arr[::-1]     # Reversed
```

### 2D Indexing

```python
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

matrix[0, 1]      # Row 0, Column 1
matrix[1, :]      # Second row
matrix[:, 2]      # Third column
matrix[1:, :2]    # Rows 1-end, Columns 0-1
```

## Array Operations

### Element-wise Operations

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b          # [5, 7, 9]
a - b          # [-3, -3, -3]
a * b          # [4, 10, 18]
a / b          # [0.25, 0.4, 0.5]
a ** 2         # [1, 4, 9]
np.sqrt(a)     # [1, 1.41, 1.73]
```

### Broadcasting

```python
# Adding scalar to array
arr = np.array([1, 2, 3])
arr + 10  # [11, 12, 13]

# Adding arrays of different shapes
a = np.array([[1], [2], [3]])  # Shape (3, 1)
b = np.array([1, 2, 3])         # Shape (3,)
c = a + b                        # Shape (3, 3) due to broadcasting
```

## Aggregate Functions

```python
arr = np.array([1, 2, 3, 4, 5])

np.sum(arr)      # 15
np.mean(arr)     # 3.0
np.std(arr)      # Standard deviation
np.var(arr)      # Variance
np.max(arr)      # 5
np.min(arr)      # 1
np.median(arr)   # 3.0
```

## Reshaping Arrays

```python
arr = np.arange(12)

arr.reshape(3, 4)           # 2D array (3x4)
arr.reshape(2, 2, 3)        # 3D array
arr.reshape(-1, 4)          # Infer first dimension
arr.flatten()               # Flatten to 1D
arr.ravel()                 # Flatten (view instead of copy)
```

## Linear Algebra

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

np.dot(a, b)                # Matrix multiplication
a @ b                       # Matrix multiplication (Python 3.5+)
np.linalg.inv(a)           # Matrix inverse
np.linalg.det(a)           # Determinant
np.linalg.solve(a, [1, 2]) # Solve system of equations
```

## Useful Functions

```python
# Sorting
arr = np.array([3, 1, 4, 1, 5, 9])
np.sort(arr)              # [1, 1, 3, 4, 5, 9]
np.argsort(arr)           # Indices that sort array

# Unique elements
np.unique(arr)            # [1, 3, 4, 5, 9]

# Where
condition = arr > 2
np.where(condition)       # Indices where condition is True

# Concatenate
a = np.array([1, 2])
b = np.array([3, 4])
np.concatenate([a, b])    # [1, 2, 3, 4]
```

## Performance Tips

1. **Vectorize Code**: Use NumPy operations instead of loops
2. **Use Views**: Slicing creates views, not copies
3. **Specify dtype**: Save memory by using appropriate data types
4. **Avoid Copies**: Be mindful of when copies are created

## Key Takeaways

- NumPy arrays are faster than Python lists
- Broadcasting allows operations on different shaped arrays
- Indexing is powerful for selecting data
- Vectorized operations are essential for ML
- NumPy is the foundation for pandas and scikit-learn
