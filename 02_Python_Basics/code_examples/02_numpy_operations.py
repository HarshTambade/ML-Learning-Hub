#!/usr/bin/env python3
"""
NumPy Operations - Practical examples of NumPy array operations
"""

import numpy as np

def main():
    print("=== NumPy Operations Examples ===")
    print()
    
    # 1. Creating arrays
    print("1. Creating Arrays")
    print("=" * 40)
    arr1d = np.array([1, 2, 3, 4, 5])
    arr2d = np.array([[1, 2, 3], [4, 5, 6]])
    zeros = np.zeros((3, 3))
    ones = np.ones((2, 4))
    range_arr = np.arange(0, 10, 2)
    linspace_arr = np.linspace(0, 1, 5)
    
    print(f"1D array: {arr1d}")
    print(f"2D array shape: {arr2d.shape}")
    print(f"Zeros: {zeros.shape}")
    print(f"Range: {range_arr}")
    print()
    
    # 2. Array indexing
    print("2. Array Indexing")
    print("=" * 40)
    print(f"arr1d[2] = {arr1d[2]}")
    print(f"arr1d[1:4] = {arr1d[1:4]}")
    print(f"arr2d[0, :] = {arr2d[0, :]}")
    print(f"arr2d[:, 1] = {arr2d[:, 1]}")
    print()
    
    # 3. Element-wise operations
    print("3. Element-wise Operations")
    print("=" * 40)
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    print(f"a ** 2 = {a ** 2}")
    print(f"np.sqrt(a) = {np.sqrt(a)}")
    print()
    
    # 4. Broadcasting
    print("4. Broadcasting")
    print("=" * 40)
    arr = np.array([1, 2, 3])
    scalar = 10
    result = arr + scalar
    print(f"array: {arr}")
    print(f"array + 10 = {result}")
    print()
    
    # 5. Aggregate functions
    print("5. Aggregate Functions")
    print("=" * 40)
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"Data: {data}")
    print(f"Sum: {np.sum(data)}")
    print(f"Mean: {np.mean(data)}")
    print(f"Std: {np.std(data):.2f}")
    print(f"Max: {np.max(data)}")
    print(f"Min: {np.min(data)}")
    print()
    
    # 6. Reshaping
    print("6. Reshaping Arrays")
    print("=" * 40)
    arr = np.arange(12)
    print(f"Original shape: {arr.shape}")
    reshaped = arr.reshape(3, 4)
    print(f"Reshaped (3, 4):\n{reshaped}")
    flattened = reshaped.flatten()
    print(f"Flattened: {flattened}")
    print()
    
    # 7. Matrix operations
    print("7. Matrix Operations")
    print("=" * 40)
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    print(f"A = \n{A}")
    print(f"B = \n{B}")
    print(f"A @ B (Matrix multiplication) =\n{A @ B}")
    print()
    
    # 8. Sorting and searching
    print("8. Sorting and Searching")
    print("=" * 40)
    unsorted = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    print(f"Unsorted: {unsorted}")
    print(f"Sorted: {np.sort(unsorted)}")
    print(f"Unique: {np.unique(unsorted)}")
    print()
    
    # 9. Statistical operations
    print("9. Statistical Operations")
    print("=" * 40)
    data = np.random.randn(1000)
    print(f"Mean: {np.mean(data):.4f}")
    print(f"Std: {np.std(data):.4f}")
    print(f"Median: {np.median(data):.4f}")
    print(f"Min: {np.min(data):.4f}")
    print(f"Max: {np.max(data):.4f}")
    print()
    
    print("=== NumPy is Powerful! ===")

if __name__ == "__main__":
    main()
