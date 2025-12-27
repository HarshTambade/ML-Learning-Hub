# Chapter 02: Advanced Exercises

Advanced practice problems for Python, NumPy, Pandas, and Data Visualization.

## NumPy Advanced Exercises

### Exercise 1: Matrix Operations
Create a 5x5 matrix with random values. Perform the following operations:
- Calculate the determinant
- Find the inverse of the matrix
- Compute eigenvalues and eigenvectors
- Verify that matrix * inverse = identity matrix

```python
import numpy as np

# Create a 5x5 matrix
A = np.random.randn(5, 5)

# Calculate determinant
det_A = np.linalg.det(A)
print(f"Determinant: {det_A}")

# Find inverse
A_inv = np.linalg.inv(A)
print(f"Inverse:\n{A_inv}")

# Verify
identity = np.dot(A, A_inv)
print(f"A * A_inv (should be I):\n{identity}")

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")
```

### Exercise 2: Advanced Indexing and Broadcasting
Create solutions using NumPy advanced features:
- Use fancy indexing to select elements
- Apply broadcasting for element-wise operations
- Use np.where for conditional operations
- Solve a system of linear equations using np.linalg.solve

### Exercise 3: Performance Optimization
Write code that:
- Compares performance between loops and vectorized operations
- Uses broadcasting instead of loops
- Measures execution time using timeit
- Demonstrates why vectorization is important

## Pandas Advanced Exercises

### Exercise 4: Data Cleaning and Transformation
Given a dataset with missing values and outliers:
- Handle missing values using different strategies (mean, median, forward fill)
- Detect and handle outliers using IQR method
- Perform normalization/standardization
- Remove duplicates considering specific columns

```python
import pandas as pd
import numpy as np

# Create sample data with issues
data = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 100, 6],  # Has NaN and outlier
    'B': [10, 20, 20, np.nan, 50, 60],  # Has NaN
    'C': [100, 200, 200, 400, 500, 500]  # Has duplicates
})

# Handle missing values
data_filled = data.fillna(data.mean())
data_ffill = data.fillna(method='ffill')

# Detect outliers using IQR
Q1 = data['A'].quantile(0.25)
Q3 = data['A'].quantile(0.75)
IQR = Q3 - Q1
outliers = data[(data['A'] < Q1 - 1.5*IQR) | (data['A'] > Q3 + 1.5*IQR)]
print(f"Outliers:\n{outliers}")

# Remove duplicates
data_unique = data.drop_duplicates(subset=['C'])
print(f"After removing duplicates:\n{data_unique}")
```

### Exercise 5: GroupBy Operations
Perform complex groupby operations:
- Group by multiple columns
- Apply multiple aggregation functions
- Use custom aggregation functions
- Chain multiple operations
- Create pivot tables with margins

### Exercise 6: Time Series Analysis
Work with time series data:
- Resample data to different frequencies
- Calculate rolling statistics (mean, std, etc.)
- Detect trends and seasonality
- Forward and backward fill missing dates
- Calculate lagged features

```python
import pandas as pd

# Create time series data
date_range = pd.date_range('2023-01-01', periods=100, freq='D')
data = pd.Series(np.random.randn(100).cumsum(), index=date_range)

# Rolling statistics
rolling_mean = data.rolling(window=7).mean()
rolling_std = data.rolling(window=7).std()

# Resample
weekly_mean = data.resample('W').mean()
monthly_mean = data.resample('M').mean()

print(f"Weekly mean:\n{weekly_mean}")
print(f"Monthly mean:\n{monthly_mean}")
```

## Visualization Advanced Exercises

### Exercise 7: Multi-Panel Visualizations
Create a complex figure with multiple subplots:
- Create a 3x3 grid of plots
- Use different plot types in each subplot
- Share axes between subplots
- Use GridSpec for custom layouts
- Add colorbar for heatmaps

### Exercise 8: Interactive Visualizations
Create interactive plots using Plotly:
- Create dropdowns to switch between datasets
- Add hover information
- Create range sliders
- Animate plots over time
- Export as HTML

```python
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Create interactive scatter with animation
data = px.data.gapminder()
fig = px.scatter(data, x="gdpPercap", y="lifeExp", size="pop",
                 color="continent", hover_name="country",
                 size_max=60, range_x=[100, 100000], range_y=[25, 90],
                 animation_frame="year", animation_group="country",
                 log_x=True, size_max=45,
                 title='GDP vs Life Expectancy')
fig.show()
```

### Exercise 9: Statistical Visualizations
Create visualizations for statistical analysis:
- Create distribution plots with different kde styles
- Joint plots with marginal distributions
- Pair plots for multivariate analysis
- Heatmaps with dendrogram clustering
- Q-Q plots for normality testing

### Exercise 10: Custom Styling
Create publication-ready figures:
- Apply custom color palettes
- Set figure size and DPI
- Add annotations and arrows
- Control font sizes and styles
- Save figures in multiple formats (png, pdf, svg)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

# Create plot
data = np.random.randn(1000)
ax.hist(data, bins=50, alpha=0.7, edgecolor='black')

# Add annotations
ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
ax.annotate('Peak', xy=(0, 100), xytext=(1, 150),
            arrowprops=dict(arrowstyle='->', color='black'))

# Customize
ax.set_xlabel('Value', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution Plot', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)

# Save in multiple formats
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.savefig('plot.pdf', bbox_inches='tight')
plt.savefig('plot.svg', bbox_inches='tight')
plt.show()
```

## Integration Challenges

### Challenge 1: End-to-End Data Pipeline
Create a complete data pipeline:
1. Load raw data from CSV
2. Clean and preprocess using Pandas
3. Perform exploratory data analysis (EDA)
4. Create multiple visualizations
5. Calculate summary statistics using NumPy
6. Export cleaned data and visualizations

### Challenge 2: Performance Comparison
Compare performance across:
- NumPy vs Pandas for same operations
- Vectorized vs loops in NumPy
- Different plotting libraries
- Memory usage of different data structures

### Challenge 3: Real-world Dataset Analysis
Analyze a real dataset (e.g., from Kaggle):
- Load and explore the data
- Identify and handle data quality issues
- Perform statistical analysis
- Create meaningful visualizations
- Draw insights from the data
- Document findings in a report

## Solutions Guide

For each exercise:
1. First attempt the problem without looking at solutions
2. Compare your approach with the reference solution
3. Understand alternative approaches
4. Optimize for readability and performance
5. Document your code with comments

## Resources

- NumPy Documentation: https://numpy.org/doc/
- Pandas Documentation: https://pandas.pydata.org/docs/
- Matplotlib Documentation: https://matplotlib.org/
- Seaborn Documentation: https://seaborn.pydata.org/
- Plotly Documentation: https://plotly.com/python/

## Tips for Success

1. **Start simple**: Understand basic operations before advanced ones
2. **Practice regularly**: Consistency is key to mastery
3. **Debug systematically**: Use print statements and debuggers
4. **Read error messages**: They provide valuable information
5. **Look at documentation**: Official docs have great examples
6. **Experiment freely**: Try different approaches
7. **Test your code**: Write test cases for your functions
8. **Optimize later**: First make it work, then optimize
