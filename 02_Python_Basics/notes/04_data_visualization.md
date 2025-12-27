# Data Visualization Guide

## Overview
Data visualization is crucial for exploring data, communicating insights, and building intuition about datasets. This guide covers the most popular visualization libraries in Python.

## Matplotlib - Foundational Library

### Basic Plots
```python
import matplotlib.pyplot as plt
import numpy as np

# Line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Sine Wave')
plt.show()

# Scatter plot
plt.scatter([1, 2, 3], [1, 4, 9])
plt.show()

# Bar plot
plt.bar(['A', 'B', 'C'], [10, 20, 15])
plt.show()
```

### Figure and Subplots
```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Create multiple plots
axes[0, 0].plot([1, 2, 3], [1, 2, 3])
axes[0, 1].scatter([1, 2, 3], [1, 4, 9])
axes[1, 0].bar(['A', 'B'], [5, 10])
axes[1, 1].hist([1, 2, 2, 3, 3, 3, 4], bins=4)

plt.tight_layout()
plt.show()
```

### Customization
```python
plt.figure(figsize=(12, 6))
plt.plot(x, y, color='red', linewidth=2, linestyle='--', label='Line')
plt.scatter(x[:10], y[:10], color='blue', s=100, alpha=0.5, label='Points')
plt.xlabel('X Label', fontsize=12)
plt.ylabel('Y Label', fontsize=12)
plt.title('Plot Title', fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.show()
```

## Seaborn - Statistical Visualization

### Distribution Plots
```python
import seaborn as sns

# Histogram with KDE
sns.histplot(data, kde=True)
plt.show()

# Distribution plot
sns.distplot(data)
plt.show()

# Count plot
sns.countplot(x='category', data=df)
plt.show()
```

### Relationship Plots
```python
# Scatter plot with regression
sns.regplot(x='feature1', y='feature2', data=df)
plt.show()

# Multiple regression lines
sns.lmplot(x='feature1', y='feature2', hue='category', data=df)
plt.show()

# Joint plot
sns.jointplot(x='feature1', y='feature2', data=df, kind='scatter')
plt.show()
```

### Matrix Plots
```python
# Correlation heatmap
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Pairplot
sns.pairplot(df, hue='target')
plt.show()
```

### Categorical Plots
```python
# Box plot
sns.boxplot(x='category', y='value', data=df)
plt.show()

# Violin plot
sns.violinplot(x='category', y='value', data=df)
plt.show()

# Bar plot with error bars
sns.barplot(x='category', y='value', data=df, ci=95)
plt.show()
```

## Plotly - Interactive Visualizations

### Basic Interactive Plots
```python
import plotly.express as px
import plotly.graph_objects as go

# Scatter plot
fig = px.scatter(df, x='feature1', y='feature2', color='category')
fig.show()

# Line plot with custom colors
fig = px.line(df, x='date', y='value', title='Time Series')
fig.show()

# Bar chart
fig = px.bar(df, x='category', y='value', color='subcategory')
fig.show()
```

### Advanced Interactive Features
```python
fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(
    x=[1, 2, 3],
    y=[4, 5, 6],
    mode='lines+markers',
    name='Line 1'
))

# Add layout
fig.update_layout(
    title='Interactive Plot',
    xaxis_title='X Axis',
    yaxis_title='Y Axis',
    hovermode='x unified'
)

fig.show()
```

## Best Practices

### Design Principles
1. **Simplicity**: Avoid chart junk and unnecessary elements
2. **Clarity**: Use clear labels, titles, and legends
3. **Color**: Use appropriate color palettes (colorblind-friendly)
4. **Consistency**: Maintain consistent style across visualizations
5. **Accuracy**: Ensure axes are properly scaled and labeled

### Common Patterns
```python
# Create figure with professional styling
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(10, 6))

# Plot with annotations
ax.plot(x, y, linewidth=2, color='#1f77b4')
ax.fill_between(x, y, alpha=0.3)
ax.annotate('Peak', xy=(peak_x, peak_y), xytext=(peak_x, peak_y+0.5),
            arrowprops=dict(arrowstyle='->'))

# Formatting
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_title('Title')
ax.legend()

plt.tight_layout()
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Visualization Types and Use Cases

| Plot Type | Use Case | Best For |
|-----------|----------|----------|
| Line Plot | Time series, trends | Continuous data over time |
| Scatter Plot | Relationships | Two continuous variables |
| Bar Chart | Comparisons | Categorical comparisons |
| Histogram | Distributions | Frequency of values |
| Box Plot | Outliers, quartiles | Distribution summary |
| Heatmap | Correlation, patterns | Matrix data |
| Pie Chart | Composition | Part-to-whole relationships |
| Violin Plot | Distribution details | Multiple distributions |

## Performance Tips
- Use aggregation for large datasets (scatter plot with 1M+ points)
- Reduce alpha transparency for overplotting
- Use raster format for many small elements
- Cache computations for interactive plots

## Resources
- Matplotlib: https://matplotlib.org/
- Seaborn: https://seaborn.pydata.org/
- Plotly: https://plotly.com/python/
- Color palettes: https://colorbrewer2.org/
