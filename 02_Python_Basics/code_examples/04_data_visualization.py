"""Data Visualization Examples

Comprehensive examples of creating visualizations using Matplotlib, Seaborn, and Plotly.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ===== MATPLOTLIB BASIC PLOTS =====
def matplotlib_basic_plots():
    """Create basic plots using Matplotlib."""
    print("\n=== Matplotlib Basic Plots ===")
    
    # Generate sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Line plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(x, y, linewidth=2, color='blue')
    plt.title('Line Plot')
    plt.xlabel('X')
    plt.ylabel('sin(X)')
    plt.grid(True, alpha=0.3)
    
    # Scatter plot
    plt.subplot(1, 3, 2)
    np.random.seed(42)
    x_scatter = np.random.randn(50)
    y_scatter = np.random.randn(50)
    plt.scatter(x_scatter, y_scatter, alpha=0.6, s=100, color='red')
    plt.title('Scatter Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Bar plot
    plt.subplot(1, 3, 3)
    categories = ['A', 'B', 'C', 'D']
    values = [10, 24, 36, 18]
    plt.bar(categories, values, color='green', alpha=0.7)
    plt.title('Bar Plot')
    plt.ylabel('Values')
    
    plt.tight_layout()
    print("Matplotlib plots created (use plt.show() to display)")

# ===== MATPLOTLIB SUBPLOTS =====
def matplotlib_subplots():
    """Create multiple subplots using Matplotlib."""
    print("\n=== Matplotlib Subplots ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Plot 1: Line plot
    x = np.linspace(0, 10, 100)
    axes[0, 0].plot(x, np.sin(x), 'b-', linewidth=2)
    axes[0, 0].set_title('Sine Wave')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('sin(X)')
    
    # Plot 2: Histogram
    data = np.random.randn(1000)
    axes[0, 1].hist(data, bins=30, color='orange', alpha=0.7)
    axes[0, 1].set_title('Histogram')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    
    # Plot 3: Box plot
    data_dict = {'A': np.random.randn(100),
                 'B': np.random.randn(100),
                 'C': np.random.randn(100)}
    axes[1, 0].boxplot(data_dict.values(), labels=data_dict.keys())
    axes[1, 0].set_title('Box Plot')
    axes[1, 0].set_ylabel('Values')
    
    # Plot 4: Pie chart
    sizes = [30, 25, 20, 25]
    labels = ['A', 'B', 'C', 'D']
    axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%')
    axes[1, 1].set_title('Pie Chart')
    
    plt.tight_layout()
    print("Subplots created (use plt.show() to display)")

# ===== SEABORN DISTRIBUTION PLOTS =====
def seaborn_distribution():
    """Create distribution plots using Seaborn."""
    print("\n=== Seaborn Distribution Plots ===")
    
    # Create sample data
    data = pd.DataFrame({
        'values': np.random.normal(loc=100, scale=15, size=1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram with KDE
    sns.histplot(data['values'], kde=True, ax=axes[0], bins=30)
    axes[0].set_title('Histogram with KDE')
    
    # Count plot
    sns.countplot(x='category', data=data, ax=axes[1], palette='Set2')
    axes[1].set_title('Count Plot')
    
    plt.tight_layout()
    print("Seaborn distribution plots created")

# ===== SEABORN RELATIONSHIP PLOTS =====
def seaborn_relationship():
    """Create relationship plots using Seaborn."""
    print("\n=== Seaborn Relationship Plots ===")
    
    # Create sample data
    np.random.seed(42)
    n = 200
    x = np.random.randn(n)
    y = 2 * x + np.random.randn(n) * 0.5
    
    data = pd.DataFrame({'X': x, 'Y': y, 'Category': np.random.choice(['A', 'B'], n)})
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Scatter plot with regression line
    sns.regplot(x='X', y='Y', data=data, ax=axes[0], scatter_kws={'s': 50, 'alpha': 0.6})
    axes[0].set_title('Regression Plot')
    
    # Scatter plot colored by category
    sns.scatterplot(x='X', y='Y', hue='Category', data=data, ax=axes[1], s=100, alpha=0.7)
    axes[1].set_title('Scatter Plot by Category')
    
    plt.tight_layout()
    print("Seaborn relationship plots created")

# ===== SEABORN HEATMAP =====
def seaborn_heatmap():
    """Create heatmap using Seaborn."""
    print("\n=== Seaborn Heatmap ===")
    
    # Create sample correlation data
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(10, 8))
    correlation = data.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
                cbar_kws={'label': 'Correlation'}, square=True)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    print("Heatmap created")

# ===== SEABORN CATEGORICAL PLOTS =====
def seaborn_categorical():
    """Create categorical plots using Seaborn."""
    print("\n=== Seaborn Categorical Plots ===")
    
    # Create sample data
    data = pd.DataFrame({
        'category': np.repeat(['A', 'B', 'C'], 50),
        'value': np.concatenate([
            np.random.normal(100, 15, 50),
            np.random.normal(110, 15, 50),
            np.random.normal(95, 15, 50)
        ])
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Box plot
    sns.boxplot(x='category', y='value', data=data, ax=axes[0], palette='Set2')
    axes[0].set_title('Box Plot')
    
    # Violin plot
    sns.violinplot(x='category', y='value', data=data, ax=axes[1], palette='Set2')
    axes[1].set_title('Violin Plot')
    
    # Strip plot with jitter
    sns.stripplot(x='category', y='value', data=data, ax=axes[2], 
                  jitter=True, alpha=0.5, size=8)
    axes[2].set_title('Strip Plot')
    
    plt.tight_layout()
    print("Categorical plots created")

# ===== TIME SERIES PLOTTING =====
def time_series_plot():
    """Create time series plot."""
    print("\n=== Time Series Plot ===")
    
    # Create sample time series data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    values = np.cumsum(np.random.randn(100)) + 100
    
    data = pd.DataFrame({'Date': dates, 'Value': values})
    
    plt.figure(figsize=(12, 5))
    plt.plot(data['Date'], data['Value'], linewidth=2, color='blue')
    plt.fill_between(data['Date'], data['Value'], alpha=0.3)
    plt.title('Time Series Plot')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    print("Time series plot created")

# ===== CUSTOM STYLING =====
def custom_styling():
    """Create plot with custom styling."""
    print("\n=== Custom Styling ===")
    
    # Set style
    sns.set_style('darkgrid')
    sns.set_palette('husl')
    
    # Create sample data
    x = np.linspace(0, 10, 100)
    
    plt.figure(figsize=(10, 6))
    
    # Plot multiple lines with different styles
    for i in range(1, 5):
        plt.plot(x, np.sin(x + i * 0.5), label=f'sin(x + {i*0.5})', linewidth=2)
    
    plt.title('Multiple Sine Waves', fontsize=16, fontweight='bold')
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    print("Custom styled plot created")

# ===== MAIN EXECUTION =====
if __name__ == '__main__':
    print("Data Visualization Examples")
    print("=" * 50)
    
    matplotlib_basic_plots()
    matplotlib_subplots()
    seaborn_distribution()
    seaborn_relationship()
    seaborn_heatmap()
    seaborn_categorical()
    time_series_plot()
    custom_styling()
    
    # Uncomment below to display plots
    # plt.show()
    
    print("\n" + "=" * 50)
    print("All visualization examples completed!")
    print("Run with plt.show() to display plots")
