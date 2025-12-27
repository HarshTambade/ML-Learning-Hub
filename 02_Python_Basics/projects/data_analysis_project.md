# Chapter 02 Project: Data Analysis with Python

## Project Overview

Analyze a real-world dataset using Python, NumPy, and Pandas to practice data manipulation and analysis skills.

## Dataset Options

1. **Iris Dataset** (built-in with sklearn)
2. **Titanic Dataset** (from Kaggle)
3. **House Prices Dataset** (from Kaggle)
4. **Stock Price Data** (from Yahoo Finance)
5. **Weather Data** (from weather APIs)

## Project Requirements

### Phase 1: Data Loading & Exploration
- [ ] Load dataset into Pandas DataFrame
- [ ] Display basic info (shape, columns, dtypes)
- [ ] Check first/last rows
- [ ] Identify missing values
- [ ] Get summary statistics

### Phase 2: Data Cleaning
- [ ] Handle missing values (drop/fill)
- [ ] Remove duplicates if any
- [ ] Fix data types
- [ ] Identify and handle outliers
- [ ] Create clean dataset

### Phase 3: Exploratory Data Analysis (EDA)
- [ ] Calculate summary statistics per column
- [ ] Analyze distributions
- [ ] Find correlations between variables
- [ ] Identify patterns and relationships
- [ ] Document findings

### Phase 4: Visualization
- [ ] Create histograms for numerical features
- [ ] Plot scatter plots for relationships
- [ ] Generate correlation heatmap
- [ ] Create box plots for outliers
- [ ] Make 1-2 custom visualizations

### Phase 5: Insights & Report
- [ ] Summarize key findings
- [ ] Answer 3-5 business questions
- [ ] Provide recommendations
- [ ] Create a professional report

## Sample Code Structure

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
df = pd.read_csv('dataset.csv')

# 2. Explore
print(df.info())
print(df.describe())
print(df.isnull().sum())

# 3. Clean
df.dropna(inplace=True)  # or fill with median/mean
df.drop_duplicates(inplace=True)

# 4. Analyze
corr_matrix = df.corr()
print(df.groupby('column').mean())

# 5. Visualize
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
```

## Learning Outcomes

- ✅ Comfortable loading and exploring datasets
- ✅ Skilled in data cleaning and preprocessing
- ✅ Able to perform basic EDA
- ✅ Create publication-quality visualizations
- ✅ Extract actionable insights from data

## Deliverables

1. **Jupyter Notebook** with commented code
2. **Analysis Report** (markdown or PDF)
3. **Visualizations** (saved as PNG/PDF)
4. **Summary Statistics** table
5. **Key Findings** document (1 page max)

## Evaluation Criteria

- Data quality (cleaning completeness)
- Analysis depth (number of insights)
- Code quality (clean, commented)
- Visualization clarity (easy to understand)
- Report quality (professional presentation)
- Findings validity (based on data)

## Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Tutorial](https://numpy.org/doc/stable/user/basics.html)
- [Matplotlib Guide](https://matplotlib.org/stable/tutorials/)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Kaggle Datasets](https://www.kaggle.com/datasets)

## Submission Guidelines

1. Create a folder with your project name
2. Include notebook, report, and visualizations
3. Add a README.md explaining your analysis
4. Provide clear instructions to run your code
5. Submit complete and functional code

## Success Checklist

- [ ] Dataset loaded successfully
- [ ] All data types correctly identified
- [ ] Missing values handled appropriately
- [ ] At least 10 data insights found
- [ ] 4+ high-quality visualizations created
- [ ] Report is clear and professional
- [ ] Code is clean and well-commented
- [ ] Analysis is statistically sound
