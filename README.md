# Predictive Analysis of Customer Churn Using Behavioral Metrics

A simple, learning-focused Python script that ingests the Telco Customer Churn dataset, performs basic data cleaning, feature engineering, aggregation, and visualization to explore behavioral drivers of customer churn.

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Data Source](#data-source)  
3. [Prerequisites](#prerequisites)  
4. [Getting Started](#getting-started)  
5. [Script Breakdown](#script-breakdown)  
   - [Data Cleaning](#data-cleaning)  
   - [Feature Engineering](#feature-engineering)  
   - [Aggregations & Summaries](#aggregations--summaries)  
   - [Visualizations](#visualizations)  
   - [Advanced Crosstab Analysis](#advanced-crosstab-analysis)  
6. [Tips & Hints](#tips--hints)  
7. [Next Steps](#next-steps)  
8. [License](#license)

## Project Overview

This script walks through exploratory data analysis (EDA) steps on customer behavioral metrics to understand and visualize factors related to churn. It is designed for learners to:

- Handle missing or malformed data  
- Convert categorical features into machine-friendly formats  
- Engineer new features  
- Compute group-level metrics and compare churn rates  
- Generate simple plots for insights

## Data Source

We use Kaggle’s Telco Customer Churn dataset:  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Download the CSV file `WA_Fn-UseC_-Telco-Customer-Churn.csv` and place it in the same directory as the script.

## Prerequisites

- Python 3.6 or higher  
- Required Python packages:
  - pandas  
  - numpy  
  - matplotlib  

Install dependencies with:

```bash
pip install pandas numpy matplotlib
```

## Getting Started

1. Clone or download this repository.  
2. Ensure the dataset CSV is in the project folder.  
3. Run the analysis script:

```bash
python churn_analysis.py
```

## Script Breakdown

Below is a high-level explanation of what each section of the script does.

### Data Cleaning

- Read CSV into a pandas DataFrame.  
- Convert the `TotalCharges` column to numeric, coercing blanks to NaN.  
- Replace missing `TotalCharges` with the column median.  
- Map `SeniorCitizen` from `0/1` to `"No"/"Yes"` for clarity.

### Feature Engineering

- **Tenure Ratio**:  
  Create a new column `tenure_ratio` that scales tenure by contract length:
  - Two-year contracts → tenure / 24  
  - One-year contracts → tenure / 12  
  - Month-to-month → raw tenure  
- **High-Risk Flag**:  
  Mark customers on a month-to-month plan with monthly charges > \$70 as `"Yes"` in a new `Flagged` column.

### Aggregations & Summaries

- Compute churn rate by payment method using `groupby` + lambda:
  ```python
  churn_by_payment = df.groupby("PaymentMethod")["Churn"].apply(
      lambda x: (x == 'Yes').mean() * 100
  )
  ```
- Calculate average monthly charges for churned vs. retained customers with `groupby('Churn')['MonthlyCharges'].mean()`.

### Visualizations

1. **Bar Plot**: Churn counts by contract type.  
2. **KDE Plot**: Tenure distribution of churned customers.

Each plot is built with Matplotlib, annotated with titles, labels, and data labels.

### Advanced Crosstab Analysis

- Use `pd.crosstab` to compare churn vs. Internet service type.  
- Add total counts and compute churn rate percentages.

```python
cross_tab = pd.crosstab(
    df['InternetService'],
    df['Churn'],
    margins=True,
    margins_name="Total"
)
cross_tab['Churn Rate (%)'] = (cross_tab['Yes'] / cross_tab['Total'] * 100).round(1)
```

## Tips & Hints

- Convert categorical variables with `pd.get_dummies()` when building predictive models.  
- Watch out for skewed distributions and outliers in `TotalCharges`.  
- Use `.agg()` for multi-metric summaries, e.g.:
  ```python
  df.groupby('Churn')['tenure'].agg(['mean','min','max'])
  ```

## Next Steps

- Build a machine learning model (e.g., logistic regression) to predict churn.  
- Automate hyperparameter tuning and cross-validation.  
- Deploy the analysis as a Jupyter notebook with interactive widgets.  
- Explore additional visualizations (heatmaps, pairplots).

## License

This project is provided under the MIT License. Feel free to use and modify for educational purposes.  
