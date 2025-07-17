"""
Title: "Predictive Analysis of Customer Churn Using Behavioral Metrics"

Where to Get Data:
Kaggle’s Telco Customer Churn dataset:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Tips:
Convert categorical variables (e.g., gender, Contract) into dummy variables (pd.get_dummies()).
Handle skewed data (e.g., TotalCharges might have outliers).
Use .agg() for multi-metric summaries (e.g., mean tenure of churned vs. retained customers).
Tasks to Perform
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data
file_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

# Data Cleaning:
# Fix missing values in TotalCharges (e.g., replace with median).
# CORRECTION: Convert to numeric first (original data has blank strings)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
x = df["TotalCharges"].median()  
df.fillna({"TotalCharges": x}, inplace=True)

# Convert SeniorCitizen from 0/1 to categorical ("Yes"/"No").
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

# Feature Engineering:
# Create a tenure-to-contract ratio (e.g., tenure / 24 for 2-year contracts).
# FIX: Calculate per-customer ratio instead of global sum
conditions = [
    df['Contract'] == 'Two year',
    df['Contract'] == 'One year',
    df['Contract'] == 'Month-to-month'
]
choices = [
    df['tenure'] / 24,  # For two-year contracts
    df['tenure'] / 12,  # For one-year contracts
    df['tenure']         # Monthly stays as-is
]
df['tenure_ratio'] = np.select(conditions, choices, default=np.nan)

# Flag high-risk customers (e.g., those with >$70 monthly charges and month-to-month contracts).
df["Flagged"] = np.where(
    (df['Contract'] == 'Month-to-month') & (df['MonthlyCharges'] > 70),  # Condition
    "Yes",  # Value if True
    "No"    # Value if False
)

# Aggregations:
# Group by PaymentMethod and compare churn rates.
# IMPROVED: Directly calculate churn rates
churn_by_payment = df.groupby("PaymentMethod")["Churn"].apply(
    lambda x: (x == 'Yes').mean() * 100
).sort_values(ascending=False).reset_index(name='Churn Rate (%)')

# Calculate average monthly charges for churned vs. retained customers.
avg_charges = df.groupby('Churn')['MonthlyCharges'].mean()
print(avg_charges)

# Visualization:
# Bar plot of churn rates by contract type.
# 1. Prepare the data
churn_counts = df[df['Churn'] == 'Yes'].groupby('Contract').size().reset_index(name='Churn Count')

# 2. Create the plot
plt.figure(figsize=(10, 6))  # Set figure size

# Create bar chart with custom colors
bars = plt.bar(churn_counts['Contract'], 
               churn_counts['Churn Count'],
               color=['#1f77b4', '#ff7f0e', '#2ca02c'])  # Blue, orange, green

# 3. Customize the chart
plt.title('Churn Count by Contract Type', fontsize=16, pad=20)
plt.xlabel('Contract Type', fontsize=12)
plt.ylabel('Number of Churned Customers', fontsize=12)
plt.xticks(fontsize=11)
plt.grid(axis='y', alpha=0.3)

# Add data labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., 
             height + 3,  # Position above bar
             f'{height}', 
             ha='center', 
             va='bottom',
             fontsize=11)

# 4. Show or save the plot
plt.tight_layout()
plt.show()

# Kernel Density Estimate (KDE) plot of tenure for churned customers.
# Filter churned customers
churned_df = df[df['Churn'] == 'Yes']

# FIXED: Correct KDE plotting method
plt.figure(figsize=(10, 6))
churned_df['tenure'].plot.kde(color='red', linewidth=2)

# Customizations
plt.fill_betweenx(
    churned_df['tenure'].plot.kde().get_lines()[0].get_ydata(),
    churned_df['tenure'].plot.kde().get_lines()[0].get_xdata(),
    color='red', 
    alpha=0.3
)
plt.title('Tenure Distribution of Churned Customers')
plt.xlabel('Tenure (Months)')
plt.grid(True, alpha=0.3)
plt.show()

# Advanced:
# Use pd.crosstab() to analyze churn vs. InternetService type.
# 1. Create the cross-tabulation
cross_tab = pd.crosstab(
    index=df['InternetService'],
    columns=df['Churn'],
    margins=True,          # Add "All" row/column
    margins_name="Total"   # Name for the margins
)

# 2. Add percentage calculations
# Churn rate per Internet Service type
cross_tab['Churn Rate (%)'] = (cross_tab['Yes'] / cross_tab['Total'] * 100).round(1)

# Within category distribution
cross_tab_percentage = pd.crosstab(
    index=df['InternetService'],
    columns=df['Churn'],
    normalize='index'      # Row-wise percentages
).round(4) * 100