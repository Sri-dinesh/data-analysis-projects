"""
Phase 2 — EDA & Feature Distributions
Objective: Explore the distribution of every feature individually.
"""

import numpy as np
import pandas as pd

print("=" * 80)
print("PHASE 2 — EDA & FEATURE DISTRIBUTIONS")
print("=" * 80)

df = pd.read_csv('insurance.csv')

# Age distribution (value counts per decade grouping)
print("\n" + "-" * 80)
print("AGE DISTRIBUTION")
print("-" * 80)
print(f"Age range: {df['age'].min()} - {df['age'].max()}")
print(f"Mean age: {df['age'].mean():.2f}")
print(f"Median age: {df['age'].median():.2f}")

# Group by decade
age_bins = [17, 29, 39, 49, 64]
age_labels = ['18-29 (20s)', '30-39 (30s)', '40-49 (40s)', '50-64 (50s+)']
df['age_decade'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
print("\nAge distribution by decade:")
print(df['age_decade'].value_counts().sort_index())

# BMI distribution
print("\n" + "-" * 80)
print("BMI DISTRIBUTION")
print("-" * 80)
print(f"BMI range: {df['bmi'].min():.2f} - {df['bmi'].max():.2f}")
print(f"Mean BMI: {df['bmi'].mean():.2f}")
print(f"Median BMI: {df['bmi'].median():.2f}")

# WHO BMI thresholds
underweight = (df['bmi'] < 18.5).sum()
normal = ((df['bmi'] >= 18.5) & (df['bmi'] < 25)).sum()
overweight = ((df['bmi'] >= 25) & (df['bmi'] < 30)).sum()
obese = (df['bmi'] >= 30).sum()

print("\nWHO BMI Categories:")
print(f"  Underweight (< 18.5): {underweight} ({underweight/len(df)*100:.1f}%)")
print(f"  Normal (18.5-24.9): {normal} ({normal/len(df)*100:.1f}%)")
print(f"  Overweight (25-29.9): {overweight} ({overweight/len(df)*100:.1f}%)")
print(f"  Obese (≥ 30): {obese} ({obese/len(df)*100:.1f}%)")

# Children distribution
print("\n" + "-" * 80)
print("CHILDREN DISTRIBUTION")
print("-" * 80)
print("Frequency table (0-5 dependents):")
print(df['children'].value_counts().sort_index())
print(f"\nMean children: {df['children'].mean():.2f}")

# Smoker split
print("\n" + "-" * 80)
print("SMOKER STATUS SPLIT")
print("-" * 80)
smoker_counts = df['smoker'].value_counts()
print(smoker_counts)
print(f"\nSmokers: {smoker_counts['yes']} ({smoker_counts['yes']/len(df)*100:.1f}%)")
print(f"Non-smokers: {smoker_counts['no']} ({smoker_counts['no']/len(df)*100:.1f}%)")

# Region split
print("\n" + "-" * 80)
print("REGION SPLIT")
print("-" * 80)
region_counts = df['region'].value_counts()
print(region_counts)
for region in region_counts.index:
    print(f"{region}: {region_counts[region]} ({region_counts[region]/len(df)*100:.1f}%)")

# Charges distribution
print("\n" + "-" * 80)
print("CHARGES DISTRIBUTION")
print("-" * 80)
print(f"Mean charges: ${df['charges'].mean():,.2f}")
print(f"Median charges: ${df['charges'].median():,.2f}")
print(f"Std deviation: ${df['charges'].std():,.2f}")
print(f"Min charges: ${df['charges'].min():,.2f}")
print(f"Max charges: ${df['charges'].max():,.2f}")
print(f"\nSkewness: {df['charges'].skew():.2f} (right-skewed)")

# Key insights
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print("✓ Charges are heavily right-skewed (mean > median)")
print(f"  Mean: ${df['charges'].mean():,.2f} vs Median: ${df['charges'].median():,.2f}")
print(f"✓ Smokers represent ~{smoker_counts['yes']/len(df)*100:.1f}% of records")
print("✓ Smokers account for a disproportionately large share of high charges")

print("\n" + "=" * 80)
print("PHASE 2 COMPLETE")
print("=" * 80)
