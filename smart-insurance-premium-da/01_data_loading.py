"""
Phase 1 — Data Loading & Validation
Objective: Load the CSV, verify integrity, and understand the raw data shape.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches

print("=" * 80)
print("PHASE 1 — DATA LOADING & VALIDATION")
print("=" * 80)

df = pd.read_csv('insurance.csv')
print("\n✓ Dataset loaded from 'insurance.csv'")

# Null check
print("\n" + "-" * 80)
print("NULL CHECK")
print("-" * 80)
null_counts = df.isnull().sum()
print(null_counts)
print(f"\n Total missing values: {null_counts.sum()}")
if null_counts.sum() == 0:
    print(" CONFIRMED: No missing values across all 7 columns")

# Data types audit
print("\n" + "-" * 80)
print("DATA TYPES AUDIT")
print("-" * 80)
print(df.dtypes)

print("\n" + "-" * 80)
print("DATASET SHAPE & PREVIEW")
print("-" * 80)
print(f"Shape: {df.shape}")
print(f" CONFIRMED: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nFirst 10 rows:")
print(df.head(10))

# Unique value counts for categorical columns
print("\n" + "-" * 80)
print("UNIQUE VALUE COUNTS")
print("-" * 80)
print(f"\nSex: {df['sex'].unique()} → {df['sex'].nunique()} unique values")
print(df['sex'].value_counts())

print(f"\nSmoker: {df['smoker'].unique()} → {df['smoker'].nunique()} unique values")
print(df['smoker'].value_counts())

print(f"\nRegion: {df['region'].unique()} → {df['region'].nunique()} unique values")
print(df['region'].value_counts())

print(f"\nChildren: {sorted(df['children'].unique())} → {df['children'].nunique()} unique values")
print(df['children'].value_counts().sort_index())

# Descriptive statistics
print("\n" + "-" * 80)
print("DESCRIPTIVE STATISTICS (NUMERIC COLUMNS)")
print("-" * 80)
print(df.describe())

print("\n" + "=" * 80)
print("PHASE 1 COMPLETE")
print("=" * 80)
print(f" No missing values")
print(f" {df.shape[0]} rows × {df.shape[1]} columns")
print(f" Mean charges: ${df['charges'].mean():,.2f}")
print(f" Max charges: ${df['charges'].max():,.2f}")
