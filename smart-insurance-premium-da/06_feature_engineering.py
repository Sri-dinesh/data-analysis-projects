"""
Phase 6 — Feature Engineering (Clean Dataset for ML)
Objective: Produce a fully numeric, model-ready DataFrame.
"""

import numpy as np
import pandas as pd
import os

print("=" * 80)
print("PHASE 6 — FEATURE ENGINEERING")
print("=" * 80)

if os.path.exists('insurance_with_bmi_index.csv'):
    df = pd.read_csv('insurance_with_bmi_index.csv')
    print("✓ Loaded data with BMI charge index from Phase 3")
else:
    df = pd.read_csv('insurance.csv')
    bmi_arr = df['bmi'].to_numpy()
    charges_arr = df['charges'].to_numpy()
    bmi_mean = np.mean(bmi_arr)
    df['bmi_charge_index'] = charges_arr * (bmi_arr / bmi_mean)
    print("✓ Recreated BMI charge index")

# Encode Categorical Variables
print("\n" + "-" * 80)
print("6A — ENCODE CATEGORICAL VARIABLES")
print("-" * 80)

# Binary encoding for smoker (1 = smoker, 0 = non-smoker)
df['smoker_encoded'] = (df['smoker'] == 'yes').astype(int)
print(f"✓ Smoker encoded: yes=1, no=0")
print(f"  Smokers: {df['smoker_encoded'].sum()}")
print(f"  Non-smokers: {(df['smoker_encoded'] == 0).sum()}")

# Binary encoding for sex (1 = male, 0 = female)
df['sex_encoded'] = (df['sex'] == 'male').astype(int)
print(f"\n✓ Sex encoded: male=1, female=0")
print(f"  Males: {df['sex_encoded'].sum()}")
print(f"  Females: {(df['sex_encoded'] == 0).sum()}")

# One-hot encoding for region (drop first to avoid multicollinearity)
region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)
df = pd.concat([df, region_dummies], axis=1)
print(f"\n✓ Region one-hot encoded (dropped 'northeast' as reference):")
print(f"  Columns created: {list(region_dummies.columns)}")

# Interaction Features
print("\n" + "-" * 80)
print("6B — INTERACTION FEATURES")
print("-" * 80)

# BMI × Smoker (key non-linear interaction)
df['bmi_smoker'] = df['bmi'] * df['smoker_encoded']
print("✓ Created: bmi_smoker = bmi × smoker_encoded")

# Age × Smoker
df['age_smoker'] = df['age'] * df['smoker_encoded']
print("✓ Created: age_smoker = age × smoker_encoded")

# Age² (non-linear aging effect)
df['age_squared'] = df['age'] ** 2
print("✓ Created: age_squared = age²")

# BMI² (obesity penalty is non-linear)
df['bmi_squared'] = df['bmi'] ** 2
print("✓ Created: bmi_squared = bmi²")

# Children × Age (older parents with children = higher risk)
df['children_age'] = df['children'] * df['age']
print("✓ Created: children_age = children × age")

print("\n" + "-" * 80)
print("6C — FINAL FEATURE MATRIX")
print("-" * 80)

feature_cols = [
    'age', 'age_squared', 'bmi', 'bmi_squared',
    'children', 'smoker_encoded', 'sex_encoded',
    'bmi_smoker', 'age_smoker', 'children_age',
    'bmi_charge_index',
    'region_northwest', 'region_southeast', 'region_southwest'
]

X = df[feature_cols]
y = df['charges']

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"\nFeature columns ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")

os.makedirs('outputs/tables', exist_ok=True)

X.to_csv('outputs/tables/feature_matrix.csv', index=False)
print(f"\n✓ Feature matrix saved to 'outputs/tables/feature_matrix.csv'")

feature_data = X.copy()
feature_data['charges'] = y
feature_data.to_csv('outputs/tables/feature_matrix_with_target.csv', index=False)
print(f"✓ Feature matrix with target saved to 'outputs/tables/feature_matrix_with_target.csv'")

print("\n" + "-" * 80)
print("FEATURE MATRIX SUMMARY STATISTICS")
print("-" * 80)
print(X.describe())

print("\n" + "=" * 80)
print("PHASE 6 COMPLETE")
print("=" * 80)
print(f"✓ {len(feature_cols)} features engineered")
print(f"✓ All categorical variables encoded")
print(f"✓ 5 interaction features created")
print(f"✓ Model-ready dataset saved")
