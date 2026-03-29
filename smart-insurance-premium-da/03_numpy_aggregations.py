"""
Phase 3 — NumPy Aggregations
Objective: Use raw NumPy operations to compute key statistics.
"""

import numpy as np
import pandas as pd

print("=" * 80)
print("PHASE 3 — NUMPY AGGREGATIONS")
print("=" * 80)

df = pd.read_csv('insurance.csv')

# Aggregations by Smoker Status
print("\n" + "-" * 80)
print("3A — AGGREGATIONS BY SMOKER STATUS")
print("-" * 80)

# Extract charges arrays via boolean masking
smoker_charges = df[df['smoker'] == 'yes']['charges'].to_numpy()
nonsmoker_charges = df[df['smoker'] == 'no']['charges'].to_numpy()

print(f"Smoker charges array shape: {smoker_charges.shape}")
print(f"Non-smoker charges array shape: {nonsmoker_charges.shape}")

smoker_mean = np.mean(smoker_charges)
smoker_median = np.median(smoker_charges)
smoker_std = np.std(smoker_charges)

nonsmoker_mean = np.mean(nonsmoker_charges)
nonsmoker_median = np.median(nonsmoker_charges)
nonsmoker_std = np.std(nonsmoker_charges)

print("\nSMOKERS:")
print(f"  Mean:   ${smoker_mean:,.2f}")
print(f"  Median: ${smoker_median:,.2f}")
print(f"  Std:    ${smoker_std:,.2f}")

print("\nNON-SMOKERS:")
print(f"  Mean:   ${nonsmoker_mean:,.2f}")
print(f"  Median: ${nonsmoker_median:,.2f}")
print(f"  Std:    ${nonsmoker_std:,.2f}")

print(f"\n✓ Smoker mean (${smoker_mean:,.2f}) is {smoker_mean/nonsmoker_mean:.2f}× higher than non-smoker mean (${nonsmoker_mean:,.2f})")

# Aggregations by Region
print("\n" + "-" * 80)
print("3B — AGGREGATIONS BY REGION")
print("-" * 80)

regions = ['northeast', 'northwest', 'southeast', 'southwest']
for region in regions:
    region_charges = df[df['region'] == region]['charges'].to_numpy()
    region_mean = np.mean(region_charges)
    region_median = np.median(region_charges)
    region_std = np.std(region_charges)
    
    print(f"\n{region.upper()}:")
    print(f"  Count:  {len(region_charges)}")
    print(f"  Mean:   ${region_mean:,.2f}")
    print(f"  Median: ${region_median:,.2f}")
    print(f"  Std:    ${region_std:,.2f}")

# BMI-Adjusted Charge Index (NumPy Broadcasting)
print("\n" + "-" * 80)
print("3C — BMI-ADJUSTED CHARGE INDEX (NUMPY BROADCASTING)")
print("-" * 80)

bmi_arr = df['bmi'].to_numpy()          # shape (1338,)
charges_arr = df['charges'].to_numpy()  # shape (1338,)

print(f"BMI array shape: {bmi_arr.shape}")
print(f"Charges array shape: {charges_arr.shape}")

bmi_mean = np.mean(bmi_arr)
print(f"Mean BMI: {bmi_mean:.2f}")

# Broadcasting: scale each record's charge by how far BMI deviates from mean
bmi_adjusted_index = charges_arr * (bmi_arr / bmi_mean)  # shape (1338,)

print(f"BMI-adjusted index shape: {bmi_adjusted_index.shape}")
print(f"BMI-adjusted index range: ${np.min(bmi_adjusted_index):,.2f} - ${np.max(bmi_adjusted_index):,.2f}")

# Add to DataFrame
df['bmi_charge_index'] = bmi_adjusted_index
print("\n✓ BMI-adjusted charge index added to DataFrame as 'bmi_charge_index' column")

# Percentile Thresholds
print("\n" + "-" * 80)
print("3D — PERCENTILE THRESHOLDS")
print("-" * 80)

q25 = np.percentile(charges_arr, 25)  # Low-cost threshold
q50 = np.percentile(charges_arr, 50)  # Median
q75 = np.percentile(charges_arr, 75)  # High-cost threshold

print(f"25th percentile (Q1): ${q25:,.2f}")
print(f"50th percentile (Q2/Median): ${q50:,.2f}")
print(f"75th percentile (Q3): ${q75:,.2f}")
print(f"IQR (Q3 - Q1): ${q75 - q25:,.2f}")

print("\n" + "=" * 80)
print("PHASE 3 COMPLETE")
print("=" * 80)
print(f"✓ Smoker vs non-smoker aggregations computed")
print(f"✓ Regional aggregations computed for all 4 regions")
print(f"✓ BMI-adjusted charge index created using NumPy broadcasting")
print(f"✓ Percentile thresholds calculated")

# Save the DataFrame with the new column for use in later phases
df.to_csv('insurance_with_bmi_index.csv', index=False)
print(f"\n✓ Updated DataFrame saved to 'insurance_with_bmi_index.csv'")
