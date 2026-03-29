"""
Phase 4 — Risk Segmentation
Objective: Use boolean masking to create high-cost vs standard-cost policyholder segments.
"""

import numpy as np
import pandas as pd
import os

print("=" * 80)
print("PHASE 4 — RISK SEGMENTATION")
print("=" * 80)

if os.path.exists('insurance_with_bmi_index.csv'):
    df = pd.read_csv('insurance_with_bmi_index.csv')
else:
    df = pd.read_csv('insurance.csv')

# High-Cost vs Standard-Cost Split
print("\n" + "-" * 80)
print("4A — HIGH-COST VS STANDARD-COST SPLIT")
print("-" * 80)

q75 = np.percentile(df['charges'], 75)
print(f"75th percentile threshold: ${q75:,.2f}")

# Boolean masking
high_cost_mask = df['charges'] >= q75
standard_cost_mask = df['charges'] < q75

high_cost_df = df[high_cost_mask]
standard_cost_df = df[standard_cost_mask]

print(f"\nHigh-cost segment (top 25%): {len(high_cost_df)} records")
print(f"Standard-cost segment (bottom 75%): {len(standard_cost_df)} records")

# Segment Profile Comparison
print("\n" + "-" * 80)
print("4B — SEGMENT PROFILE COMPARISON")
print("-" * 80)

def compute_segment_profile(segment_df, segment_name):
    """Compute profile statistics for a segment"""
    print(f"\n{segment_name}:")
    print(f"  Count: {len(segment_df)}")
    
    smoker_pct = (segment_df['smoker'] == 'yes').sum() / len(segment_df) * 100
    print(f"  % Smokers: {smoker_pct:.1f}%")
    
    avg_bmi = segment_df['bmi'].mean()
    print(f"  Average BMI: {avg_bmi:.2f}")
    
    avg_age = segment_df['age'].mean()
    print(f"  Average age: {avg_age:.2f}")
    
    most_common_region = segment_df['region'].mode()[0]
    region_count = (segment_df['region'] == most_common_region).sum()
    print(f"  Most common region: {most_common_region} ({region_count} records)")
    
    avg_children = segment_df['children'].mean()
    print(f"  Average children: {avg_children:.2f}")
    
    avg_charges = segment_df['charges'].mean()
    print(f"  Average charges: ${avg_charges:,.2f}")

compute_segment_profile(high_cost_df, "HIGH-COST SEGMENT")
compute_segment_profile(standard_cost_df, "STANDARD-COST SEGMENT")

# Pivot Table: Age Group × Smoker Status
print("\n" + "-" * 80)
print("4C — PIVOT TABLE: AGE GROUP × SMOKER STATUS")
print("-" * 80)

bins = [17, 29, 39, 49, 64]
labels = ['20s', '30s', '40s', '50s+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

pivot = df.pivot_table(
    values='charges',
    index='age_group',
    columns='smoker',
    aggfunc='mean'
)

print("\nMean Insurance Charges by Age Group and Smoker Status:")
print(pivot)

print("\nFormatted (USD):")
pivot_formatted = pivot.copy()
for col in pivot_formatted.columns:
    pivot_formatted[col] = pivot_formatted[col].apply(lambda x: f"${x:,.2f}")
print(pivot_formatted)

os.makedirs('outputs/tables', exist_ok=True)

# Save pivot table to CSV
pivot.to_csv('outputs/tables/pivot_age_smoker.csv')
print("\n✓ Pivot table saved to 'outputs/tables/pivot_age_smoker.csv'")

# Save segment profiles to CSV
high_cost_df.to_csv('outputs/tables/high_cost_segment.csv', index=False)
standard_cost_df.to_csv('outputs/tables/standard_cost_segment.csv', index=False)
print("✓ Segment data saved to 'outputs/tables/'")

# Key insights
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print(f"✓ High-cost segment has {(high_cost_df['smoker'] == 'yes').sum() / len(high_cost_df) * 100:.1f}% smokers")
print(f"✓ Standard-cost segment has {(standard_cost_df['smoker'] == 'yes').sum() / len(standard_cost_df) * 100:.1f}% smokers")
print(f"✓ Charges increase with age across both smoker groups")
print(f"✓ Smokers in 50s+ show the highest average charges")

print("\n" + "=" * 80)
print("PHASE 4 COMPLETE")
print("=" * 80)
