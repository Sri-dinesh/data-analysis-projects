"""
Phase 5 — Visualizations
Objective: Build 4 publication-quality Matplotlib charts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

print("=" * 80)
print("PHASE 5 — VISUALIZATIONS")
print("=" * 80)

df = pd.read_csv('insurance.csv')

os.makedirs('outputs/figures', exist_ok=True)

# Scatter Plot - BMI vs Charges
print("\n" + "-" * 80)
print("CHART 1 — SCATTER PLOT: BMI vs CHARGES")
print("-" * 80)

fig, ax = plt.subplots(figsize=(12, 8))

# Separate smokers and non-smokers
smokers = df[df['smoker'] == 'yes']
nonsmokers = df[df['smoker'] == 'no']

size_scale = 50
smoker_sizes = (smokers['children'] + 1) * size_scale
nonsmoker_sizes = (nonsmokers['children'] + 1) * size_scale

scatter1 = ax.scatter(nonsmokers['bmi'], nonsmokers['charges'], 
                     c='blue', alpha=0.6, s=nonsmoker_sizes, 
                     label='Non-smoker', edgecolors='black', linewidth=0.5)
scatter2 = ax.scatter(smokers['bmi'], smokers['charges'], 
                     c='red', alpha=0.6, s=smoker_sizes, 
                     label='Smoker', edgecolors='black', linewidth=0.5)

# Add vertical line at BMI = 30 (obese threshold)
ax.axvline(x=30, color='green', linestyle='--', linewidth=2, label='BMI = 30 (Obese threshold)')

ax.set_xlabel('BMI', fontsize=12)
ax.set_ylabel('Insurance Charges (USD)', fontsize=12)
ax.set_title('BMI vs Insurance Charges — Colored by Smoker Status, Sized by Children', 
             fontsize=14, fontweight='bold')

ax.legend(loc='upper left', fontsize=10)

from matplotlib.lines import Line2D
size_legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
           markersize=np.sqrt(1*size_scale)/2, label='0 children'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
           markersize=np.sqrt(3*size_scale)/2, label='2 children'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
           markersize=np.sqrt(6*size_scale)/2, label='5 children')
]
ax.legend(handles=ax.get_legend_handles_labels()[0] + size_legend_elements, 
         loc='upper left', fontsize=9)

ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/figures/scatter_bmi_charges.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/figures/scatter_bmi_charges.png")

# Histogram - Charges Distribution by Smoker Status
print("\n" + "-" * 80)
print("CHART 2 — HISTOGRAM: CHARGES DISTRIBUTION BY SMOKER STATUS")
print("-" * 80)

fig, ax = plt.subplots(figsize=(12, 8))

ax.hist(nonsmokers['charges'], bins=50, alpha=0.6, color='blue', label='Non-smoker')
ax.hist(smokers['charges'], bins=50, alpha=0.6, color='red', label='Smoker')

nonsmoker_mean = nonsmokers['charges'].mean()
smoker_mean = smokers['charges'].mean()

ax.axvline(nonsmoker_mean, color='darkblue', linestyle='--', linewidth=2)
ax.axvline(smoker_mean, color='darkred', linestyle='--', linewidth=2)

# Annotate means
ax.text(nonsmoker_mean, ax.get_ylim()[1]*0.9, 
        f'Non-smoker mean\n${nonsmoker_mean:,.0f}', 
        ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax.text(smoker_mean, ax.get_ylim()[1]*0.8, 
        f'Smoker mean\n${smoker_mean:,.0f}', 
        ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

ax.set_xlabel('Insurance Charges (USD)', fontsize=12)
ax.set_ylabel('Frequency (count)', fontsize=12)
ax.set_title('Insurance Charges Distribution — Smokers vs Non-Smokers', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/figures/histogram_charges_smoker.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/figures/histogram_charges_smoker.png")

# Bar Chart - Average Charges by Region & Number of Dependents
print("\n" + "-" * 80)
print("CHART 3 — BAR CHART: AVERAGE CHARGES BY REGION & DEPENDENTS")
print("-" * 80)

fig, ax = plt.subplots(figsize=(14, 8))

regions = ['northeast', 'northwest', 'southeast', 'southwest']
children_counts = sorted(df['children'].unique())
n_children = len(children_counts)
n_regions = len(regions)

# Calculate average charges for each region and children count
data = np.zeros((n_regions, n_children))
for i, region in enumerate(regions):
    for j, child_count in enumerate(children_counts):
        mask = (df['region'] == region) & (df['children'] == child_count)
        if mask.sum() > 0:
            data[i, j] = df[mask]['charges'].mean()

bar_width = 0.13
x = np.arange(n_regions)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for j in range(n_children):
    offset = (j - n_children/2 + 0.5) * bar_width
    bars = ax.bar(x + offset, data[:, j], bar_width, 
                  label=f'{children_counts[j]} children', color=colors[j])
    
    # Add value labels on top of bars
    for k, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${height:,.0f}',
                   ha='center', va='bottom', fontsize=7, rotation=0)

ax.set_xlabel('Region', fontsize=12)
ax.set_ylabel('Average Charges (USD)', fontsize=12)
ax.set_title('Average Insurance Charges by Region and Number of Dependents', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([r.capitalize() for r in regions])
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/figures/bar_region_dependents.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/figures/bar_region_dependents.png")

# Heatmap - Pivot Table (Age Group × Smoker)
print("\n" + "-" * 80)
print("CHART 4 — HEATMAP: AGE GROUP × SMOKER STATUS")
print("-" * 80)

bins = [17, 29, 39, 49, 64]
labels = ['20s', '30s', '40s', '50s+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

pivot = df.pivot_table(values='charges', index='age_group', columns='smoker', aggfunc='mean')

fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')

ax.set_xticks(np.arange(len(pivot.columns)))
ax.set_yticks(np.arange(len(pivot.index)))
ax.set_xticklabels(pivot.columns)
ax.set_yticklabels(pivot.index)

plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Mean Charges (USD)', rotation=270, labelpad=20, fontsize=11)

# Annotate each cell with the dollar value
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        value = pivot.values[i, j]
        text = ax.text(j, i, f'${value:,.0f}',
                      ha="center", va="center", color="black", fontsize=12, fontweight='bold')

ax.set_xlabel('Smoker Status', fontsize=12)
ax.set_ylabel('Age Group', fontsize=12)
ax.set_title('Mean Insurance Charges — Age Group × Smoker Status', 
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/figures/heatmap_agegroup_smoker.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/figures/heatmap_agegroup_smoker.png")

print("\n" + "=" * 80)
print("PHASE 5 COMPLETE")
print("=" * 80)
print("✓ All 4 charts generated and saved to outputs/figures/")
print("  1. scatter_bmi_charges.png")
print("  2. histogram_charges_smoker.png")
print("  3. bar_region_dependents.png")
print("  4. heatmap_agegroup_smoker.png")
