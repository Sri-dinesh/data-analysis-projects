# Smart Insurance Premium Data Analysis
> AI-Based Actuarial Risk Modeling | Python for Data Analysis - Course-End Project

---

## Overview

This project performs end-to-end analysis of a real-world health insurance dataset to identify the strongest cost drivers, segment policyholders by risk, and produce a clean feature dataset ready for AI-based premium prediction and actuarial risk classification.

---

## Dataset

| Property | Detail |
|---|---|
| Source | [Kaggle - Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) |
| File | `insurance.csv` |
| Records | 1,338 policyholders |
| Features | `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges` |
| Target | `charges` — Annual insurance premium (USD) |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **NumPy** | Aggregations, boolean masking, broadcasting |
| **Pandas** | Data loading, pivot tables, feature engineering |
| **Matplotlib** | All visualizations |
| **AI (Claude)** | Actuarial insight generation and report writing |

---

## Project Structure

```
insurance_analysis/
│
├── insurance.csv                  # Raw dataset
├── README.md                      # This file
│
├── 01_data_loading.py
├── 02_eda_distributions.py
├── 03_numpy_aggregations.py
├── 04_segmentation.py
├── 05_visualizations.py
├── 06_feature_engineering.py
├── 07_ai_insights.py
│
└── outputs/
    ├── figures/                   # Saved chart PNGs
    ├── tables/                    # Pivot tables and feature matrix CSV
    └── reports/                   # AI-generated actuarial reports
```

---

## What's Covered

**Phase 1 - Data Loading & Validation**
Load the CSV, confirm zero missing values, audit data types, review descriptive statistics.

**Phase 2 - EDA & Distributions**
Explore every feature - age bands, BMI ranges, smoker split, regional breakdown, charge skewness.

**Phase 3 - NumPy Aggregations**
Mean, median, and std of charges by smoker status and region. BMI-adjusted charge index computed via NumPy broadcasting.

**Phase 4 - Risk Segmentation**
Boolean masking to split high-cost (top 25%) vs standard-cost policyholders. Pivot table of average charges by age group × smoker status.

**Phase 5 - Visualizations**
- Scatter plot: BMI vs Charges, colored by smoker, sized by children
- Histogram: Charge distribution for smokers vs non-smokers
- Bar chart: Average charges by region and number of dependents
- Heatmap: Mean charges by age group × smoker status

**Phase 6 - Feature Engineering**
Encode categorical variables, create interaction features (`bmi × smoker`, `age²`, `age × smoker`), export a model-ready feature matrix CSV.

**Phase 7 - AI-Assisted Insights**
Five AI-generated reports: top cost drivers, actuarial risk profiling, interaction feature recommendations, wellness incentive program proposal, and premium pricing fairness analysis.

---

## Key Findings

- Smokers cost **3.8× more** on average than non-smokers ($32,050 vs $8,434)
- **Smoker status** is the single strongest cost driver - outweighing age and BMI combined
- **Southeast region** carries the highest average charges across all dependent counts
- The `bmi × smoker` interaction is the most predictive engineered feature for a regression model

---

## How to Run

```bash
# Install dependencies
pip install numpy pandas matplotlib

# Run phases in order
python 01_data_loading.py
python 02_eda_distributions.py
python 03_numpy_aggregations.py
python 04_segmentation.py
python 05_visualizations.py
python 06_feature_engineering.py
python 07_ai_insights.py
```

All outputs are saved automatically to the `outputs/` directory.
