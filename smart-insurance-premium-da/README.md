# Smart Insurance Premium Data Analysis

AI-powered actuarial analysis and interactive dashboard for health insurance premium data.

## Overview

Analyzes health insurance premiums across 1,338 policyholders to identify cost drivers, segment risk profiles, and generate actionable actuarial insights. Combines data analysis with AI-powered reporting and an interactive Streamlit dashboard.

## Features

### Data Analysis Pipeline

Seven-phase automated analysis:

- Phase 1: Data loading and validation
- Phase 2: Exploratory data analysis with distributions
- Phase 3: NumPy aggregations (mean, median, std by segment)
- Phase 4: Risk segmentation (high-cost vs standard)
- Phase 5: Matplotlib visualizations (4 charts)
- Phase 6: Feature engineering (encoding, interactions, BMI index)
- Phase 7: AI-assisted insights generation

### Interactive Dashboard

- Portfolio Overview: Key metrics and KPIs
- Analytics: Pre-generated charts and interactive Plotly visualizations
- Insights: Three AI-generated actuarial reports
- AI Assistant: Query-based chatbot with Gemini AI integration
- Raw Data: Filtered data export capability

### Key Findings

- Smoking carries 3.8x premium multiplier ($32,050 vs $8,434)
- Age drives exponential charge increases across lifecycle
- BMI amplifies smoking risk (40%+ uplift for smokers with BMI > 30)
- Regional variation minimal compared to lifestyle factors

## Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configure Gemini AI (Optional)

Set your Google API key to enable AI features.

**Windows (PowerShell):**

```powershell
$env:GOOGLE_API_KEY = "your-api-key"
```

**Windows (CMD):**

```cmd
set GOOGLE_API_KEY=your-api-key
```

**macOS/Linux:**

```bash
export GOOGLE_API_KEY="your-api-key"
```

Or create a .env file:

```
GOOGLE_API_KEY=your-api-key
```

## Usage

### Run Everything (Recommended)

Executes all analysis phases and launches the dashboard in one command:

```bash
python main.py
```

This will:

1. Run all 7 analysis phases automatically
2. Generate reports and visualizations
3. Launch Streamlit dashboard at http://localhost:8501

Pipeline runs only once per session. Navigate between dashboard pages without rerunning analysis.

### Run Individual Phases

To run specific analysis phases:

```bash
python 01_data_loading.py
python 02_eda_distributions.py
python 03_numpy_aggregations.py
python 04_segmentation.py
python 05_visualizations.py
python 06_feature_engineering.py
python 07_ai_insights.py
```

## Output

All generated outputs are saved to the outputs/ directory.

### Reports (Markdown)

- report_cost_drivers.md - Top factors driving insurance costs
- report_risk_profiling.md - Actuarial risk classification
- report_business_intelligence.md - Strategic insights

### Visualizations (PNG)

- scatter_bmi_charges.png - BMI vs charges correlation
- histogram_charges_smoker.png - Charge distribution
- heatmap_agegroup_smoker.png - Age-group analysis
- bar_region_dependents.png - Regional patterns

### Data Tables (CSV)

- feature_matrix_with_target.csv - ML-ready dataset
- feature_matrix.csv - Features only
- high_cost_segment.csv - Top 25% policyholders
- standard_cost_segment.csv - Bottom 75% policyholders
- pivot_age_smoker.csv - Cross-tabulation

## Dashboard Pages

### Dashboard

Key portfolio metrics, findings, and visualizations with interactive filters for smoker status, age range, and region.

### Insights

Three comprehensive AI-generated actuarial reports providing detailed analysis of cost drivers, risk profiling, and business intelligence.

### AI Assistant

Interactive chatbot for asking questions about premiums, risk factors, wellness programs, and actuarial topics. Powered by Gemini AI.

### Raw Data

View and export filtered dataset by smoker status, region, and record count as CSV.

## Dataset

1,338 health insurance records with the following fields:

- age: Policyholder age
- sex: Gender
- bmi: Body Mass Index
- children: Number of dependents
- smoker: Smoking status (yes/no)
- region: Geographic region
- charges: Annual insurance premium (USD)

Data contains no missing values.

## Technical Details

### NumPy Operations

- Aggregation functions (mean, median, std)
- Boolean masking for segmentation
- Array broadcasting for index computation
- Percentile calculations

### Pandas Operations

- CSV data loading and validation
- Column binning (age groups)
- Pivot tables and cross-tabulations
- Grouped aggregations
- Categorical encoding
- Feature matrix assembly

### Feature Engineering

- Binary encoding (smoker, sex)
- One-hot encoding (region)
- BMI charge index (charges × BMI / mean_BMI)
- Interaction features (smoker × BMI, age interactions)
- Feature normalization and scaling

## Troubleshooting

### Data file not found

Ensure insurance.csv exists in project root with correct columns: age, sex, bmi, children, smoker, region, charges.

### Gemini API errors

- Verify GOOGLE_API_KEY is set correctly
- Check internet connectivity
- Validate API key in Google Cloud Console
- Monitor usage quotas

### Missing visualizations

Run Phase 5 individually: python 05_visualizations.py

### Streamlit issues

- Restart the process (Ctrl+C then rerun)
- Clear cache: streamlit cache clear
- Verify dependencies in requirements.txt

## Performance

The dashboard uses Streamlit's caching and session state to optimize performance:

- Analysis pipeline runs once per session (cached in session state)
- Data is cached for instant page navigation
- Charts are pre-computed and saved
- No redundant computation during navigation

## Version

Status: Production Ready
Last Updated: April 2026
