# 🎯 Smart Insurance Premium Data Analysis

> AI-Based Actuarial Risk Modeling | Python for Data Analysis + Gemini AI Integration

**Project Status:** Complete end-to-end pipeline with data analysis, AI insights, and interactive Streamlit application.

---

## 📋 Problem Statement

Analyze health insurance premium data covering age, BMI, smoking status, number of children, region, and annual charges. Identify the strongest cost drivers, compute risk segments, and build a clean feature dataset for AI-based insurance premium prediction and actuarial risk classification models.

### Core Analysis Objectives

✓ Load insurance CSV; verify no missing values; explore all feature distributions  
✓ NumPy aggregations: mean, median, std of insurance charges by smoker status and region  
✓ Boolean masking: segment high-cost (top 25%) vs standard-cost policyholders  
✓ NumPy broadcasting: compute BMI-adjusted charge index across all records  
✓ Pivot table: average charges by age group (20s/30s/40s/50s+) and smoker status  
✓ Scatter plot: BMI vs charges (colored by smoker, sized by number of children)  
✓ Histogram: insurance charges distribution for smokers vs non-smokers  
✓ Bar chart: average charges by region and number of dependents

### AI Integration

✓ AI Report: Top 3 factors driving insurance charge variation and why  
✓ Actuarial risk profiling report from age-group and smoker-status statistics  
✓ Feature interaction analysis (e.g., BMI × smoker) for predictive modeling  
✓ Wellness incentive program proposal based on lifestyle risk factors  
✓ Premium pricing fairness assessment from regional charge variation data

---

## 📂 Project Structure

```
smart-insurance-premium-da/
│
├── insurance.csv                          # Raw dataset (1,338 records)
├── insurance_with_bmi_index.csv          # Enhanced data with BMI charge index
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
│
├── Data Analysis Pipeline
├── 01_data_loading.py                    # Phase 1: Load & validate data
├── 02_eda_distributions.py               # Phase 2: Exploratory data analysis
├── 03_numpy_aggregations.py              # Phase 3: NumPy aggregations
├── 04_segmentation.py                    # Phase 4: Risk segmentation
├── 05_visualizations.py                  # Phase 5: Matplotlib visualizations
├── 06_feature_engineering.py             # Phase 6: Feature engineering
├── 07_ai_insights.py                     # Phase 7: AI-powered insights
├── run_all.py                            # Run entire pipeline
│
├── Application Layer
├── streamlit_app.py                      # Professional Streamlit UI
│
├── AI Module
├── ai/
│   ├── __init__.py
│   └── gemini_client.py                  # Gemini API integration
│
└── Generated Outputs
    ├── figures/
    │   ├── scatter_bmi_charges.png
    │   ├── histogram_charges_smoker.png
    │   ├── heatmap_agegroup_smoker.png
    │   └── bar_region_dependents.png
    ├── reports/
    │   ├── report_cost_drivers.md
    │   ├── report_risk_profiling.md
    │   └── report_business_intelligence.md
    └── tables/
        ├── feature_matrix_with_target.csv
        ├── feature_matrix.csv
        ├── high_cost_segment.csv
        └── pivot_age_smoker.csv
```

---

## 🗂️ Dataset

| Property         | Detail                                                                                           |
| ---------------- | ------------------------------------------------------------------------------------------------ |
| **Source**       | [Kaggle - Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) |
| **File**         | `insurance.csv`                                                                                  |
| **Records**      | 1,338 policyholders                                                                              |
| **Features**     | `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`                                   |
| **Target**       | `charges` — Annual insurance premium (USD)                                                       |
| **Data Quality** | No missing values ✓                                                                              |

---

## 🛠️ Tech Stack

| Component               | Technology         | Purpose                             |
| ----------------------- | ------------------ | ----------------------------------- |
| **Data Analysis**       | NumPy, Pandas      | Aggregations, masking, pivots       |
| **Visualization**       | Matplotlib, Plotly | Static (PNG) and interactive charts |
| **Application**         | Streamlit          | Interactive web dashboard           |
| **AI Integration**      | Gemini (Google)    | Actuarial insight generation        |
| **Feature Engineering** | Pandas, NumPy      | Encoding, interactions, BMI index   |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- `pip` package manager
- (Optional) Google API Key for Gemini AI features

### 1. Environment Setup

```bash
# Clone/navigate to project directory
cd smart-insurance-premium-da

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure AI (Optional)

To enable Gemini AI features, set the `GOOGLE_API_KEY` environment variable:

**Windows (PowerShell):**

```powershell
$env:GOOGLE_API_KEY = "your-api-key-here"
python -m streamlit run streamlit_app.py
```

**Windows (CMD):**

```cmd
set GOOGLE_API_KEY=your-api-key-here
python -m streamlit run streamlit_app.py
```

**macOS/Linux:**

```bash
export GOOGLE_API_KEY="your-api-key-here"
python -m streamlit run streamlit_app.py
```

**Or create a `.env` file:**

```
GOOGLE_API_KEY=your-api-key-here
```

### 3. Run the Analysis Pipeline

Generate all data analysis outputs and reports:

```bash
python run_all.py
```

This executes all 7 phases in sequence:

- Phase 1: Data loading and validation
- Phase 2: Exploratory data analysis
- Phase 3: NumPy aggregations
- Phase 4: Segmentation analysis
- Phase 5: Visualizations
- Phase 6: Feature engineering
- Phase 7: AI-assisted insights

**Output:** Generated files in `outputs/` directory

### 4. Launch Streamlit Application

```bash
streamlit run streamlit_app.py
```

**Access:** Open browser to `http://localhost:8501`

---

## 📊 Features

### Dashboard View

- **Portfolio Overview:** Key metrics (1,338 policies, mean charges, smoker %, risk multiplier)
- **Key Findings:** Smoker impact and segmentation opportunities
- **Analysis Visualizations:** 4 pre-generated matplotlib charts
- **Interactive Charts:** Dynamic Plotly charts with filtering
  - Age vs Charges scatter plot
  - Age group comparison bar chart

### Insights View

- **AI-Generated Reports** (3 comprehensive reports):
  1. **Top Cost Drivers:** Smoker status (3.8×), age (2.2×), BMI (20-40%)
  2. **Risk Profiling:** 4 actuarial tiers from low to very high risk
  3. **Business Intelligence:** Portfolio metrics, regional analysis, wellness ROI

### AI Assistant

- **Gemini-Powered Chatbot:** Ask questions about:
  - Cost drivers and premium variation
  - Risk profiling and segmentation
  - Actuarial best practices
  - Wellness program design
  - Pricing fairness and compliance
- **Fallback:** Local responses when API unavailable
- **Context-Aware:** Uses real portfolio statistics

### Raw Data View

- **Filtered Dataset:** By smoker status, region, and record count
- **Data Export:** Download filtered data as CSV

---

## 📈 Key Findings

### 1. Smoker Status Dominance

- **Impact:** 3.8× premium multiplier
- **Average smoker cost:** $32,050
- **Average non-smoker cost:** $8,434
- **Portfolio impact:** 20% of policyholders (smokers) generate ~50% of charges

### 2. Age-Related Progression

- **20s:** ~$9,500 average
- **30s:** ~$12,500 average
- **40s:** ~$15,000 average
- **50s+:** ~$21,000 average
- **Mechanism:** Exponential increase in chronic disease prevalence

### 3. BMI Interaction Effects

- **Non-smokers:** 20% premium uplift for BMI > 30
- **Smokers:** 40%+ premium uplift for BMI > 30
- **Implication:** BMI amplifies smoking risk

### 4. Regional Variations

- Minor regional differences (~10%)
- Primary drivers remain age and smoking regardless of geography
- Note: Region not a primary cost driver factor

---

## 🎯 Analysis Outputs

### Generated Reports (/outputs/reports/)

1. **report_cost_drivers.md** — Detailed actuarial explanation of top 3 factors
2. **report_risk_profiling.md** — 4-tier risk classification with implications
3. **report_business_intelligence.md** — Portfolio analytics and strategic insights

### Generated Visualizations (/outputs/figures/)

- `scatter_bmi_charges.png` — BMI vs charges with smoker coloring
- `histogram_charges_smoker.png` — Distribution comparison
- `heatmap_agegroup_smoker.png` — Age-group × smoker-status matrix
- `bar_region_dependents.png` — Regional and dependent analysis

### Generated Data Tables (/outputs/tables/)

- `feature_matrix_with_target.csv` — ML-ready dataset
- `feature_matrix.csv` — Cleaned features only
- `high_cost_segment.csv` — Top 25% policyholders
- `standard_cost_segment.csv` — Bottom 75% policyholders
- `pivot_age_smoker.csv` — Age-group × smoker pivot

---

## 💡 Use Cases

### 1. Premium Pricing Model

- Evidence-based factors for justifying premium differentiation
- 3.8× smoker multiplier backed by actuarial analysis
- Age-based progression justified by claim frequency data

### 2. Risk Management

- Identify high-risk segments (smokers age 50+)
- Segment portfolio for targeted interventions
- Wellness program ROI projections

### 3. Product Development

- Smoking cessation discount programs
- BMI-based wellness incentives
- Age-appropriate preventive care offerings

### 4. Fairness & Compliance

- Evidence of non-discriminatory factors
- Geographic neutrality demonstrated
- Actuarially justified pricing structure

---

## 🔧 Configuration

### Environment Variables

```bash
# Required for AI features
GOOGLE_API_KEY=sk-...     # Your Gemini API key

# Optional
GEMINI_MODEL=gemini-2.5-flash-lite  # Default model
```

### Streamlit Configuration

Edit `.streamlit/config.toml` to customize:

```toml
[theme]
primaryColor="#007AFF"
backgroundColor="#FAFAFA"
secondaryBackgroundColor="#F2F2F7"
textColor="#1C1C1E"
```

---

## 🐛 Troubleshooting

### "No data file found"

- Ensure `insurance.csv` exists in project root
- Check file permissions
- Verify CSV format (columns: age, sex, bmi, children, smoker, region, charges)

### Gemini API errors

- Verify `GOOGLE_API_KEY` is set correctly
- Check internet connectivity
- Validate API key permissions in Google Cloud Console
- Monitor API usage quotas

### Streamlit crashes

- Restart kernel: `Ctrl+C` then rerun
- Clear cache: `streamlit cache clear`
- Verify pandas/numpy versions in requirements.txt

### Missing visualizations

- Run Phase 5: `python 05_visualizations.py`
- Verify matplotlib is installed (included in requirements.txt)
- Check `outputs/figures/` directory exists

---

## 📚 Extended Analysis

### NumPy Operations Used

- **np.mean(), np.median(), np.std()** — Aggregation functions
- **Boolean masking** — Segment selection by smoker status
- **np.broadcasting** — BMI-charge index computation
- **np.percentile()** — High-cost threshold calculation

### Pandas Operations Used

- **pd.read_csv()** — Data loading
- **pd.cut()** — Age group binning
- **pivot_table()** — Age × smoker cross-tabulation
- **groupby(), agg()** — Grouped aggregations
- **pd.get_dummies()** — One-hot encoding for regions
- **pd.concat()** — Feature matrix assembly

### Feature Engineering

- **Smoker encoded:** binary (1=yes, 0=no)
- **Sex encoded:** binary (1=male, 0=female)
- **Region encoded:** one-hot (3 binary columns)
- **BMI charge index:** charges × (BMI / mean_BMI)
- **Interaction features:** smoker × BMI, age × BMI

---

## 📝 Python Concepts Demonstrated

### Data Analysis

- Missing value detection and handling
- Descriptive statistics and distributions
- Correlation and causality reasoning

### NumPy

- Array operations and broadcasting
- Boolean masking and indexing
- Aggregation and statistical functions

### Pandas

- DataFrame operations
- Grouping and pivoting
- Categorical encoding

### Visualization

- Matplotlib: publication-quality static charts
- Plotly: interactive web-based visualizations
- Color and size encoding for multivariate analysis

### AI Integration

- RESTful API client design
- Context-based prompt engineering
- Error handling and fallback patterns

### Web Application

- Streamlit multi-page interface
- Caching and performance optimization
- State management
- File I/O and data downloads

---

## 🎓 Learning Path

**Beginner:** Review `02_eda_distributions.py` to understand exploratory analysis  
**Intermediate:** Study `03_numpy_aggregations.py` for NumPy techniques  
**Advanced:** Examine `06_feature_engineering.py` for ML pipeline design  
**Production:** Inspect `streamlit_app.py` for professional application patterns

---

## 📞 Support

For issues:

1. Check `.env` file for API configuration
2. Review error messages in terminal/console
3. Verify data file integrity: `insurance.csv`
4. Test individual analysis phases independently: `python 01_data_loading.py`

---

## 📄 License

Educational project for data analysis course. Use freely for learning purposes.

---

**Last Updated:** April 2026  
**Status:** Production-ready with full pipeline + AI integration
│
└── outputs/
├── figures/ # Saved chart PNGs
├── tables/ # Pivot tables and feature matrix CSV
└── reports/ # AI-generated actuarial reports

````

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
````

---

## Gemini (Generative AI) Integration

This project includes a lightweight integration with Google's Generative Language
API (Gemini) for the Streamlit Data Assistant. The Streamlit app can send the
user question plus a short dataset summary to Gemini and display the model's
response.

Setup:

- Set your API key in the environment: `GOOGLE_API_KEY`.
- The Streamlit assistant uses the fixed model `gemini-2.5-flash-lite` (no model selection in the UI).

Example (Windows PowerShell):

```powershell
setx GOOGLE_API_KEY "YOUR_API_KEY_HERE"
# Restart your terminal / IDE so env vars are available to Streamlit
streamlit run streamlit_app.py
```

Notes:

- The minimal client is implemented at `ai/gemini_client.py` and uses an API
  key via the Generative Language REST endpoint. For production workflows
  prefer Google client libraries and service account authentication.

All outputs are saved automatically to the `outputs/` directory.
