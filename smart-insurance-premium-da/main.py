#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from typing import Optional
import requests

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# DATA ANALYSIS PIPELINE RUNNER

def run_script(script_name, phase_num, phase_name):
    print("\n" + "=" * 80)
    print(f"RUNNING PHASE {phase_num}: {phase_name}")
    print("=" * 80)
    
    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            check=True
        )
        elapsed = time.time() - start_time
        print(f"\n✓ Phase {phase_num} completed successfully in {elapsed:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Phase {phase_num} failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Script not found: {script_name}")
        return False


def run_analysis_pipeline():
    print("=" * 80)
    print("INSURANCE PREMIUM DATA ANALYSIS - FULL PIPELINE")
    print("\nThis will run all 7 phases of the analysis:")
    print("  Phase 1: Data Loading")
    print("  Phase 2: EDA & Distributions")
    print("  Phase 3: NumPy Aggregations")
    print("  Phase 4: Segmentation Analysis")
    print("  Phase 5: Advanced Visualizations")
    print("  Phase 6: Feature Engineering")
    print("  Phase 7: AI-Assisted Insights")
    
    phases = [
        ("01_data_loading.py", 1, "Data Loading"),
        ("02_eda_distributions.py", 2, "EDA & Distributions"),
        ("03_numpy_aggregations.py", 3, "NumPy Aggregations"),
        ("04_segmentation.py", 4, "Segmentation Analysis"),
        ("05_visualizations.py", 5, "Advanced Visualizations"),
        ("06_feature_engineering.py", 6, "Feature Engineering"),
        ("07_ai_insights.py", 7, "AI-Assisted Insights")
    ]
    
    results = []
    start_time = time.time()
    
    for script, phase_num, phase_name in phases:
        success = run_script(script, phase_num, phase_name)
        results.append((phase_num, phase_name, success))
        
        if not success:
            print(f"\n⚠ Warning: Phase {phase_num} failed. Continuing with remaining phases...")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    
    for phase_num, phase_name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  Phase {phase_num} ({phase_name}): {status}")
    
    successful = sum(1 for _, _, success in results if success)
    total = len(results)
    
    print(f"\nCompleted: {successful}/{total} phases")
    print(f"Total execution time: {total_time:.2f}s")
    
    if successful == total:
        print("\n✓ All phases completed successfully!")
        print("\nGenerated outputs:")
        print("  • outputs/figures/ - Visualization plots")
        print("  • outputs/tables/ - Data tables and matrices")
        print("  • outputs/reports/ - AI-generated insights reports")
    else:
        print(f"\n⚠ {total - successful} phase(s) failed. Check output above for details.")
        return False
    
    print("=" * 80)
    return True


# GEMINI AI CLIENT

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1/models"


def _get_gemini_api_key() -> str:
    """Get Gemini API key from environment."""
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is not set. Set it to your API key."
        )
    return key


def generate_text(
    prompt: str,
    model: Optional[str] = None,
    max_output_tokens: int = 1024,
    temperature: float = 0.3,
    timeout: int = 30,
) -> str:
    """Generate text using Google Generative Language API (v1).
    
    Args:
        prompt: The text prompt to send to the model.
        model: Model name (e.g. "gemini-2.0-flash").
        max_output_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0-1.0).
        timeout: HTTP request timeout in seconds.
    
    Returns:
        The generated text response.
    """
    model = model or DEFAULT_GEMINI_MODEL
    api_key = _get_gemini_api_key()
    url = f"{GEMINI_API_BASE}/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_output_tokens),
        }
    }

    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    if resp.status_code != 200:
        body = None
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        raise RuntimeError(f"Generative Language API error {resp.status_code}: {body}")

    data = resp.json()

    # Response structure: candidates[0].content.parts[0].text
    if isinstance(data, dict):
        candidates = data.get("candidates")
        if candidates and isinstance(candidates, list):
            first = candidates[0]
            if isinstance(first, dict):
                content = first.get("content", {})
                parts = content.get("parts", [])
                if parts and isinstance(parts, list):
                    part = parts[0]
                    if isinstance(part, dict) and "text" in part:
                        return part["text"].strip()

    return str(data)


gemini_generate_text = generate_text

# PAGE CONFIGURATION & STYLING

st.set_page_config(
    page_title="Smart Insurance Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --primary-color: #007AFF;
        --secondary-color: #5AC8FA;
        --success-color: #34C759;
        --warning-color: #FF9500;
        --danger-color: #FF3B30;
        --text-primary: #ffffff;
        --text-secondary: #a0a0b0;
        --bg-primary: #0d1321;
        --bg-secondary: #1a2332;
        --border-color: #2a3548;
    }

    * {
        margin: 0;
        padding: 0;
    }

    body, .main {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        background-color: #0d1321 !important;
        color: #ffffff !important;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #0d1321 !important;
    }

    [data-testid="stMainBlockContainer"] {
        background-color: #0d1321 !important;
    }

    .main .block-container {
        background-color: #0d1321 !important;
        padding: 2rem 1rem;
    }

    [data-testid="stMarkdownContainer"] {
        background-color: transparent;
        color: #ffffff !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a2332;
        border-radius: 12px;
        padding: 0.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }

    .metric-card {
        background: #1a2332;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
        transition: all 0.2s ease;
        color: #ffffff;
    }

    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        border-color: var(--primary-color);
        transform: translateY(-2px);
    }

    .header-section {
        padding: 2rem 2rem 1rem 2rem;
        border-bottom: 1px solid var(--border-color);
        background: linear-gradient(135deg, #007AFF 0%, #5AC8FA 100%);
        color: #ffffff;
        border-radius: 12px;
        margin-bottom: 2rem;
    }

    .header-section h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
        color: #ffffff;
    }

    .header-section p {
        font-size: 1rem;
        opacity: 0.95;
        margin: 0;
        font-weight: 500;
        color: #ffffff;
    }

    .insight-box {
        background: #1a2332;
        padding: 1.5rem;
        border-left: 4px solid var(--primary-color);
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
        border: 1px solid var(--border-color);
        color: #ffffff;
    }

    .insight-box h3 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        color: #ffffff;
        font-weight: 600;
        font-size: 1.1rem;
    }

    .insight-box p {
        margin: 0.25rem 0;
        color: #ffffff;
        line-height: 1.6;
    }

    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0.5rem 0;
    }

    .stat-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 0;
    }

    .divider {
        height: 1px;
        background: var(--border-color);
        margin: 2rem 0;
    }

    /* Streamlit overrides for better visibility */
    section[data-testid="stSidebar"] {
        background-color: #1a2332;
        border-right: 1px solid var(--border-color);
        color: #ffffff;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    .stSelectbox, .stSlider, .stTextInput, .stTextArea {
        background-color: #1a2332;
        color: #ffffff;
    }

    [data-testid="stForm"] {
        background-color: #1a2332;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
        color: #ffffff;
    }

    [data-testid="stDataFrame"] {
        background-color: #1a2332;
        color: #ffffff;
    }

    .stDataFrame {
        background-color: #1a2332 !important;
        color: #ffffff !important;
    }

    /* Charts */
    [data-testid="stPlotlyChart"] {
        background-color: #1a2332;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }

    /* Info/Warning boxes */
    .stAlert {
        border-radius: 8px;
        background-color: #1a2332;
        color: #ffffff;
    }

    /* Additional text color fixes */
    p, span, div, label {
        color: #ffffff !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }

    /* Streamlit specific overrides */
    [data-testid="stHeading"] {
        color: #ffffff !important;
    }

    [data-testid="stText"] {
        color: #ffffff !important;
    }

    [data-testid="stButton"] {
        color: #ffffff !important;
    }

    /* Selectbox and input elements */
    [data-baseweb="select"] {
        background-color: #1a2332 !important;
        color: #ffffff !important;
    }

    [data-baseweb="input"] {
        background-color: #1a2332 !important;
        color: #ffffff !important;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] p {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# DATA LOADING & UTILITIES

@st.cache_resource
def load_data():
    """Load insurance data with caching for performance."""
    candidates = [
        'insurance_with_bmi_index.csv',
        'insurance.csv'
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                return df
            except Exception:
                continue
    raise FileNotFoundError('No insurance data file found.')


@st.cache_data
def load_reports():
    """Load AI-generated reports."""
    reports = {}
    report_files = [
        ('Cost Drivers', 'outputs/reports/report_cost_drivers.md'),
        ('Risk Profiling', 'outputs/reports/report_risk_profiling.md'),
        ('Business Intelligence', 'outputs/reports/report_business_intelligence.md')
    ]
    for name, path in report_files:
        if os.path.exists(path):
            with open(path, 'r') as f:
                reports[name] = f.read()
    return reports


@st.cache_data
def load_figures():
    """Load pre-generated visualization files."""
    figures = {}
    figure_files = [
        ('BM vs Charges', 'outputs/figures/scatter_bmi_charges.png'),
        ('Charges Distribution', 'outputs/figures/histogram_charges_smoker.png'),
        ('Age-Smoker Heatmap', 'outputs/figures/heatmap_agegroup_smoker.png'),
        ('Regional Analysis', 'outputs/figures/bar_region_dependents.png')
    ]
    for name, path in figure_files:
        if os.path.exists(path):
            figures[name] = path
    return figures


def create_age_groups(df):
    """Create age group categories."""
    if 'age_group' not in df.columns:
        bins = [17, 29, 39, 49, 64]
        labels = ['20s', '30s', '40s', '50s+']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    return df


def query_gemini_api(prompt: str, context: str) -> str:
    """Query Gemini API with context and error handling."""
    try:
        full_prompt = f"""You are an expert actuarial analyst and data scientist specializing in insurance premium analysis.

Dataset Context:
{context}

User Question: {prompt}

Provide a clear, concise, professional response with specific data points when relevant. Use currency formatting ($) and percentages where appropriate. Keep the response focused and actionable."""
        
        response = gemini_generate_text(
            full_prompt,
            max_output_tokens=1024,
            temperature=0.3
        )
        return response
    except Exception as e:
        return f"⚠️ Error contacting Gemini API: {str(e)}"


# KEY METRICS COMPUTATION

def compute_portfolio_metrics(df):
    df = create_age_groups(df)
    
    smokers = df[df['smoker'] == 'yes']
    nonsmokers = df[df['smoker'] == 'no']
    
    metrics = {
        'total_policies': len(df),
        'total_charges': df['charges'].sum(),
        'mean_charge': df['charges'].mean(),
        'median_charge': df['charges'].median(),
        'smoker_count': len(smokers),
        'smoker_pct': len(smokers) / len(df) * 100,
        'smoker_mean': smokers['charges'].mean() if len(smokers) > 0 else 0,
        'nonsmoker_mean': nonsmokers['charges'].mean() if len(nonsmokers) > 0 else 0,
        'smoker_multiplier': (smokers['charges'].mean() / nonsmokers['charges'].mean()) if len(nonsmokers) > 0 else 0,
        'avg_age': df['age'].mean(),
        'avg_bmi': df['bmi'].mean(),
        'avg_children': df['children'].mean(),
    }
    return metrics


# STREAMLIT UI

def run_streamlit_ui():
    # Initialize session state for pipeline execution tracking
    if 'pipeline_executed' not in st.session_state:
        st.session_state.pipeline_executed = False
    
    # Run pipeline only once per session
    if not st.session_state.pipeline_executed:
        with st.spinner("📊 Running data analysis pipeline for the first time..."):
            print("\n" + "=" * 80)
            print("INSURANCE PREMIUM DATA ANALYSIS - FULL PIPELINE")
            print("=" * 80)
            print("\nRunning all 7 phases of the analysis...")
            
            phases = [
                ("01_data_loading.py", 1, "Data Loading"),
                ("02_eda_distributions.py", 2, "EDA & Distributions"),
                ("03_numpy_aggregations.py", 3, "NumPy Aggregations"),
                ("04_segmentation.py", 4, "Segmentation Analysis"),
                ("05_visualizations.py", 5, "Advanced Visualizations"),
                ("06_feature_engineering.py", 6, "Feature Engineering"),
                ("07_ai_insights.py", 7, "AI-Assisted Insights")
            ]
            
            start_time = time.time()
            for script, phase_num, phase_name in phases:
                run_script(script, phase_num, phase_name)
            
            total_time = time.time() - start_time
            print(f"\n✓ All phases completed in {total_time:.2f}s")
            print("=" * 80)
            
            st.session_state.pipeline_executed = True
            st.success("✅ Data analysis complete! Dashboard ready.")
    
    df = load_data()
    df = create_age_groups(df)
    metrics = compute_portfolio_metrics(df)
    
    reports = load_reports()
    figures = load_figures()
    
    st.markdown("""
    <div class="header-section">
        <h1>Smart Insurance Premium Analytics</h1>
        <p>AI-Powered Actuarial Risk Analysis for 1,338 Policyholders</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select View",
        ["📊 Dashboard", "🔍 Insights", "💬 AI Assistant", "📈 Raw Data"],
        label_visibility="collapsed"
    )
    
    # PAGE 1: DASHBOARD
    if page == "📊 Dashboard":
        st.subheader("Portfolio Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="stat-label">Total Policies</div>
                <div class="stat-value">{metrics['total_policies']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="stat-label">Mean Annual Charge</div>
                <div class="stat-value">${metrics['mean_charge']:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="stat-label">Smoker Prevalence</div>
                <div class="stat-value">{metrics['smoker_pct']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="stat-label">Smoker Risk Multiplier</div>
                <div class="stat-value">{metrics['smoker_multiplier']:.2f}×</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        # Cost drivers insight
        st.markdown("""
        ### 🔑 Key Findings
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="insight-box">
                <h3>Smoker Impact</h3>
                <p>Smokers pay <strong>${metrics['smoker_mean']:,.0f}</strong> annually vs <strong>${metrics['nonsmoker_mean']:,.0f}</strong> for non-smokers — a <strong>{metrics['smoker_multiplier']:.2f}×</strong> premium multiplier.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="insight-box">
                <h3>Portfolio Composition</h3>
                <p><strong>{metrics['smoker_count']:,}</strong> smokers ({metrics['smoker_pct']:.1f}%) generate approximately 50% of total portfolio charges, indicating strong segmentation opportunity.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        # Visualizations
        st.subheader("Analysis Visualizations")
        
        if figures:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'BM vs Charges' in figures:
                    st.image(figures['BM vs Charges'], caption="BMI vs Insurance Charges")
                if 'Charges Distribution' in figures:
                    st.image(figures['Charges Distribution'], caption="Income Distribution by Smoker Status")
            
            with col2:
                if 'Age-Smoker Heatmap' in figures:
                    st.image(figures['Age-Smoker Heatmap'], caption="Average Charges by Age & Smoker Status")
                if 'Regional Analysis' in figures:
                    st.image(figures['Regional Analysis'], caption="Regional & Dependent Analysis")
        
        # Interactive Plotly charts
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.subheader("Interactive Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_smoker = st.selectbox("Filter by Smoker Status", ["All", "Yes", "No"])
        with col2:
            age_range = st.slider("Age Range", int(df['age'].min()), int(df['age'].max()), 
                                  (int(df['age'].min()), int(df['age'].max())))
        with col3:
            selected_region = st.selectbox("Region", ["All"] + sorted(df['region'].unique().tolist()))
        
        # Apply filters
        filtered_df = df.copy()
        if selected_smoker != "All":
            filtered_df = filtered_df[filtered_df['smoker'] == selected_smoker.lower()]
        if selected_region != "All":
            filtered_df = filtered_df[filtered_df['region'] == selected_region]
        filtered_df = filtered_df[(filtered_df['age'] >= age_range[0]) & (filtered_df['age'] <= age_range[1])]
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Age vs Charges scatter
            fig = px.scatter(filtered_df, x='age', y='charges', color='smoker',
                           size='bmi', hover_data=['region', 'children'],
                           title="Age vs Charges (colored by smoker, sized by BMI)",
                           labels={'charges': 'Annual Charge ($)', 'age': 'Age (years)'})
            fig.update_layout(height=400)
            st.plotly_chart(fig)
        
        with col2:
            # Age group pivot
            age_group_pivot = filtered_df.pivot_table(
                values='charges', index='age_group', columns='smoker', aggfunc='mean'
            )
            fig = go.Figure(data=[
                go.Bar(name='Non-Smoker', x=age_group_pivot.index, y=age_group_pivot.get('no', []), marker_color='#34C759'),
                go.Bar(name='Smoker', x=age_group_pivot.index, y=age_group_pivot.get('yes', []), marker_color='#FF3B30')
            ])
            fig.update_layout(title="Average Charges by Age Group", 
                            xaxis_title="Age Group", yaxis_title="Average Charge ($)",
                            height=400, barmode='group')
            st.plotly_chart(fig)
    
    # PAGE 2: INSIGHTS
    elif page == "🔍 Insights":
        st.subheader("Actuarial Insights & Reports")
        
        report_selection = st.selectbox(
            "Select Report",
            list(reports.keys()) if reports else ["No reports available"]
        )
        
        if report_selection in reports:
            st.markdown(reports[report_selection])
        else:
            st.warning("No AI-generated reports found. Run the analysis pipeline to generate reports.")
    
    # PAGE 3: AI ASSISTANT
    elif page == "💬 AI Assistant":
        st.subheader("Interactive AI Chatbot")
        
        # Build dataset context for AI
        pivot = df.pivot_table(values='charges', index='age_group', columns='smoker', aggfunc='mean')
        context = f"""
Dataset: Health Insurance Premium Analysis
Total Records: {metrics['total_policies']:,}
Mean Annual Charge: ${metrics['mean_charge']:,.0f}
Median Annual Charge: ${metrics['median_charge']:,.0f}
Smoker Count: {metrics['smoker_count']:,} ({metrics['smoker_pct']:.1f}%)
Smoker Mean Charge: ${metrics['smoker_mean']:,.0f}
Non-Smoker Mean Charge: ${metrics['nonsmoker_mean']:,.0f}
Smoker Premium Multiplier: {metrics['smoker_multiplier']:.2f}×
Average Age: {metrics['avg_age']:.1f} years
Average BMI: {metrics['avg_bmi']:.1f}
Average Children per Policy: {metrics['avg_children']:.2f}

Age Group Analysis:
{pivot.to_string()}
"""
        
        st.info("💡 Ask me about insurance premiums, risk factors, actuarial insights, wellness programs, or pricing fairness.")
        
        user_question = st.text_area(
            "Your Question",
            placeholder="e.g., What are the top factors driving insurance costs? Or: Can you help design a wellness program?",
            height=100
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            use_ai = st.checkbox("Use Gemini API", value=True)
        with col2:
            if use_ai and not os.getenv("GOOGLE_API_KEY"):
                st.warning("⚠️ GOOGLE_API_KEY environment variable not set. Set it for cloud AI responses.")
        
        if st.button("Get Response", type="primary"):
            if not user_question.strip():
                st.error("Please enter a question.")
            else:
                with st.spinner("🤔 Analyzing..."):
                    if use_ai:
                        response = query_gemini_api(user_question, context)
                    else:
                        # Fallback to simple local response
                        response = f"Local Mode: Based on the dataset, I can see that smoking status has a {metrics['smoker_multiplier']:.2f}× impact on premiums, with smokers paying ${metrics['smoker_mean']:,.0f} vs ${metrics['nonsmoker_mean']:,.0f} for non-smokers."
                    
                    st.markdown("### Assistant Response")
                    st.markdown(response)
    
    # PAGE 4: RAW DATA
    elif page == "📈 Raw Data":
        st.subheader("Raw Dataset")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_smoker = st.selectbox("Smoker Filter", ["All", "Yes", "No"])
        with col2:
            filter_region = st.selectbox("Region Filter", ["All"] + sorted(df['region'].unique().tolist()))
        with col3:
            records_to_show = st.selectbox("Show Records", [50, 100, 500, "All"])
        
        filtered_data = df.copy()
        if filter_smoker != "All":
            filtered_data = filtered_data[filtered_data['smoker'] == filter_smoker.lower()]
        if filter_region != "All":
            filtered_data = filtered_data[filtered_data['region'] == filter_region]
        
        if records_to_show != "All":
            filtered_data = filtered_data.head(int(records_to_show))
        
        st.dataframe(filtered_data)
        
        # Download button
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"insurance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


# MAIN ENTRY POINT
# Pipeline runs once on first load, then never again during navigation

if __name__ == '__main__':
    run_streamlit_ui()
