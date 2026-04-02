import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def load_data():
    candidates = [
        'outputs/tables/feature_matrix_with_target.csv',
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
    raise FileNotFoundError('No data file found. Add insurance.csv to the project folder.')


def build_short_context(df):
    total = len(df)
    mean_charge = df['charges'].mean() if 'charges' in df.columns else None
    median_charge = df['charges'].median() if 'charges' in df.columns else None
    smoker_pct = (df['smoker'] == 'yes').mean() * 100 if 'smoker' in df.columns else None
    ctx = {
        'rows': int(total),
        'mean_charges': float(mean_charge) if mean_charge is not None else None,
        'median_charges': float(median_charge) if median_charge is not None else None,
        'smoker_pct': float(smoker_pct) if smoker_pct is not None else None,
    }
    return ctx


def answer_question(user_q: str, df: pd.DataFrame) -> str:
    """Lightweight local assistant that answers basic dataset questions.

    Supports: mean/median charges, smoker split, region summaries, age-group pivot,
    top cost drivers (simple correlations), and example prompts.
    """
    q = (user_q or '').strip()
    if not q:
        return 'Please ask a question about the dataset (examples available below).'

    ql = q.lower()

    out_lines = []
    if 'charges' in df.columns:
        mean_charge = float(df['charges'].mean())
        median_charge = float(df['charges'].median())
        out_lines.append(f"Records: {len(df):,}")
        out_lines.append(f"Mean charges: ${mean_charge:,.2f}")
        out_lines.append(f"Median charges: ${median_charge:,.2f}")
    else:
        return 'Dataset does not contain `charges` column.'

    if 'smoker' in df.columns and 'smoker' in ql:
        smoker_pct = (df['smoker'] == 'yes').mean() * 100
        smokers = df[df['smoker'] == 'yes']
        nonsmokers = df[df['smoker'] == 'no']
        smoker_mean = smokers['charges'].mean() if len(smokers) else float('nan')
        nonsmoker_mean = nonsmokers['charges'].mean() if len(nonsmokers) else float('nan')
        ratio = (smoker_mean / nonsmoker_mean) if nonsmoker_mean and nonsmoker_mean > 0 else None
        out_lines.append(f"Smoker prevalence: {smoker_pct:.1f}%")
        out_lines.append(f"Mean smoker charges: ${smoker_mean:,.0f}")
        out_lines.append(f"Mean non-smoker charges: ${nonsmoker_mean:,.0f}")
        if ratio:
            out_lines.append(f"Smoker / Non-smoker mean ratio: {ratio:.2f}×")
        return '\n'.join(out_lines)

    if 'region' in df.columns and 'region' in ql:
        region_counts = df['region'].value_counts()
        region_means = df.groupby('region')['charges'].mean()
        lines = [f"{r.title()}: {region_counts[r]} records, mean ${region_means[r]:,.0f}" for r in region_counts.index]
        return "\n".join(lines)

    if 'age group' in ql or 'age_group' in df.columns and 'age' in ql:
        if 'age_group' not in df.columns:
            bins = [17, 29, 39, 49, 64]
            labels = ['20s', '30s', '40s', '50s+']
            df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
        pivot = df.pivot_table(values='charges', index='age_group', columns='smoker', aggfunc='mean')
        return pivot.to_string(float_format='${:,.0f}'.format)

    if 'driver' in ql or 'top' in ql or 'cost' in ql:
        numeric = df.select_dtypes(include=[np.number])
        if 'charges' not in numeric.columns:
            return 'No numeric `charges` column for driver analysis.'
        corr = numeric.corr().get('charges', pd.Series()).abs().drop('charges', errors='ignore')
        corr = corr.sort_values(ascending=False)
        lines = [f"{i}: corr={v:.2f}" for i, v in corr.items()[:3]] if not corr.empty else ["No numeric correlations available."]
        if 'smoker' in df.columns:
            smokers = df[df['smoker'] == 'yes']['charges']
            nonsmokers = df[df['smoker'] == 'no']['charges']
            if len(smokers) and len(nonsmokers):
                ratio = smokers.mean() / nonsmokers.mean()
                lines.insert(0, f"Smoker status multiplier: {ratio:.2f}× (smoker mean / non-smoker mean)")
        return '\n'.join(lines)

    if 'example' in ql or 'help' in ql or 'what can you' in ql:
        examples = [
            'What is the mean insurance charge?',
            'What percent of policyholders are smokers?',
            'Show average charges by region',
            'What are the top cost drivers?'
        ]
        return 'I can answer simple dataset questions. Examples:\n' + '\n'.join(examples)

    return "I can provide summaries (mean/median), smoker split, region summaries, age-group pivots, and simple driver heuristics. Try asking e.g. 'What is the mean insurance charge?' or 'What percent are smokers?'"


def main():
    st.set_page_config(page_title='Smart Insurance Dashboard', layout='wide')
    st.title('Smart Insurance — Interactive Dashboard')
    st.markdown('Professional, minimal dashboard showing bar and line charts with dynamic inputs.')

    df = load_data()

    if not {'age', 'bmi', 'sex', 'smoker', 'region', 'charges', 'children'}.issubset(df.columns):
        try:
            raw = pd.read_csv('insurance.csv')
            for c in ['age', 'bmi', 'sex', 'smoker', 'region', 'charges', 'children']:
                if c not in df.columns and c in raw.columns:
                    df[c] = raw[c]
        except Exception:
            pass

    if 'age_group' not in df.columns:
        bins = [17, 29, 39, 49, 64]
        labels = ['20s', '30s', '40s', '50s+']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

    with st.sidebar.expander('Data Assistant', expanded=True):
        st.markdown('Ask concise questions about the dataset. Answers are generated locally from the data.')
        user_q = st.text_area('Your question', height=120)
        if st.button('Ask Assistant') and user_q.strip():
            with st.spinner('Thinking...'):
                answer = answer_question(user_q, df)
            st.markdown('**Assistant**')
            st.write(answer)

    st.sidebar.markdown('---')
    st.sidebar.header('Filters')
    regions = sorted(df['region'].dropna().unique())
    selected_regions = st.sidebar.multiselect('Region', options=regions, default=regions)
    smoker_choice = st.sidebar.selectbox('Smoker', options=['All', 'yes', 'no'])
    sex_choice = st.sidebar.selectbox('Sex', options=['All'] + sorted(df['sex'].dropna().unique().tolist()))

    age_min, age_max = int(df['age'].min()), int(df['age'].max())
    age_range = st.sidebar.slider('Age range', age_min, age_max, (age_min, age_max))
    bmi_min, bmi_max = float(df['bmi'].min()), float(df['bmi'].max())
    bmi_range = st.sidebar.slider('BMI range', float(bmi_min), float(bmi_max), (float(bmi_min), float(bmi_max)))

    st.sidebar.markdown('---')
    st.sidebar.header('Chart options')
    chart_type = st.sidebar.selectbox('Chart Type', ['Bar chart', 'Line graph'])

    dff = df.copy()
    dff = dff[dff['region'].isin(selected_regions)]
    if smoker_choice != 'All':
        dff = dff[dff['smoker'] == smoker_choice]
    if sex_choice != 'All':
        dff = dff[dff['sex'] == sex_choice]
    dff = dff[(dff['age'] >= age_range[0]) & (dff['age'] <= age_range[1])]
    dff = dff[(dff['bmi'] >= bmi_range[0]) & (dff['bmi'] <= bmi_range[1])]

    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    col1.metric('Records', len(dff))
    col2.metric('Mean Charges', f"${dff['charges'].mean():,.0f}")
    col3.metric('Median Charges', f"${dff['charges'].median():,.0f}")
    smoker_pct = (dff['smoker'] == 'yes').mean() * 100 if 'smoker' in dff.columns else 0
    col4.metric('Smoker %', f"{smoker_pct:.1f}%")

    st.markdown('---')

    if chart_type == 'Bar chart':
        st.header('Bar Chart — Grouped Aggregation')
        group_by = st.selectbox('Group by', ['region', 'smoker', 'sex', 'children', 'age_group'])
        agg_choice = st.selectbox('Aggregation', ['mean_charges', 'count'])

        if agg_choice == 'mean_charges':
            grouped = dff.groupby(group_by)['charges'].mean().reset_index().sort_values('charges', ascending=False)
            fig = px.bar(grouped, x=group_by, y='charges', labels={'charges': 'Mean Charges (USD)'}, text=grouped['charges'].map(lambda x: f"${x:,.0f}"))
        else:
            grouped = dff.groupby(group_by).size().reset_index(name='count').sort_values('count', ascending=False)
            fig = px.bar(grouped, x=group_by, y='count', labels={'count': 'Count'})

        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.header('Line Graph — Trend / Aggregation')
        x_axis = st.selectbox('X axis', ['age', 'bmi', 'age_group'])
        bin_size = None
        if x_axis in ['age', 'bmi']:
            bins = st.slider('Bins (for aggregation)', 5, 50, 10)
            labels = None
            dff['_x_bin'] = pd.cut(dff[x_axis], bins=bins)
            grouped = dff.groupby('_x_bin')['charges'].mean().reset_index()
            grouped['x_label'] = grouped['_x_bin'].astype(str)
            fig = px.line(grouped, x='x_label', y='charges', markers=True, labels={'charges': 'Mean Charges (USD)', 'x_label': x_axis})
        else:
            grouped = dff.groupby('age_group')['charges'].mean().reset_index()
            fig = px.line(grouped, x='age_group', y='charges', markers=True, labels={'charges': 'Mean Charges (USD)', 'age_group': 'Age Group'})

        if st.checkbox('Show data points', value=True):
            fig.update_traces(mode='lines+markers')

        fig.update_layout(xaxis_title=x_axis, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')
    st.header('Filtered Data')
    st.dataframe(dff.reset_index(drop=True))

    csv = dff.to_csv(index=False).encode('utf-8')
    st.download_button('Download filtered data as CSV', data=csv, file_name='filtered_insurance.csv', mime='text/csv')


if __name__ == '__main__':
    main()
