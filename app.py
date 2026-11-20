import io
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from graph import analytics_graph
from tools.file_loader import load_dataframe
from config import NVIDIA_API_KEY, NVIDIA_MODEL




st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")
st.title("📊 AI Data Analyst")
st.caption("Multi-agent data profiling, cleaning, analytics and interactive dashboards — powered by LangGraph, LangChain and NVIDIA.")

with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Upload CSV or Excel, review cleaning options, then run the agent workflow.")
    fill_missing = st.checkbox("Fill missing values", value=True, help="Numeric columns use median; text/category columns use mode.")
    remove_duplicates = st.checkbox("Remove exact duplicate rows", value=True)
    trim_text = st.checkbox("Trim whitespace", value=True)
    st.divider()
    if NVIDIA_API_KEY:
        st.success(f"NVIDIA configured · {NVIDIA_MODEL}")
    else:
        st.warning("NVIDIA API key not found. Core analysis works without it; add NVIDIA_API_KEY to .env for AI-generated insights.")

uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

def dataframe_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="cleaned_data")
    return output.getvalue()

if uploaded is None:
    st.info("Start by uploading a dataset. You can test the app with any CSV or Excel file.")
    sample = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=8, freq="MS"),
        "Region": ["East", "West", "North", "East", "South", "West", "North", "South"],
        "Product": ["A", "B", "A", "C", "B", "C", "A", "B"],
        "Sales": [1200, 900, 1500, 800, 1100, 950, 1700, 1300],
        "Units": [12, 9, 15, 8, 11, 10, 17, 13],
    })
    st.subheader("Example dataset")
    st.dataframe(sample, use_container_width=True)
    st.download_button("Download example CSV", sample.to_csv(index=False).encode("utf-8"), "sample_sales.csv", "text/csv")
    st.stop()

try:
    original_df = load_dataframe(uploaded)
except Exception as exc:
    st.error(f"Could not read the uploaded file: {exc}")
    st.stop()

if original_df.empty:
    st.error("The uploaded file has no rows.")
    st.stop()
if len(original_df.columns) == 0:
    st.error("The uploaded file has no columns.")
    st.stop()

st.subheader("Original data preview")
st.caption(f"{len(original_df):,} rows · {len(original_df.columns):,} columns")
st.dataframe(original_df.head(100), use_container_width=True)

if st.button("🚀 Run multi-agent analysis", type="primary", use_container_width=True):
    with st.spinner("LangGraph agents are profiling, cleaning and analyzing your data..."):
        try:
            result = analytics_graph.invoke({
                "original_df": original_df,
                "cleaning_options": {
                    "fill_missing": fill_missing,
                    "remove_duplicates": remove_duplicates,
                    "trim_text": trim_text,
                },
            })
            st.session_state["analysis_result"] = result
            st.session_state["source_name"] = uploaded.name
        except Exception as exc:
            st.exception(exc)

result = st.session_state.get("analysis_result")
if not result:
    st.caption("Click **Run multi-agent analysis** to generate the cleaned dataset and dashboard.")
    st.stop()

cleaned = result["cleaned_df"]
profile = result["profile"]
analytics = result["analytics"]
plan = result["dashboard_plan"]

st.divider()
st.header("📈 Dashboard")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows after cleaning", f"{len(cleaned):,}", delta=f"{len(cleaned)-len(original_df):+,} vs original")
k2.metric("Columns", f"{len(cleaned.columns):,}")
k3.metric("Missing values remaining", f"{int(cleaned.isna().sum().sum()):,}")
k4.metric("Duplicate rows remaining", f"{int(cleaned.duplicated().sum()):,}")

tab_dashboard, tab_quality, tab_cleaning, tab_insights, tab_ask = st.tabs([
    "Interactive Dashboard", "Data Quality", "Cleaning Log", "AI Insights", "Ask Your Data"
])

with tab_dashboard:
    st.subheader("Explore your data")
    all_cols = list(cleaned.columns)
    numeric_cols = list(cleaned.select_dtypes(include="number").columns)
    cat_cols = list(cleaned.select_dtypes(include=["object", "string", "category", "bool"]).columns)
    date_cols = list(cleaned.select_dtypes(include=["datetime", "datetimetz"]).columns)

    c1, c2, c3 = st.columns(3)
    filter_col = c1.selectbox("Optional filter column", ["(none)"] + all_cols)
    filtered = cleaned.copy()
    if filter_col != "(none)":
        options = filtered[filter_col].dropna().unique().tolist()
        chosen = c2.multiselect("Filter values", options, default=[])
        if chosen:
            filtered = filtered[filtered[filter_col].isin(chosen)]
    st.caption(f"Showing {len(filtered):,} of {len(cleaned):,} rows after filters.")

    if numeric_cols:
        metric_col = c3.selectbox("KPI metric", numeric_cols)
        total = filtered[metric_col].sum()
        avg = filtered[metric_col].mean()
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{metric_col} total", f"{total:,.2f}")
        m2.metric(f"{metric_col} average", f"{avg:,.2f}" if pd.notna(avg) else "N/A")
        m3.metric(f"{metric_col} median", f"{filtered[metric_col].median():,.2f}" if filtered[metric_col].notna().any() else "N/A")

    left, right = st.columns(2)
    if numeric_cols:
        with left:
            y_col = st.selectbox("Numeric distribution", numeric_cols, key="hist_col")
            st.plotly_chart(px.histogram(filtered, x=y_col, title=f"Distribution of {y_col}", marginal="box"), use_container_width=True)
    if cat_cols:
        with right:
            cat_col = st.selectbox("Category breakdown", cat_cols, key="cat_col")
            top = filtered[cat_col].astype("string").fillna("(missing)").value_counts().head(20).rename_axis(cat_col).reset_index(name="Count")
            st.plotly_chart(px.bar(top, x=cat_col, y="Count", title=f"Top categories: {cat_col}"), use_container_width=True)

    if len(numeric_cols) >= 2:
        st.subheader("Relationship between numeric variables")
        x_default = 0
        x_col = st.selectbox("X-axis", numeric_cols, index=x_default, key="scatter_x")
        y_col = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="scatter_y")
        color_col = st.selectbox("Color by (optional)", ["(none)"] + cat_cols, key="scatter_color")
        color_arg = None if color_col == "(none)" else color_col
        st.plotly_chart(px.scatter(filtered, x=x_col, y=y_col, color=color_arg, title=f"{y_col} vs {x_col}"), use_container_width=True)

    if date_cols and numeric_cols:
        st.subheader("Trend over time")
        date_col = st.selectbox("Date column", date_cols, key="date_col")
        trend_metric = st.selectbox("Trend metric", numeric_cols, key="trend_metric")
        trend_df = filtered.dropna(subset=[date_col]).copy()
        if not trend_df.empty:
            trend_df[date_col] = pd.to_datetime(trend_df[date_col], errors="coerce")
            trend_df = trend_df.groupby(date_col, as_index=False)[trend_metric].sum().sort_values(date_col)
            st.plotly_chart(px.line(trend_df, x=date_col, y=trend_metric, markers=True, title=f"{trend_metric} over time"), use_container_width=True)

    st.subheader("Filtered data")
    st.dataframe(filtered, use_container_width=True, height=300)

with tab_quality:
    st.subheader("Original data quality report")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Original rows", f"{profile['rows']:,}")
    q2.metric("Original columns", f"{profile['columns']:,}")
    q3.metric("Missing values", f"{profile['missing_total']:,}")
    q4.metric("Exact duplicates", f"{profile['duplicates']:,}")
    quality_df = pd.DataFrame({
        "Column": list(original_df.columns),
        "Data type": [str(original_df[c].dtype) for c in original_df.columns],
        "Missing": [int(original_df[c].isna().sum()) for c in original_df.columns],
        "Missing %": [round(float(original_df[c].isna().mean()*100), 2) for c in original_df.columns],
        "Unique values": [int(original_df[c].nunique(dropna=True)) for c in original_df.columns],
    })
    st.dataframe(quality_df, use_container_width=True)
    if profile["missing_by_column"]:
        st.bar_chart(pd.Series(profile["missing_by_column"], name="Missing values"))
    else:
        st.success("No missing values found in the original dataset.")

with tab_cleaning:
    st.subheader("Cleaning actions")
    for line in result.get("cleaning_log", []):
        st.write("• " + line)
    st.subheader("Cleaned data preview")
    st.dataframe(cleaned.head(200), use_container_width=True)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download cleaned CSV", cleaned.to_csv(index=False).encode("utf-8"), "cleaned_data.csv", "text/csv", use_container_width=True)
    d2.download_button("⬇️ Download cleaned Excel", dataframe_to_excel(cleaned), "cleaned_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

with tab_insights:
    st.subheader("AI-generated findings")
    st.markdown(result.get("insights", "No insights were generated."))
    st.subheader("Computed numeric summary")
    if analytics.get("numeric_summary"):
        summary_df = pd.DataFrame(analytics["numeric_summary"]).T
        st.dataframe(summary_df, use_container_width=True)
    else:
        st.info("No numeric columns were found.")
    if analytics.get("strongest_correlations"):
        st.subheader("Strongest numeric correlations")
        st.dataframe(pd.DataFrame(analytics["strongest_correlations"]), use_container_width=True)

with tab_ask:
    st.subheader("Ask a question about your dataset")
    st.caption("Answers are grounded in computed summaries. For complex questions, inspect the filtered table and charts as well.")
    question = st.text_input("Your question", placeholder="Which numeric column has the highest average?")
    if st.button("Answer question"):
        if not question.strip():
            st.warning("Enter a question first.")
        elif NVIDIA_API_KEY:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            context = json.dumps({"profile": profile, "analytics": analytics}, default=str)[:18000]
            prompt = (
                "Answer the user's question using only this computed dataset profile and analytics. "
                "If the information is not present, say what cannot be determined. Do not invent values. "
                f"\nDATA: {context}\nQUESTION: {question}"
            )
            try:
                answer = ChatNVIDIA(model=NVIDIA_MODEL, temperature=0, max_tokens=700).invoke(prompt)
                st.markdown(str(answer.content))
            except Exception as exc:
                st.error(f"NVIDIA request failed: {exc}")
        else:
            st.info("Add NVIDIA_API_KEY to your .env file to enable natural-language Q&A. Current computed summaries are available in the other tabs.")
