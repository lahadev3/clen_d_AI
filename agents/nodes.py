import json
import pandas as pd
from state import AnalyticsState
from config import NVIDIA_API_KEY, NVIDIA_MODEL
from tools.analysis_tools import profile_dataframe, analyze_dataframe
from tools.cleaning_tools import clean_dataframe

def profiler_node(state: AnalyticsState) -> dict:
    df = state["original_df"]
    return {"profile": profile_dataframe(df)}

def cleaner_node(state: AnalyticsState) -> dict:
    options = state.get("cleaning_options", {})
    cleaned, log = clean_dataframe(
        state["original_df"],
        fill_missing=options.get("fill_missing", True),
        remove_duplicates=options.get("remove_duplicates", True),
        trim_text=options.get("trim_text", True),
    )
    return {"cleaned_df": cleaned, "cleaning_log": log}

def analytics_node(state: AnalyticsState) -> dict:
    return {"analytics": analyze_dataframe(state["cleaned_df"])}

def _get_llm():
    if not NVIDIA_API_KEY:
        return None
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    return ChatNVIDIA(model=NVIDIA_MODEL, temperature=0.2, max_tokens=900)

def insights_node(state: AnalyticsState) -> dict:
    analytics = state.get("analytics", {})
    profile = state.get("profile", {})
    llm = _get_llm()
    if llm is None:
        insights = (
            "NVIDIA AI is not configured, so these are rule-based findings. "
            f"The cleaned dataset contains {analytics.get('rows', 0):,} rows and "
            f"{analytics.get('columns', 0)} columns. "
        )
        numeric = analytics.get("numeric_summary", {})
        if numeric:
            first = next(iter(numeric.items()))
            insights += f"For numeric column '{first[0]}', mean={first[1]['mean']}, median={first[1]['median']}, range={first[1]['min']} to {first[1]['max']}. "
        correlations = analytics.get("strongest_correlations", [])
        if correlations:
            c = correlations[0]
            insights += f"The strongest observed numeric correlation is between {c['column_1']} and {c['column_2']} (r={c['correlation']}). Correlation does not imply causation."
        insights += "\n\nAdd NVIDIA_API_KEY to .env to enable LLM-generated explanations."
        return {"insights": insights}
    compact = {
        "profile": {k: v for k, v in profile.items() if k not in ("dtypes",)},
        "analytics": analytics,
    }
    prompt = (
        "You are a careful senior data analyst. Explain this dataset using ONLY the supplied computed facts. "
        "Do not invent metrics or claim causation from correlation. Return: 1) 3-5 key findings, "
        "2) data quality caveats, 3) 2 useful next questions. Use plain English.\n\n"
        + json.dumps(compact, default=str)[:18000]
    )
    try:
        response = llm.invoke(prompt)
        return {"insights": str(response.content)}
    except Exception as exc:
        return {"insights": f"AI insight generation failed: {exc}\n\nComputed analytics are still available below."}

def dashboard_plan_node(state: AnalyticsState) -> dict:
    df = state["cleaned_df"]
    numeric = [str(c) for c in df.select_dtypes(include="number").columns]
    categorical = [str(c) for c in df.select_dtypes(include=["object", "string", "category", "bool"]).columns]
    datetime = [str(c) for c in df.select_dtypes(include=["datetime", "datetimetz"]).columns]
    return {"dashboard_plan": {
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "datetime_columns": datetime,
        "default_x": categorical[0] if categorical else (datetime[0] if datetime else None),
        "default_y": numeric[0] if numeric else None,
    }}
