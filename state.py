from typing import Any, TypedDict
import pandas as pd

class AnalyticsState(TypedDict, total=False):
    original_df: pd.DataFrame
    cleaned_df: pd.DataFrame
    cleaning_options: dict[str, Any]
    profile: dict[str, Any]
    cleaning_log: list[str]
    analytics: dict[str, Any]
    insights: str
    dashboard_plan: dict[str, Any]
    error: str
