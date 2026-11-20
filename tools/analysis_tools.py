import pandas as pd
import numpy as np

def profile_dataframe(df: pd.DataFrame) -> dict:
    missing = df.isna().sum()
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "missing_total": int(missing.sum()),
        "missing_by_column": {str(k): int(v) for k, v in missing.items() if v},
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "numeric_columns": [str(c) for c in df.select_dtypes(include=np.number).columns],
        "categorical_columns": [str(c) for c in df.select_dtypes(include=["object", "string", "category", "bool"]).columns],
        "datetime_columns": [str(c) for c in df.select_dtypes(include=["datetime", "datetimetz"]).columns],
    }

def analyze_dataframe(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include=np.number)
    categorical = df.select_dtypes(include=["object", "string", "category", "bool"])
    numeric_summary = {}
    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s):
            numeric_summary[str(col)] = {
                "mean": round(float(s.mean()), 4),
                "median": round(float(s.median()), 4),
                "min": float(s.min()),
                "max": float(s.max()),
                "std": round(float(s.std()), 4) if len(s) > 1 else 0.0,
            }
    categorical_summary = {}
    for col in categorical.columns:
        counts = df[col].astype("string").fillna("(missing)").value_counts().head(5)
        categorical_summary[str(col)] = {str(k): int(v) for k, v in counts.items()}
    corr = numeric.corr(numeric_only=True).round(3) if numeric.shape[1] >= 2 else pd.DataFrame()
    strongest = []
    if not corr.empty:
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                value = corr.iloc[i, j]
                if pd.notna(value):
                    strongest.append({"column_1": str(cols[i]), "column_2": str(cols[j]), "correlation": float(value)})
        strongest.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "strongest_correlations": strongest[:5],
    }
