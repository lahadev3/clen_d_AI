import pandas as pd

def clean_dataframe(
    df: pd.DataFrame,
    fill_missing: bool = True,
    remove_duplicates: bool = True,
    trim_text: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Conservative, reproducible cleaning. Never mutates the input DataFrame."""
    cleaned = df.copy(deep=True)
    log: list[str] = []
    rows_before, cols_before = cleaned.shape

    # Normalize column labels while retaining a readable name.
    old_columns = list(cleaned.columns)
    cleaned.columns = [str(c).strip() or f"unnamed_{i+1}" for i, c in enumerate(cleaned.columns)]
    if old_columns != list(cleaned.columns):
        log.append("Trimmed whitespace from column names and renamed blank column labels.")

    if trim_text:
        text_cols = cleaned.select_dtypes(include=["object", "string"]).columns
        for col in text_cols:
            cleaned[col] = cleaned[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        log.append(f"Trimmed surrounding whitespace in {len(text_cols)} text column(s).")

    # Convert columns that are mostly numeric strings to numeric values.
    converted = []
    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        series = cleaned[col]
        non_null = series.notna().sum()
        if non_null:
            candidate = pd.to_numeric(series, errors="coerce")
            if candidate.notna().sum() / non_null >= 0.85:
                cleaned[col] = candidate
                converted.append(col)
    if converted:
        log.append("Converted mostly numeric text columns: " + ", ".join(map(str, converted)) + ".")

    if remove_duplicates:
        duplicate_count = int(cleaned.duplicated().sum())
        cleaned = cleaned.drop_duplicates().copy()
        log.append(f"Removed {duplicate_count} exact duplicate row(s).")

    if fill_missing:
        filled = []
        for col in cleaned.columns:
            missing = int(cleaned[col].isna().sum())
            if not missing:
                continue
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                median = cleaned[col].median()
                if pd.notna(median):
                    cleaned[col] = cleaned[col].fillna(median)
                    filled.append(f"{col}: {missing} missing value(s) filled with median")
            else:
                modes = cleaned[col].mode(dropna=True)
                if not modes.empty:
                    cleaned[col] = cleaned[col].fillna(modes.iloc[0])
                    filled.append(f"{col}: {missing} missing value(s) filled with mode")
        log.extend(filled or ["No fillable missing values found."])
    else:
        log.append("Missing-value filling was disabled; remaining missing values were preserved.")

    log.append(f"Input shape: {rows_before:,} rows × {cols_before:,} columns.")
    log.append(f"Cleaned shape: {cleaned.shape[0]:,} rows × {cleaned.shape[1]:,} columns.")
    return cleaned, log
