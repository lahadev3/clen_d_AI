from io import BytesIO
import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

def load_dataframe(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    if name.endswith(".csv"):
        # Try common encodings; pandas will raise a useful error if none work.
        last_error = None
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(BytesIO(raw), encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
        raise ValueError(f"Could not decode CSV file: {last_error}")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(BytesIO(raw))
    raise ValueError("Unsupported file. Upload a CSV or Excel (.xlsx/.xls) file.")
