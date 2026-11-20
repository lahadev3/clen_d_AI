# AI Data Analyst — Multi-Agent CSV/Excel Analytics

A runnable Streamlit project using LangGraph, LangChain, NVIDIA NIM (`ChatNVIDIA`), Pandas and Plotly.

## Features
- Upload CSV, XLSX or XLS files
- Profile missing values, duplicates, data types and unique values
- Configurable cleaning: trim whitespace, convert mostly numeric text columns, remove exact duplicates, fill missing numeric values with median and categorical values with mode
- LangGraph workflow with profiling, cleaning, analytics, insight and dashboard-planning nodes
- Interactive dashboard with KPI cards, category breakdown, numeric distributions, scatter plots, filters, and date trends when applicable
- Download cleaned data as CSV or Excel
- NVIDIA-generated insights and natural-language Q&A when an API key is configured
- Works without NVIDIA key for core data profiling, cleaning, analytics and charts

## Requirements
Python 3.10+ recommended. An internet connection is needed to install packages and to call NVIDIA.

## Setup (Windows PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```
Edit `.env` and set `NVIDIA_API_KEY` to a key from https://build.nvidia.com/. Set `NVIDIA_MODEL` to a model currently available to your NVIDIA API account. Never commit `.env`.

## Run
```powershell
streamlit run app.py
```
Open the local URL printed by Streamlit, usually http://localhost:8501.

## Workflow
`START -> profile -> clean -> analyze -> insights -> dashboard_plan -> END`

The graph stores intermediate artifacts in shared state. Pandas performs deterministic calculations and transformations; NVIDIA explains computed results rather than being trusted to calculate data values.

## Cleaning behavior
- Exact duplicate rows are removed only when the checkbox is enabled.
- Numeric missing values are filled with the median; non-numeric missing values use the mode when one exists.
- Values are not automatically removed as outliers.
- The uploaded source is never modified in place. Review the cleaning log and download the cleaned result.
- Some Excel files with unusual formatting, merged headers, multiple sheets, formulas, or very large size may need custom import logic.

## Troubleshooting
- NVIDIA authentication/model errors: verify the API key and model ID are enabled for your account.
- Excel import error: confirm `openpyxl` installed and the workbook is not password-protected.
- If PowerShell blocks activation, use Command Prompt and run `venv\Scripts\activate.bat`.
