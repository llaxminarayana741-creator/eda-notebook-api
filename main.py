import base64
import io
import os

import pandas as pd
import numpy as np
import nbformat
import matplotlib.pyplot as plt
import seaborn as sns

from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
from nbformat.v4 import new_notebook, new_markdown_cell

# =========================
# CONFIG
# =========================
API_KEY = os.getenv("API_KEY", "mysecretkey")

app = FastAPI(title="EDA Notebook API", version="5.0.0")


# =========================
# AUTH (FIXED + DEBUG)
# =========================
def verify_token(authorization: str = Header(None)):
    print("===== AUTH DEBUG =====")
    print("HEADER RECEIVED:", authorization)
    print("EXPECTED API_KEY:", API_KEY)

    if not authorization:
        raise HTTPException(status_code=401, detail="No Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid format")

    token = authorization.split(" ")[1]
    print("TOKEN RECEIVED:", token)

    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid token")


# =========================
# DATA CLEANING
# =========================
def clean_data(df):
    original_rows = len(df)

    # remove duplicates
    df = df.drop_duplicates()

    # fill missing
    df = df.ffill().bfill()

    # safe numeric conversion
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass

    # remove outliers (max 5%)
    numeric_cols = df.select_dtypes(include=np.number).columns
    removed_rows = 0

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        before = len(df)
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
        after = len(df)

        removed_rows += (before - after)

    print(f"Rows before: {original_rows}, after cleaning: {len(df)}")

    return df


# =========================
# NOTEBOOK GENERATION
# =========================
def build_notebook(df):
    nb = new_notebook()
    cells = []

    cells.append(new_markdown_cell("# 📊 Automated EDA Report"))

    # Preview
    cells.append(new_markdown_cell("## Dataset Preview"))
    cells.append(new_markdown_cell(df.head().to_html()))

    # Info
    buffer = io.StringIO()
    df.info(buf=buffer)
    cells.append(new_markdown_cell("## Dataset Info"))
    cells.append(new_markdown_cell(f"```\n{buffer.getvalue()}\n```"))

    # Describe
    cells.append(new_markdown_cell("## Summary Statistics"))
    cells.append(new_markdown_cell(df.describe(include='all').to_html()))

    # Missing values
    cells.append(new_markdown_cell("## Missing Values"))
    cells.append(new_markdown_cell(df.isnull().sum().to_frame().to_html()))

    # ================= GRAPHS =================

    # Histogram
    plt.figure(figsize=(10,6))
    df.hist(figsize=(12,8))
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    img = base64.b64encode(buf.getvalue()).decode()
    cells.append(new_markdown_cell("## Distribution"))
    cells.append(new_markdown_cell(f"![Histogram](data:image/png;base64,{img})"))

    # Boxplot
    plt.figure(figsize=(10,6))
    df.plot(kind='box', subplots=True, layout=(4,4), figsize=(12,10))
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    img = base64.b64encode(buf.getvalue()).decode()
    cells.append(new_markdown_cell("## Boxplots"))
    cells.append(new_markdown_cell(f"![Boxplot](data:image/png;base64,{img})"))

    # Heatmap
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    img = base64.b64encode(buf.getvalue()).decode()
    cells.append(new_markdown_cell("## Correlation Heatmap"))
    cells.append(new_markdown_cell(f"![Heatmap](data:image/png;base64,{img})"))

    nb["cells"] = cells
    return nbformat.writes(nb)


# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {
        "status": "ok",
        "service": "EDA Notebook API",
        "version": "5.0.0"
    }


@app.post("/run")
async def run(file: UploadFile = File(...), _: None = Depends(verify_token)):

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload CSV only")

    content = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV read error: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV is empty")

    # clean
    df = clean_data(df)

    # CSV output
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_b64 = base64.b64encode(csv_buffer.getvalue().encode()).decode()

    # Notebook
    notebook_json = build_notebook(df)
    notebook_b64 = base64.b64encode(notebook_json.encode()).decode()

    return JSONResponse({
        "csv_file": csv_b64,
        "notebook_file": notebook_b64
    })
