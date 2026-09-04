import base64
import io
import os
import traceback

import pandas as pd
import numpy as np
import nbformat
from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

# =========================
# CONFIG
# =========================
API_KEY = os.getenv("API_KEY", "mysecretkey")

app = FastAPI(title="EDA Notebook API", version="5.0.0")

# =========================
# AUTH
# =========================
def verify_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth format")

    token = authorization.split(" ")[1]

    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid token")

# =========================
# SAFE DATA CLEANING
# =========================
def clean_data(df):
    before_rows = len(df)

    # remove duplicates
    df = df.drop_duplicates()

    # handle missing values safely
    df = df.ffill().bfill()

    # convert numeric safely (FIXED)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass

    # fill NaN created by coercion
    df = df.fillna(0)

    after_rows = len(df)

    return df, before_rows, after_rows

# =========================
# NOTEBOOK CREATION
# =========================
def build_notebook():
    nb = new_notebook()
    cells = []

    cells.append(new_markdown_cell("# 📊 Automated EDA Report"))

    cells.append(new_code_cell("""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cleaned_data.csv")

print("Shape:", df.shape)

display(df.head())
"""))

    cells.append(new_markdown_cell("## Dataset Info"))
    cells.append(new_code_cell("df.info()"))

    cells.append(new_markdown_cell("## Summary Statistics"))
    cells.append(new_code_cell("display(df.describe(include='all'))"))

    cells.append(new_markdown_cell("## Missing Values"))
    cells.append(new_code_cell("display(df.isnull().sum())"))

    cells.append(new_markdown_cell("## Histograms"))
    cells.append(new_code_cell("""
df.hist(figsize=(12,8))
plt.show()
"""))

    cells.append(new_markdown_cell("## Boxplots"))
    cells.append(new_code_cell("""
df.plot(kind='box', subplots=True, layout=(4,4), figsize=(12,10))
plt.show()
"""))

    cells.append(new_markdown_cell("## Correlation Heatmap"))
    cells.append(new_code_cell("""
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()
"""))

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

    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Upload CSV only")

        content = await file.read()

        # SAFE CSV LOAD (FIXED)
        try:
            df = pd.read_csv(io.BytesIO(content), encoding="utf-8")
        except:
            df = pd.read_csv(io.BytesIO(content), encoding="latin1")

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty")

        # CLEAN
        df, before, after = clean_data(df)

        if df.empty:
            raise HTTPException(status_code=400, detail="All data removed after cleaning")

        # SAVE CLEAN CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_text = csv_buffer.getvalue()
        csv_b64 = base64.b64encode(csv_text.encode()).decode()

        # NOTEBOOK
        notebook_json = build_notebook()
        notebook_b64 = base64.b64encode(notebook_json.encode()).decode()

        return JSONResponse({
            "rows_before": before,
            "rows_after": after,
            "cleaned_csv": csv_b64,
            "notebook_file": notebook_b64
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "trace": traceback.format_exc()
            }
        )
