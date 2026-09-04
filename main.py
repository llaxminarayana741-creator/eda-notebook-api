import base64
import io
import os

import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Depends
from fastapi.responses import JSONResponse

# =========================
# CONFIG
# =========================
API_KEY = os.getenv("API_KEY", "mysecretkey")

app = FastAPI(title="EDA Notebook API", version="5.0.0")


# =========================
# AUTH
# =========================
def verify_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = authorization.split(" ")[1]
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid token")


# =========================
# SAFE CLEANING (FIXED)
# =========================
def clean_data(df):
    before_rows = len(df)

    # remove duplicates
    df = df.drop_duplicates()

    # fill missing values safely
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].median())

    # convert numeric safely
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass

    # remove outliers safely (DO NOT DROP ALL DATA)
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            continue

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # keep most data (limit removal)
        mask = (df[col] >= lower) & (df[col] <= upper)

        if mask.sum() > 0.9 * len(df):  # keep 90% data
            df = df[mask]

    after_rows = len(df)

    return df, before_rows, after_rows


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
    except:
        raise HTTPException(status_code=400, detail="Invalid CSV file")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV is empty")

    # CLEAN
    df, before, after = clean_data(df)

    if df.empty:
        raise HTTPException(status_code=400, detail="All data removed during cleaning")

    # =========================
    # CREATE CLEAN CSV
    # =========================
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_b64 = base64.b64encode(csv_buffer.getvalue().encode()).decode()

    # =========================
    # GENERATE NOTEBOOK TEXT (ADVANCED)
    # =========================
    notebook_code = f"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

df = pd.read_csv("cleaned_data.csv")

print("Dataset Shape:", df.shape)

# Preview
df.head()

# Info
df.info()

# Summary
df.describe()

# Missing values
df.isnull().sum()

# Histograms
df.hist(figsize=(12,8))
plt.show()

# Boxplots
df.plot(kind='box', subplots=True, layout=(4,4), figsize=(12,10))
plt.show()

# Correlation
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()

# Scatter plots
numeric_cols = df.select_dtypes(include='number').columns

if len(numeric_cols) >= 2:
    sns.pairplot(df[numeric_cols[:5]])
    plt.show()

print("Data cleaned from", {before}, "to", {after}, "rows")
"""

    notebook_b64 = base64.b64encode(notebook_code.encode()).decode()

    return JSONResponse({
        "cleaned_csv": csv_b64,
        "notebook_code": notebook_b64
    })
