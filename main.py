import base64
import io
import os

import pandas as pd
import numpy as np
import nbformat
from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

# NEW: for executing notebook
from nbconvert.preprocessors import ExecutePreprocessor

# =========================
# CONFIG
# =========================
API_KEY = os.getenv("API_KEY", "mysecretkey")

app = FastAPI(title="EDA Notebook API", version="2.0.0")


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
# DATA CLEANING
# =========================
def clean_data(df):
    df = df.drop_duplicates()

    # handle missing values
    df = df.ffill().bfill()

    # convert numeric safely
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    # remove outliers (IQR)
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    return df


# =========================
# NOTEBOOK GENERATION
# =========================
def build_notebook(csv_text):
    nb = new_notebook()

    cells = []

    cells.append(new_markdown_cell("# 📊 Automated EDA Report"))

    cells.append(new_code_cell(f"""
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns

cleaned_csv = \"\"\"{csv_text}\"\"\"
df = pd.read_csv(io.StringIO(cleaned_csv))

df.head()
"""))

    cells.append(new_markdown_cell("## Dataset Info"))
    cells.append(new_code_cell("df.info()"))

    cells.append(new_markdown_cell("## Summary Statistics"))
    cells.append(new_code_cell("df.describe(include='all')"))

    cells.append(new_markdown_cell("## Missing Values"))
    cells.append(new_code_cell("df.isnull().sum()"))

    cells.append(new_markdown_cell("## Correlation Heatmap"))
    cells.append(new_code_cell("""
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()
"""))

    cells.append(new_markdown_cell("## Distribution"))
    cells.append(new_code_cell("""
df.hist(figsize=(12,8))
plt.show()
"""))

    cells.append(new_markdown_cell("## Outliers (Boxplot)"))
    cells.append(new_code_cell("""
df.plot(kind='box', subplots=True, layout=(4,4), figsize=(12,10))
plt.show()
"""))

    nb['cells'] = cells
    return nb


# =========================
# EXECUTE NOTEBOOK
# =========================
def execute_notebook(nb):
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {})
    return nb


# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/run")
async def run(file: UploadFile = File(...), _: None = Depends(verify_token)):

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload CSV only")

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    # clean data
    df = clean_data(df)

    # CSV to base64
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_text = csv_buffer.getvalue()
    csv_b64 = base64.b64encode(csv_text.encode()).decode()

    # notebook
    nb = build_notebook(csv_text)

    # EXECUTE notebook (IMPORTANT)
    nb = execute_notebook(nb)

    notebook_json = nbformat.writes(nb)
    notebook_b64 = base64.b64encode(notebook_json.encode()).decode()

    return JSONResponse({
        "csv_file": csv_b64,
        "notebook_file": notebook_b64
    })
