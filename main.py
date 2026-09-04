import base64
import io
import os

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

app = FastAPI()


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
# DATA CLEANING (ADVANCED)
# =========================
def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df = df.ffill().bfill()

    # Convert numeric columns properly
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    # Remove outliers using IQR (for numeric columns)
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    return df


# =========================
# NOTEBOOK GENERATION
# =========================
def build_notebook(cleaned_csv):
    nb = new_notebook()
    cells = []

    cells.append(new_markdown_cell("# 📊 Advanced EDA Report"))

    cells.append(new_code_cell(f"""
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns

data = \"\"\"{cleaned_csv}\"\"\"
df = pd.read_csv(io.StringIO(data))

df.head()
"""))

    cells.append(new_markdown_cell("## Dataset Info"))
    cells.append(new_code_cell("df.info()"))

    cells.append(new_markdown_cell("## Statistical Summary"))
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

    cells.append(new_markdown_cell("## Boxplot (Outliers)"))
    cells.append(new_code_cell("""
df.plot(kind='box', subplots=True, layout=(4,4), figsize=(12,10))
plt.show()
"""))

    nb["cells"] = cells
    return nbformat.writes(nb)


# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {"status": "API running"}


@app.post("/run")
async def run(file: UploadFile = File(...), _: None = Depends(verify_token)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload CSV only")

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    # Clean data
    df = clean_data(df)

    # Convert cleaned CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_text = csv_buffer.getvalue()
    csv_b64 = base64.b64encode(csv_text.encode()).decode()

    # Notebook
    notebook = build_notebook(csv_text)
    notebook_b64 = base64.b64encode(notebook.encode()).decode()

    return JSONResponse({
        "csv_file": csv_b64,
        "notebook_file": notebook_b64
    })
