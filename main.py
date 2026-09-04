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

app = FastAPI(title="EDA Notebook API", version="FINAL-2.0.0")


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
# CLEAN DATA
# =========================
def clean_data(df):
    before = len(df)

    df = df.drop_duplicates()
    df = df.ffill().bfill()

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass

    df = df.fillna(0)

    after = len(df)
    return df, before, after


# =========================
# NOTEBOOK BUILDER (ADVANCED)
# =========================
def build_notebook(csv_text):
    nb = new_notebook()
    cells = []

    # TITLE
    cells.append(new_markdown_cell("# 📊 Automated EDA Report"))

    # LOAD DATA
    cells.append(new_code_cell(f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

csv_data = \"\"\"{csv_text}\"\"\"
df = pd.read_csv(io.StringIO(csv_data))

df.head()
"""))

    # OVERVIEW
    cells.append(new_markdown_cell("## Dataset Overview"))
    cells.append(new_code_cell("""
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
"""))

    # INFO
    cells.append(new_markdown_cell("## Dataset Info"))
    cells.append(new_code_cell("df.info()"))

    # SUMMARY
    cells.append(new_markdown_cell("## Summary Statistics"))
    cells.append(new_code_cell("df.describe(include='all')"))

    # MISSING VALUES
    cells.append(new_markdown_cell("## Missing Values"))
    cells.append(new_code_cell("""
missing = df.isnull().sum()
percent = (missing / len(df)) * 100
pd.DataFrame({"Missing": missing, "Percent": percent})
"""))

    # DUPLICATES
    cells.append(new_markdown_cell("## Duplicate Rows"))
    cells.append(new_code_cell("""
print("Duplicate rows:", df.duplicated().sum())
df[df.duplicated()]
"""))

    # COLUMN TYPES
    cells.append(new_markdown_cell("## Column Types"))
    cells.append(new_code_cell("""
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include='object').columns

print("Numerical Columns:", num_cols)
print("Categorical Columns:", cat_cols)
"""))

    # VALUE COUNTS
    cells.append(new_markdown_cell("## Categorical Value Counts"))
    cells.append(new_code_cell("""
for col in cat_cols:
    print(f"\\nColumn: {col}")
    print(df[col].value_counts())
"""))

    # HISTOGRAMS
    cells.append(new_markdown_cell("## Histograms"))
    cells.append(new_code_cell("""
df.hist(figsize=(12,8))
plt.tight_layout()
plt.show()
"""))

    # BOXPLOTS
    cells.append(new_markdown_cell("## Boxplots"))
    cells.append(new_code_cell("""
df.plot(kind='box', subplots=True, layout=(4,4), figsize=(12,10))
plt.tight_layout()
plt.show()
"""))

    # HEATMAP
    cells.append(new_markdown_cell("## Correlation Heatmap"))
    cells.append(new_code_cell("""
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()
"""))

    # SCATTER
    cells.append(new_markdown_cell("## Scatter Plots"))
    cells.append(new_code_cell("""
num_cols = df.select_dtypes(include=np.number).columns

if len(num_cols) >= 2:
    for i in range(min(3, len(num_cols)-1)):
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df[num_cols[i]], y=df[num_cols[i+1]])
        plt.title(f"{num_cols[i]} vs {num_cols[i+1]}")
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
        "version": "FINAL-2.0.0"
    }


@app.post("/run")
async def run(file: UploadFile = File(...), _: None = Depends(verify_token)):
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Upload CSV only")

        content = await file.read()

        try:
            df = pd.read_csv(io.BytesIO(content), encoding="utf-8")
        except:
            df = pd.read_csv(io.BytesIO(content), encoding="latin1")

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty")

        df, before, after = clean_data(df)

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_text = csv_buffer.getvalue()

        csv_b64 = base64.b64encode(csv_text.encode()).decode()

        notebook_json = build_notebook(csv_text)
        notebook_b64 = base64.b64encode(notebook_json.encode()).decode()

        return JSONResponse({
            "rows_before": before,
            "rows_after": after,
            "csv_file": csv_b64,          # IMPORTANT for n8n
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
