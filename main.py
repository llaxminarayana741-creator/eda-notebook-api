import base64
import io
import os
import traceback
import math

import pandas as pd
import numpy as np
import nbformat

from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbclient import NotebookClient

# =========================
# CONFIG
# =========================
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is not set on this deployment.")

MAX_FILE_SIZE_MB = 15

app = FastAPI(title="EDA Notebook API", version="FINAL-3.0.0")


# =========================
# AUTH
# =========================
def verify_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0] != "Bearer":
        raise HTTPException(status_code=401, detail="Invalid auth format")
    if parts[1] != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid token")


# =========================
# CLEAN DATA
# =========================
def clean_data(df: pd.DataFrame):
    """
    Cleans a dataframe WITHOUT destroying categorical columns.
    - Drops exact duplicate rows
    - Fills numeric NaNs with the column median
    - Fills categorical NaNs with the column mode
    - Caps (winsorizes) numeric outliers using the IQR method instead of dropping rows
    """
    before = len(df)
    df = df.drop_duplicates()

    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().any():
            mode = df[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            df[col] = df[col].fillna(fill_value)

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)

    after = len(df)
    return df, before, after


# =========================
# NOTEBOOK BUILDER
# =========================
async def build_notebook(df: pd.DataFrame, csv_b64: str) -> str:
    nb = new_notebook()
    cells = []

    cells.append(new_markdown_cell("# 📊 Automated EDA Report"))

    cells.append(new_code_cell(
        "import base64, io\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        f'csv_b64 = "{csv_b64}"\n'
        "df = pd.read_csv(io.StringIO(base64.b64decode(csv_b64).decode('utf-8')))\n"
        "df.head()"
    ))

    cells.append(new_markdown_cell("## Dataset Overview"))
    cells.append(new_code_cell('print("Shape:", df.shape)\nprint("Columns:", df.columns.tolist())'))

    cells.append(new_markdown_cell("## Dataset Info"))
    cells.append(new_code_cell("df.info()"))

    cells.append(new_markdown_cell("## Summary Statistics"))
    cells.append(new_code_cell("df.describe(include='all')"))

    cells.append(new_markdown_cell("## Missing Values"))
    cells.append(new_code_cell(
        "missing = df.isnull().sum()\n"
        "percent = (missing / len(df)) * 100\n"
        'pd.DataFrame({"Missing": missing, "Percent": percent})'
    ))

    cells.append(new_markdown_cell("## Duplicate Rows"))
    cells.append(new_code_cell('print("Duplicate rows:", df.duplicated().sum())\ndf[df.duplicated()]'))

    cells.append(new_markdown_cell("## Column Types"))
    cells.append(new_code_cell(
        "num_cols = df.select_dtypes(include=np.number).columns\n"
        "cat_cols = df.select_dtypes(exclude=np.number).columns\n"
        'print("Numerical Columns:", list(num_cols))\n'
        'print("Categorical Columns:", list(cat_cols))'
    ))

    cells.append(new_markdown_cell("## Categorical Value Counts"))
    cells.append(new_code_cell(
        "for col in cat_cols:\n"
        '    print(f"\\nColumn: {col}")\n'
        "    print(df[col].value_counts())"
    ))

    cells.append(new_markdown_cell("## Histograms"))
    cells.append(new_code_cell(
        "if len(num_cols) > 0:\n"
        "    df[num_cols].hist(figsize=(12, 3 * ((len(num_cols) - 1) // 3 + 1)))\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "else:\n"
        "    print('No numeric columns to plot.')"
    ))

    cells.append(new_markdown_cell("## Boxplots"))
    cells.append(new_code_cell(
        "if len(num_cols) > 0:\n"
        "    n = len(num_cols)\n"
        "    cols = 3\n"
        "    rows = (n - 1) // cols + 1\n"
        "    df[num_cols].plot(kind='box', subplots=True, layout=(rows, cols), figsize=(12, 3 * rows))\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "else:\n"
        "    print('No numeric columns to plot.')"
    ))

    cells.append(new_markdown_cell("## Correlation Heatmap"))
    cells.append(new_code_cell(
        "if len(num_cols) > 1:\n"
        "    plt.figure(figsize=(10, 6))\n"
        "    sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')\n"
        "    plt.show()\n"
        "else:\n"
        "    print('Not enough numeric columns for a correlation heatmap.')"
    ))

    cells.append(new_markdown_cell("## Scatter Plots (Top Correlated Pairs)"))
    cells.append(new_code_cell(
        "if len(num_cols) >= 2:\n"
        "    corr = df[num_cols].corr().abs()\n"
        "    pairs = (\n"
        "        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))\n"
        "        .stack()\n"
        "        .sort_values(ascending=False)\n"
        "        .head(3)\n"
        "    )\n"
        "    for (a, b), _ in pairs.items():\n"
        "        plt.figure(figsize=(6, 4))\n"
        "        sns.scatterplot(x=df[a], y=df[b])\n"
        "        plt.title(f'{a} vs {b}')\n"
        "        plt.show()\n"
        "else:\n"
        "    print('Not enough numeric columns for scatter plots.')"
    ))

    nb["cells"] = cells

    client = NotebookClient(nb, timeout=120, kernel_name="python3")
    await client.async_execute()

    return nbformat.writes(nb)


# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {"status": "ok", "service": "EDA Notebook API", "version": "FINAL-3.0.0"}


@app.post("/run")
async def run(file: UploadFile = File(...), _: None = Depends(verify_token)):
    try:
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Upload CSV only")

        content = await file.read()

        if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit")

        try:
            df = pd.read_csv(io.BytesIO(content), encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(content), encoding="latin1")

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty")

        df, before, after = clean_data(df)

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_b64 = base64.b64encode(csv_buffer.getvalue().encode()).decode()

        notebook_json = await build_notebook(df, csv_b64)
        notebook_b64 = base64.b64encode(notebook_json.encode()).decode()

        return JSONResponse({
            "rows_before": before,
            "rows_after": after,
            "csv_file": csv_b64,
            "notebook_file": notebook_b64,
        })

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )
