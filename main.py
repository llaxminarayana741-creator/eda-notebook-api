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
    # Fail loudly at startup instead of silently accepting a hardcoded default.
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

    # Numeric: median fill
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Categorical: mode fill (guard against an all-null column)
    for col in categorical_cols:
        if df[col].isnull().any():
            mode = df[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            df[col] = df[col].fillna(fill_value)

    # IQR-based outlier capping (not dropping) for numeric columns
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
def _pick_columns(df: pd.DataFrame):
    """
    Auto-detect which columns drive the template, so the same notebook
    structure works on any uploaded CSV instead of hardcoded column names
    like "Churn" / "Contract" / "MonthlyCharges".
    """
    numeric_cols = list(df.select_dtypes(include=np.number).columns)
    categorical_cols = list(df.select_dtypes(include="object").columns)

    # Exclude ID-like numeric columns (e.g. "customerID" stored as an int,
    # or a fully-unique row index) from being picked as an analysis column —
    # they're identifiers, not measurements, and would produce meaningless
    # "average by group" charts.
    def _looks_like_id(col):
        name = col.lower()
        if "id" in name or name in ("index", "unnamed: 0"):
            return True
        # Only flag as an index column if it's a perfectly sequential integer
        # run (e.g. 0..n-1 or 1..n) — a real measurement (like TotalCharges)
        # can also be all-unique but isn't sequential, so it's kept.
        series = df[col]
        if pd.api.types.is_integer_dtype(series) and series.nunique() == len(series):
            sorted_vals = series.sort_values().to_numpy()
            if len(sorted_vals) > 1 and (sorted_vals[1:] - sorted_vals[:-1] == 1).all():
                return True
        return False

    numeric_cols = [c for c in numeric_cols if not _looks_like_id(c)]

    # Target: prefer a binary categorical column (like "Churn"), else the
    # last categorical column with a reasonable number of categories.
    target_col = None
    for c in categorical_cols:
        if df[c].nunique() == 2:
            target_col = c
            break
    if target_col is None:
        for c in reversed(categorical_cols):
            if 2 <= df[c].nunique() <= 15:
                target_col = c
                break

    # A second grouping column (like "Contract"), distinct from the target,
    # with a manageable number of categories for count plots / crosstabs.
    group_col = None
    for c in categorical_cols:
        if c != target_col and 2 <= df[c].nunique() <= 15:
            group_col = c
            break

    primary_num = numeric_cols[0] if len(numeric_cols) > 0 else None
    secondary_num = numeric_cols[1] if len(numeric_cols) > 1 else None

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "target_col": target_col,
        "group_col": group_col,
        "primary_num": primary_num,
        "secondary_num": secondary_num,
    }


async def build_notebook(df: pd.DataFrame, csv_b64: str) -> str:
    cols = _pick_columns(df)
    num1, num2 = cols["primary_num"], cols["secondary_num"]
    target, group = cols["target_col"], cols["group_col"]

    nb = new_notebook()
    cells = []

    # ---- Load Dataset ----
    cells.append(new_markdown_cell("Load Dataset"))
    load_cell = new_code_cell(
        "#@title Load Dataset (click to view code) { display-mode: \"form\" }\n"
        "import base64, io\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        f'csv_b64 = "{csv_b64}"\n'
        "df = pd.read_csv(io.StringIO(base64.b64decode(csv_b64).decode('utf-8')))\n\n"
        "df.head()"
    )
    # Collapse this cell's source by default: Colab respects "cellView": "form"
    # (shows a "Show code" toggle instead of the raw text), and JupyterLab/
    # Notebook 7 respect metadata.jupyter.source_hidden for the same effect.
    load_cell["metadata"]["cellView"] = "form"
    load_cell["metadata"]["jupyter"] = {"source_hidden": True}
    cells.append(load_cell)

    # ---- dataset dimensions ----
    cells.append(new_markdown_cell("dataset dimensions (rows columns)"))
    cells.append(new_code_cell("df.shape\ndf.columns"))

    # ---- dataset information ----
    cells.append(new_markdown_cell("dataset information"))
    cells.append(new_code_cell(
        "df.info()\n"
        "df.dtypes\n"
        "df.describe()\n"
        "df.describe(include=\"object\")\n"
        "df.nunique()"
    ))

    # ---- Missing values ----
    cells.append(new_markdown_cell("Missing values"))
    cells.append(new_code_cell("df.isnull().sum()"))
    cells.append(new_code_cell("(df.isnull().sum() / len(df)) * 100"))

    # ---- Duplicate rows ----
    cells.append(new_markdown_cell("Duplicate rows"))
    cells.append(new_code_cell("df.duplicated().sum()\ndf[df.duplicated()]"))

    # ---- numerical and categorical columns ----
    cells.append(new_markdown_cell("numerical and categorical columns"))
    cells.append(new_code_cell(
        "numerical_columns = df.select_dtypes(include=np.number).columns\n"
        "categorical_columns = df.select_dtypes(include=\"object\").columns\n\n"
        "print(\"Numerical Columns:\")\n"
        "print(numerical_columns)\n\n"
        "print(\"\\nCategorical Columns:\")\n"
        "print(categorical_columns)"
    ))

    cells.append(new_code_cell(
        "for column in categorical_columns:\n"
        "    print(column)\n"
        "    print(df[column].unique())\n"
        "    print()"
    ))

    cells.append(new_code_cell(
        "for column in categorical_columns:\n"
        "    print(df[column].value_counts())\n"
        "    print()"
    ))

    # ---- Histograms / Distogram / Box plots (primary numeric column) ----
    if num1:
        cells.append(new_markdown_cell("Histograms"))
        cells.append(new_code_cell(
            f'plt.figure(figsize=(8, 5))\n'
            f'plt.hist(df["{num1}"], bins=30, edgecolor="black")\n\n'
            f'plt.xlabel("{num1}")\n'
            f'plt.ylabel("Frequency")\n'
            f'plt.title("Distribution of {num1}")\n\n'
            f'plt.show()'
        ))

        cells.append(new_markdown_cell("Distogram"))
        cells.append(new_code_cell(
            f'plt.figure(figsize=(8, 5))\n\n'
            f'sns.histplot(df["{num1}"], kde=True)\n\n'
            f'plt.title("Distribution of {num1}")\n'
            f'plt.show()'
        ))

        cells.append(new_markdown_cell("Box plots"))
        cells.append(new_code_cell(
            f'plt.figure(figsize=(8, 4))\n\n'
            f'sns.boxplot(x=df["{num1}"])\n\n'
            f'plt.title("Box Plot of {num1}")\n'
            f'plt.show()'
        ))

    # ---- Count plot (target + group column) ----
    if target or group:
        cells.append(new_markdown_cell("Count plot"))
        code = ""
        if target:
            code += (
                f'plt.figure(figsize=(6, 4))\n\n'
                f'sns.countplot(x="{target}", data=df)\n\n'
                f'plt.title("{target} Distribution")\n'
                f'plt.show()\n\n'
            )
        if group:
            code += (
                f'plt.figure(figsize=(8, 5))\n\n'
                f'sns.countplot(x="{group}", data=df)\n\n'
                f'plt.title("Customers by {group}")\n'
                f'plt.show()'
            )
        cells.append(new_code_cell(code.strip()))

    # ---- Pie chart (target column) ----
    if target:
        cells.append(new_markdown_cell("Pie chart"))
        cells.append(new_code_cell(
            f'df["{target}"].value_counts().plot(\n'
            f'    kind="pie",\n'
            f'    autopct="%1.1f%%",\n'
            f'    figsize=(6, 6)\n'
            f')\n\n'
            f'plt.title("{target} Percentage")\n'
            f'plt.ylabel("")\n'
            f'plt.show()'
        ))

    # ---- Bivariate analysis ----
    cells.append(new_markdown_cell("Bivariate analysis"))
    if num1 and num2:
        cells.append(new_code_cell(
            f'plt.figure(figsize=(8, 5))\n\n'
            f'plt.scatter(df["{num1}"], df["{num2}"], alpha=0.5)\n\n'
            f'plt.xlabel("{num1}")\n'
            f'plt.ylabel("{num2}")\n'
            f'plt.title("{num1} vs {num2}")\n\n'
            f'plt.show()'
        ))
    if target and num1:
        cells.append(new_code_cell(
            f'plt.figure(figsize=(8, 5))\n\n'
            f'sns.boxplot(x="{target}", y="{num1}", data=df)\n\n'
            f'plt.title("{num1} by {target}")\n'
            f'plt.show()'
        ))
    if group and target:
        cells.append(new_code_cell(
            f'plt.figure(figsize=(9, 5))\n\n'
            f'sns.countplot(x="{group}", hue="{target}", data=df)\n\n'
            f'plt.title("{group} vs {target}")\n'
            f'plt.show()'
        ))
        cells.append(new_code_cell(f'pd.crosstab(df["{group}"], df["{target}"])'))
        cells.append(new_code_cell(
            f'pd.crosstab(\n'
            f'    df["{group}"],\n'
            f'    df["{target}"],\n'
            f'    normalize="index"\n'
            f') * 100'
        ))

    # ---- Group based analysis ----
    if group and num1:
        cells.append(new_markdown_cell("Grp based analysis"))
        cells.append(new_code_cell(f'df.groupby("{group}")["{num1}"].mean()'))
        cells.append(new_code_cell(
            f'df.groupby("{group}")["{num1}"].mean().plot(\n'
            f'    kind="bar",\n'
            f'    figsize=(8, 5)\n'
            f')\n\n'
            f'plt.ylabel("Average {num1}")\n'
            f'plt.title("Average {num1} by {group}")\n'
            f'plt.show()'
        ))

    # ---- Average <numeric> by <target> ----
    if target and num1:
        cells.append(new_markdown_cell(f"Average {num1} by {target}"))
        cells.append(new_code_cell(f'df.groupby("{target}")["{num1}"].mean()'))
        cells.append(new_code_cell(
            f'df.groupby("{target}")["{num1}"].mean().plot(\n'
            f'    kind="bar",\n'
            f'    figsize=(6, 4)\n'
            f')\n\n'
            f'plt.ylabel("Average {num1}")\n'
            f'plt.title("Average {num1} by {target}")\n'
            f'plt.show()'
        ))

    # ---- Multivariate analysis ----
    cells.append(new_markdown_cell("Multivariate analysis"))
    cells.append(new_code_cell(
        "numeric_df = df.select_dtypes(include=np.number)\n\n"
        "correlation = numeric_df.corr()\n\n"
        "correlation"
    ))
    cells.append(new_code_cell(
        "plt.figure(figsize=(10, 6))\n\n"
        "sns.heatmap(\n"
        "    correlation,\n"
        "    annot=True,\n"
        "    cmap=\"coolwarm\",\n"
        "    fmt=\".2f\"\n"
        ")\n\n"
        "plt.title(\"Correlation Heatmap\")\n"
        "plt.show()"
    ))

    # ---- Pair plot ----
    if len(cols["numeric_cols"]) >= 2:
        cells.append(new_markdown_cell("Pair plot"))
        num_list = cols["numeric_cols"][:4]
        num_list_str = ", ".join(f'"{c}"' for c in num_list)
        cells.append(new_code_cell(
            f"sns.pairplot(\n"
            f"    df[[{num_list_str}]].dropna()\n"
            f")\n\n"
            f"plt.show()"
        ))
        if target:
            with_target_str = ", ".join(f'"{c}"' for c in num_list + [target])
            cells.append(new_code_cell(
                f"sns.pairplot(\n"
                f"    df[[{with_target_str}]].dropna(),\n"
                f'    hue="{target}"\n'
                f")\n\n"
                f"plt.show()"
            ))

    # ---- Missing Values Visualization ----
    cells.append(new_markdown_cell("Missing Values Visualization"))
    cells.append(new_code_cell(
        "plt.figure(figsize=(12, 5))\n\n"
        "df.isnull().sum().plot(kind=\"bar\")\n\n"
        "plt.xlabel(\"Columns\")\n"
        "plt.ylabel(\"Missing Values\")\n"
        "plt.title(\"Missing Values in Dataset\")\n\n"
        "plt.show()"
    ))
    cells.append(new_code_cell(
        "missing_percentage = df.isnull().mean() * 100\n\n"
        "missing_percentage[missing_percentage > 0]"
    ))

    # ---- Target variable analysis ----
    if target:
        cells.append(new_markdown_cell("Target variable analysis"))
        cells.append(new_code_cell(f'df["{target}"].value_counts()'))
        cells.append(new_code_cell(f'df["{target}"].value_counts(normalize=True) * 100'))

    nb["cells"] = cells

    # Actually execute the notebook server-side so graphs/tables are baked
    # into the delivered .ipynb, instead of shipping unexecuted source cells.
    # Uses async_execute() (not the sync execute()) because this function is
    # awaited from inside an async FastAPI route — the sync version tries to
    # spawn its own event loop + signal handlers on a non-main thread and
    # crashes with "add_signal_handler() can only be called from the main thread".
    client = NotebookClient(nb, timeout=180, kernel_name="python3")
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
