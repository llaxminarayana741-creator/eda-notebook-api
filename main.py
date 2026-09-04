from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends
import pandas as pd
import io
import base64
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

app = FastAPI()

# 🔐 API KEY (use env in production)
API_KEY = os.getenv("API_KEY", "mysecretkey")


# ✅ AUTH FUNCTION
def verify_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = authorization.split(" ")[1]
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# 🧹 CLEAN DATA
def clean_data(df):
    original_shape = df.shape

    df = df.drop_duplicates()
    df = df.ffill().bfill()

    summary = {
        "rows_before": original_shape[0],
        "rows_after": df.shape[0],
        "columns": list(df.columns),
        "missing_after": int(df.isnull().sum().sum())
    }

    return df, summary


# 📓 CREATE NOTEBOOK
def create_notebook(csv_text, summary):
    nb = new_notebook()

    cells = []

    cells.append(new_markdown_cell("# EDA Report"))

    cells.append(new_code_cell(
        "import pandas as pd\n"
        "import io\n"
        f"csv_data = '''{csv_text}'''\n"
        "df = pd.read_csv(io.StringIO(csv_data))\n"
        "df.head()"
    ))

    cells.append(new_markdown_cell("## Summary"))

    cells.append(new_code_cell(f"summary = {summary}\nsummary"))

    cells.append(new_code_cell(
        "df.describe()"
    ))

    cells.append(new_code_cell(
        "import matplotlib.pyplot as plt\n"
        "df.hist(figsize=(10,6))\n"
        "plt.show()"
    ))

    nb["cells"] = cells

    return nbformat.writes(nb)


# 🚀 MAIN API
@app.post("/run")
async def run(
    file: UploadFile = File(...),
    _: None = Depends(verify_token)
):

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        if df.empty:
            raise HTTPException(status_code=400, detail="Empty CSV")

        # 🧹 Clean
        df, summary = clean_data(df)

        # 📁 CSV → BASE64
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_text = csv_buffer.getvalue()

        csv_base64 = base64.b64encode(csv_text.encode()).decode()

        # 📓 NOTEBOOK → BASE64
        notebook_json = create_notebook(csv_text, summary)
        notebook_base64 = base64.b64encode(notebook_json.encode()).decode()

        return {
            "csv_file": csv_base64,
            "notebook_file": notebook_base64,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ❤️ HEALTH CHECK
@app.get("/")
def home():
    return {"status": "API Running"}
