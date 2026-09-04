from fastapi import FastAPI, UploadFile, File, Header, HTTPException
import pandas as pd
import io

app = FastAPI()

API_KEY = "mysecretkey"

@app.post("/run")
async def run(file: UploadFile = File(...), authorization: str = Header(None)):

    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    df = df.drop_duplicates()
    df = df.fillna(method="ffill")

    df.to_csv("cleaned_data.csv", index=False)

    return {"message": "Success"}
