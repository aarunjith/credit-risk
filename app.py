from typing import Optional
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from loguru import logger

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.post('/predict')
async def predict_score(cust_data: Request):
    data = await cust_data.json()
    pass
