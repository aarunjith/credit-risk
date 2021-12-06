from typing import Optional
import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from loguru import logger
from trainer import Trainer

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

trainer = Trainer()


@app.post('/predict')
async def predict_score(cust_data: Request):
    data = await cust_data.json()
    data = pd.DataFrame.from_dict(data, orient='index').T
    data = trainer.preprocess(data)
    logger.info(data)


@app.get('/train')
async def start_training():
    pass
