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
    score = trainer.model.predict_proba(data)[:, 1][0]
    return {"predicted_score": float(score)}


@app.post('/add_data')
async def predict_score(cust_data: Request):
    data = await cust_data.json()
    try:
        data = pd.DataFrame.from_dict(data, orient='index').T
        X = data[[col for col in data.columns if col != 'Cost Matrix(Risk)']]
        y = list(data['Cost Matrix(Risk)'])
        X = trainer.preprocess(X)
        logger.info(f"Shape before : {trainer.X_train.shape}")
        trainer.X_train = pd.concat([trainer.X_train, X])
        logger.info(f"Shape after : {trainer.X_train.shape}")
        trainer.y_train = np.array(list(trainer.y_train) + y)
        return {"status": "SUCCESS"}
    except Exception as e:
        return {"status": "FAILURE", "error": e}


@app.get('/train')
async def start_training():
    result = trainer.train()
    return result
