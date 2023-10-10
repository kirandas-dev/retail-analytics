from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd


app = FastAPI()

forecast_pipe = load('../Models/Forecasting/arima_pipeline.joblib')


# Solution:
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get('/health', status_code=200)
def healthcheck():
    return 'Forecasting model is all ready to go!'