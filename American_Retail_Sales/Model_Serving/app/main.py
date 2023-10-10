from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
import os 
from American_Retail_Sales.src.models.forecast_predict_model import ARIMATransformer

models_dir = "models/forecasting"  # Update the directory name
os.makedirs(models_dir, exist_ok=True)
model_filename = os.path.join(models_dir, f'arima_pipeline.joblib')
print(model_filename)
app = FastAPI()

# Load your historical data
train_data = pd.read_csv('https://raw.githubusercontent.com/kirandas-dev/data-ML/main/combined_time_series.csv', low_memory=False)

# Convert the 'date' column in your DataFrame to Timestamp objects
train_data['date'] = pd.to_datetime(train_data['date'])
forecast_pipe = load(model_filename)
# Check if the loading was successful
if forecast_pipe:
    print("Pipeline loaded successfully.")
else:
    print("Pipeline loading failed.")

# Solution:
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get('/health', status_code=200)
def healthcheck():
    return 'Forecasting model is all ready to go!'


@app.get('/national', status_code=200)
def sales_forecast(input_date: str):
    try:
        # Convert the input date to a pandas datetime object
        input_date = pd.to_datetime(input_date)
        
        # Filter historical data up to the input date
        historical_data_up_to_input_date = train_data[train_data['date'] <= input_date]
        
        # Ensure that there's enough data for forecasting
        if len(historical_data_up_to_input_date) < 7:
            return JSONResponse(content={"error": "Not enough historical data for forecasting."}, status_code=400)
        
        # Load the forecasting pipeline
        #forecast_pipe = load('Models/Forecasting/arima_pipeline.joblib')
        
        
        # Use the pipeline to make sales forecasts
        sales_forecast = forecast_pipe.transform(historical_data_up_to_input_date['sales'])
        print ("interim sales_forecast", sales_forecast)
        # Return the sales forecasts as JSON
        return JSONResponse(content={"sales_forecast": sales_forecast.tolist()}, status_code=200)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get('/stores/items/', status_code=200)
def sales_forecast(input_date: str):
    try:
        # Convert the input date to a pandas datetime object
        input_date = pd.to_datetime(input_date)
        
        # Filter historical data up to the input date
        historical_data_up_to_input_date = train_data[train_data['date'] <= input_date]
        
        # Ensure that there's enough data for forecasting
        if len(historical_data_up_to_input_date) < 7:
            return JSONResponse(content={"error": "Not enough historical data for forecasting."}, status_code=400)
        
        # Load the forecasting pipeline
        #forecast_pipe = load('Models/Forecasting/arima_pipeline.joblib')
        
        
        # Use the pipeline to make sales forecasts
        sales_forecast = forecast_pipe.transform(historical_data_up_to_input_date['sales'])
        print ("interim sales_forecast", sales_forecast)
        # Return the sales forecasts as JSON
        return JSONResponse(content={"sales_forecast": sales_forecast.tolist()}, status_code=200)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)