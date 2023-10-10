from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
import os 
from src.models.forecast_predict_model import ARIMATransformer
import joblib

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
def sales_predict(input_date: str, item_id: str, store_id: str):
    # Define the store names and groups
    store_names = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
    group1 = ['CA_1', 'CA_2', 'CA_3', 'CA_4']
    group2 = ['TX_1', 'TX_2', 'TX_3']
    group3 = ['WI_1', 'WI_2', 'WI_3']

    # Convert input_date to a datetime object
    input_date = pd.to_datetime(input_date)

    # Extract date components
    day_of_week = input_date.dayofweek
    month = input_date.month
    year = input_date.year

    # Determine the model_group based on the store_id
    if store_id in group1:
        model_group = 1
    elif store_id in group2:
        model_group = 2
    elif store_id in group3:
        model_group = 3
    else:
        # Handle the case when the store_id doesn't match any group
        model_group = None

    if model_group is not None:
        # Load the trained model for the determined model_group
        model_file_path = f"../Model_Serving/Models/predictive/model_group_{model_group}.joblib"
        loaded_model = joblib.load(model_file_path)

        # Prepare the input data for prediction
        input_data = pd.DataFrame({
            'day_of_week': [day_of_week],
            'month': [month],
            'year': [year],
            'store_id': [store_id],
            'item_id': [item_id]
        })

        # Perform target encoding on categorical features using the loaded encoders
        for feature, encoder in loaded_model['encoders'].items():
            input_data[feature] = encoder.transform(input_data[feature])

        # Make predictions using the loaded model
        predicted_sales = loaded_model['model'].predict(input_data)

        if model_group is not None:
        # Convert the NumPy array to a Python list
            predicted_sales_list = predicted_sales.tolist()

        # Return the predicted sales as a JSON response
            return JSONResponse(content={"Predicted Sales": predicted_sales_list})
        else:
            return {"message": "Invalid store_id"}