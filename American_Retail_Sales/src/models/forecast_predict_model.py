from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import joblib
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

daily_revenue = pd.read_csv('https://raw.githubusercontent.com/kirandas-dev/data-ML/main/combined_time_series.csv', low_memory=False)

# Convert the 'date' column in your DataFrame to Timestamp objects
daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])

class ARIMATransformer(BaseEstimator, TransformerMixin):
    def __init__(self, p, d, q):
        self.p = p
        self.d = d
        self.q = q

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Fit ARIMA model and make predictions
        model = ARIMA(X, order=(self.p, self.d, self.q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=7)  # Forecast next 7 days

        return forecast

# Step 4: Split the data into train and test sets
train_size = len(daily_revenue) - 7  # Leave the last 7 days for testing
train_data = daily_revenue[:train_size]
test_data = daily_revenue[train_size:]

# Create and train the pipeline
pipeline = Pipeline([
    ('arima', ARIMATransformer(p=5, d=0, q=5))  # Adjust p, d, and q as (5, 0, 5)
])

pipeline.fit(train_data['sales'])


import os 
models_dir = "../Model_Serving/models/forecasting"  # Update the directory name
os.makedirs(models_dir, exist_ok=True)  # Create the directory if it doesn't exist
model_filename = os.path.join(models_dir, f'arima_pipeline.joblib')


joblib.dump(pipeline, model_filename)


