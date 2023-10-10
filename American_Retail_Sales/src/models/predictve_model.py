# Import necessary libraries
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import joblib

# Load the preprocessed data from a CSV file
data = pd.read_csv('../data/processed/final_merged_events.csv', low_memory=False)

# List of store names
store_names = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']

# Define the groups
group1 = ['CA_1', 'CA_2', 'CA_3', 'CA_4']
group2 = ['TX_1', 'TX_2', 'TX_3']
group3 = ['WI_1', 'WI_2', 'WI_3']



# Define a custom transformer for feature engineering
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['date'] = pd.to_datetime(X['date'])

        # Extract date components and create new features
        X['day_of_week'] = X['date'].dt.dayofweek
        X['month'] = X['date'].dt.month
        X['year'] = X['date'].dt.year

        # Drop the original 'date' column if needed
        # X = X.drop(columns=['date'])

        return X

# Define the LightGBM model
model = lgb.LGBMRegressor()

# Define the categorical and date feature groups
date_features = ['day_of_week', 'month', 'year']
categorical_features = ['store_id', 'item_id']

# Define a pipeline for target encoding

# Define the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('feature_engineer', FeatureEngineer(), []),
        ('target_encoder', TargetEncoder(),categorical_features)  # Apply target encoding
    ] 
)

# Create the final pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),  # Preprocessing steps
    ('model', model)  # LightGBM model
])

# Initialize lists to store RMSE for each group
rmse_scores = []

# Iterate over each group and filter the data accordingly
for group_idx, group in enumerate([group1, group2, group3]):
    # Filter the data for the current group
    group_data = data[data['store_id'].isin(group)]

    # Define the target variable
    target = 'sales'

    # Split the data into training, validation, and test sets
    train_data, test_data = train_test_split(group_data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # Extract features and target variables for training, validation, and test sets
    X_train, y_train = train_data.drop(columns=[target]), train_data[target]
    X_val, y_val = val_data.drop(columns=[target]), val_data[target]
    X_test, y_test = test_data.drop(columns=[target]), test_data[target]

    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    test_predictions = pipeline.predict(X_test)
    encoder = pipeline.named_steps['preprocessor'].named_transformers_['target_encoder']
    #encoder = pipeline.named_steps['target_encoder']

    print (encoder)
    model_and_encoders = {
        'model': pipeline,
        'encoders': encoder
    }

    models_dir = "../Model_Serving/models/predictive"   # Update the directory name
    os.makedirs(models_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Define the model file path for saving
    model_file_path = os.path.join(models_dir, f'model_group_{group_idx + 1}.joblib')

    # Save the model to the specified file path using joblib
    joblib.dump(model_and_encoders, model_file_path)


    # Calculate RMSE on the test set
    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    print(f'Test RMSE for the group{group_idx + 1}: {rmse}')

    # Append the RMSE score to the list
    rmse_scores.append(rmse)

# Calculate the average RMSE across groups
average_rmse = np.mean(rmse_scores)
print(f'Average RMSE across groups: {average_rmse}')
