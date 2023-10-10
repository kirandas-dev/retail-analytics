# American Retail Sales

## Overview

This project involves the development of two distinct predictive and forecasting models for an American retailer with 10 stores across three different states (California, Texas, and Wisconsin) that sell items from three categories: hobbies, foods, and household. The goal is to provide accurate sales predictions and forecasts to optimize inventory management and business operations.

## Project Organization

The project follows a structured directory organization for ease of management and reproducibility. Here is a brief overview of the project's directory structure:

- `data`: Contains subdirectories for external, interim, processed, and raw data.
- `models`: Stores trained and serialized models, model predictions, or summaries.
- `notebooks`: Jupyter notebooks for exploratory data analysis and model development.
- `reports`: Generated analysis reports in various formats, including figures.
- `src`: Source code organized into subdirectories:
  - `models`: Scripts for model training and predictions.
- `app`: The main folder containing the FastAPI application and related files.
  - `main.py`: Where the FastAPI app is defined, along with its endpoints.
  - `models`: A subdirectory where trained models are stored, organized into subdirectories (e.g., `predictive/` and `forecasting/`).
├── app/
│   ├── main.py
│   ├── models/
│   │   ├── predictive/
│   │   │   ├── model_group_1.joblib
│   │   │   ├── model_group_2.joblib
│   │   │   └── model_group_2.joblib
│   │   ├── forecasting/
│   │   │   ├── arima_pipeline.joblib

## Setup

To set up the project environment, create a virtual environment and install the required dependencies using the `requirements.txt` file:

```shell
pip install -r requirements.txt
