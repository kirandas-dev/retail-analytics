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

## Downloading Data
Please download the raw data and kindly save it to the data/raw folder. Data size is heavy and cannot unfortunately be uploaded to Github.:
Training data: (https://drive.google.com/file/d/1-0x5Vfri1i-OL3ek2GnhZGye-4eqbqQA/view?usp=drive_link)

Evaluation data: (https://drive.google.com/file/d/1-2PahhxOmBOCnVNTpgaNGkirjlJz9wYH/view?usp=drive_link)

Calendar: (https://drive.google.com/file/d/1-6cH8c0tKTFu8EzMJyVfdhxrny6rdgrM/view?usp=drive_link)

Events: (https://drive.google.com/file/d/1_RmDGfRTMkqF4OO9NibNoRhbEjc0OZW4/view?usp=drive_link)

Items price per week: (https://drive.google.com/file/d/1--W-RjAnypyvbwUCsSZVldrA2Ja2jtDA/view?usp=drive_link)