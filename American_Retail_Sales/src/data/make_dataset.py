# Import necessary libraries
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np

class DataPreprocessing:
    def __init__(self, sales_train_path, sales_test_path, items_weekly_sell_prices_path, calendar_path, calendar_events_path):
        self.sales_train_path = sales_train_path
        self.sales_test_path = sales_test_path
        self.items_weekly_sell_prices_path = items_weekly_sell_prices_path
        self.calendar_path = calendar_path
        self.calendar_events_path = calendar_events_path

    def read_data(self):
        self.sales_train = pd.read_csv(self.sales_train_path, low_memory=False)
        self.sales_test = pd.read_csv(self.sales_test_path, low_memory=False)
        self.items_weekly_sell_prices = pd.read_csv(self.items_weekly_sell_prices_path, low_memory=False)
        self.calendar = pd.read_csv(self.calendar_path, low_memory=False)
        self.calendar_events = pd.read_csv(self.calendar_events_path, low_memory=False)

    def create_event_counts(self):
        event_counts = self.calendar_events['date'].value_counts().reset_index()
        event_counts.columns = ['date', 'event_count']
        self.calendar_events = pd.merge(self.calendar_events, event_counts, on='date', how='left')
        self.calendar_events = self.calendar_events.drop_duplicates('date', keep='first')

    def map_dates_to_weeks(self):
        date_mapping = dict(zip(self.calendar['d'], self.calendar['date']))
        self.sales_train.rename(columns=date_mapping, inplace=True)

    def melt_sales_data(self):
        self.long_format_sales = pd.melt(self.sales_train, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                                         var_name='date', value_name='units_sold')

    def merge_with_date_to_week_mapping(self):
        date_to_week_mapping = self.calendar[['date', 'wm_yr_wk']]
        self.combined_data = pd.merge(self.long_format_sales, date_to_week_mapping, on='date')

    def merge_with_sell_prices(self):
        self.combined_data = pd.merge(self.combined_data, self.items_weekly_sell_prices, on=['item_id', 'wm_yr_wk', 'store_id'], how='left')
        self.combined_data['sales'] = self.combined_data['sell_price'] * self.combined_data['units_sold']
        self.combined_data = self.combined_data.sort_values(by=['store_id', 'item_id'])
        self.combined_data.reset_index(drop=True, inplace=True)

    def fill_missing_sell_prices(self):
        self.combined_data['sell_price'].fillna(method='ffill', inplace=True)
        self.combined_data['sales'] = self.combined_data['sell_price'] * self.combined_data['units_sold']

    def save_combined_data(self, save_path):
        self.combined_data.to_csv(save_path, index=False)

    def merge_with_calendar_events(self):
        combined_data_with_events = pd.merge(self.combined_data, self.calendar_events, on='date', how='left')
        combined_data_with_events.drop('event_type', axis=1, inplace=True)
        combined_data_with_events['event_count'].fillna(0, inplace=True)
        combined_data_with_events['event_name'].fillna('NoEvent', inplace=True)
        self.combined_data_with_events = combined_data_with_events

    def save_combined_data_with_events(self, save_path):
        self.combined_data_with_events.to_csv(save_path, index=False)

if __name__ == '__main__':
    # Specify file paths
    sales_train_path = '../data/raw/sales_train.csv'
    sales_test_path = '../data/raw/sales_test.csv'
    items_weekly_sell_prices_path = '../data/raw/items_weekly_sell_prices.csv'
    calendar_path = '../data/raw/calendar.csv'
    calendar_events_path = '../data/raw/calendar_events.csv'

    # Create an instance of the DataPreprocessing class
    data_processor = DataPreprocessing(sales_train_path, sales_test_path, items_weekly_sell_prices_path, calendar_path, calendar_events_path)

    # Perform data preprocessing steps
    data_processor.read_data()
    data_processor.create_event_counts()
    data_processor.map_dates_to_weeks()
    data_processor.melt_sales_data()
    data_processor.merge_with_date_to_week_mapping()
    data_processor.merge_with_sell_prices()
    data_processor.fill_missing_sell_prices()

    # Save the processed data
    save_path = '../data/processed/final_merged.csv'
    data_processor.save_combined_data(save_path)

    # Merge with calendar events and save the final data
    data_processor.merge_with_calendar_events()
    save_path_with_events = '../data/processed/final_merged_events.csv'
    data_processor.save_combined_data_with_events(save_path_with_events)
