import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller

def load_data(file_path):
    """
    Load CSV data file

    Parameters:
        file_path: path to the data file

    Returns:
        DataFrame: loaded data
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully, shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def clean_data(data, method='interpolate'):
    """
    Clean data by handling missing values and outliers

    Parameters:
        data: original DataFrame
        method: method for handling missing values ('interpolate', 'forward_fill', 'drop')

    Returns:
        DataFrame: cleaned data
    """
    data_clean = data.copy()
    
    print(f"Original data shape: {data_clean.shape}")
    print(f"Missing value statistics:")
    print(data_clean.isnull().sum())
    
    missing_count = data_clean.isnull().sum().sum()
    if missing_count > 0:
        print(f"Total {missing_count} missing values")
        
        if method == 'interpolate':
            numeric_columns = data_clean.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_columns:
                data_clean[col] = data_clean[col].interpolate(method='linear', limit_direction='both')
            print(f"Handled missing values using linear interpolation")
        elif method == 'forward_fill':
            data_clean = data_clean.fillna(method='ffill').fillna(method='bfill')
            print(f"Handled missing values using forward/backward fill")
        elif method == 'drop':
            data_clean = data_clean.dropna()
            print(f"Dropped rows with missing values")
        
        print(f"Shape after cleaning: {data_clean.shape}")
    else:
        print("No missing values in the data")
    
    data_clean = data_clean.reset_index(drop=True)
    
    return data_clean


def get_data_info(data):
    """
    Get basic information about the data

    Parameters:
        data: DataFrame

    Returns:
        None (prints information)
    """
    print("Basic data information:")
    print(f"Data shape: {data.shape}")
    print(f"Data types:")
    print(data.dtypes)
    print(f"Statistical summary:")
    print(data.describe())
    print(f"Column names: {list(data.columns)}")


def split_data(data, train_ratio=0.8, random_state=42):
    """
    Split data into training and test sets

    Parameters:
        data: DataFrame
        train_ratio: proportion for training set (default 0.8)
        random_state: random seed

    Returns:
        tuple: (training set, test set)
    """
    train_size = int(len(data) * train_ratio)
    
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"Training set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")
    
    return train_data, test_data


def check_stationarity(series, name):
    """
    Perform Augmented Dickey-Fuller test for stationarity on a single series.

    Parameters:
        series: pandas Series
        name: name of the series (for printing)
    """
    clean_series = series.dropna()
    result = adfuller(clean_series)
    p_value = result[1]
    verdict = "stationary" if p_value < 0.05 else "non-stationary"
    print(f"{name}: p-value = {p_value:.6f} ({verdict})")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file_path = os.path.join(project_root, "rice_beef_coffee_price_changes.csv")
    
    data = load_data(data_file_path)
    if data is not None:
        data_clean = clean_data(data)
        remaining_nulls = data_clean.isnull().sum().sum()
        
        if remaining_nulls > 0:
            print(f"Warning: {remaining_nulls} missing values remain!")
        else:
            print("No missing values after cleaning.")
            get_data_info(data_clean)
            
            # Define the price column that needs to be tested for stationarity
            price_cols = [
                'Price_beef_kilo', 'Price_rice_kilo', 'Price_coffee_kilo',
                'Price_beef_infl', 'Price_rice_infl', 'Price_coffee_infl'
            ]
            
            print("\n" + "="*50)
            print("Stationarity Test (ADF)")
            print("="*50)
            for col in price_cols:
                if col in data_clean.columns:
                    check_stationarity(data_clean[col], col)
                else:
                    print(f"Column {col} not found, skipping.")
            
            train_data, test_data = split_data(data_clean)
    else:
        print("Data loading failed, cannot proceed.")