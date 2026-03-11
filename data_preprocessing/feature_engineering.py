import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from data_preprocessing.load_data import load_data, clean_data, split_data 


def create_temporal_features(data):
    """
    Create time-related features

    Parameters:
        data: DataFrame

    Returns:
        DataFrame: data with added time features
    """
    data_features = data.copy()
    
    if 'Year' in data.columns and 'Month' in data.columns:
        month_mapping = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        
        data_features['Month_num'] = data_features['Month'].map(month_mapping)
        
        data_features['Quarter'] = (data_features['Month_num'] - 1) // 3 + 1
        
        data_features['Is_Q1'] = (data_features['Quarter'] == 1).astype(int)
        data_features['Is_Q2'] = (data_features['Quarter'] == 2).astype(int)
        data_features['Is_Q3'] = (data_features['Quarter'] == 3).astype(int)
        data_features['Is_Q4'] = (data_features['Quarter'] == 4).astype(int)
        
        data_features['Sin_month'] = np.sin(2 * np.pi * data_features['Month_num'] / 12)
        data_features['Cos_month'] = np.cos(2 * np.pi * data_features['Month_num'] / 12)
        
        print("Temporal features created")
    
    return data_features


def create_lag_features(data, columns, lags=[1, 3, 6, 12]):
    """
    Create lag features

    Parameters:
        data: DataFrame
        columns: list of column names to create lag features for
        lags: list of lag steps

    Returns:
        DataFrame: data with added lag features
    """
    data_features = data.copy()
    
    for column in columns:
        if column in data_features.columns:
            for lag in lags:
                data_features[f"{column}_lag_{lag}"] = data_features[column].shift(lag)
    
    print(f"Lag features created, lag steps: {lags}")
    
    return data_features


def create_rolling_features(data, columns, windows=[3, 6, 12]):
    """
    Create rolling window features

    Parameters:
        data: DataFrame
        columns: list of column names to create rolling features for
        windows: list of window sizes

    Returns:
        DataFrame: data with added rolling features
    """
    data_features = data.copy()
    
    for column in columns:
        if column in data_features.columns:
            for window in windows:
                data_features[f"{column}_rolling_mean_{window}"] = data_features[column].rolling(window=window).mean()
                data_features[f"{column}_rolling_std_{window}"] = data_features[column].rolling(window=window).std()
    
    print(f"Rolling window features created, window sizes: {windows}")
    
    return data_features


def create_price_ratio_features(data):
    """
    Create price ratio features

    Parameters:
        data: DataFrame

    Returns:
        DataFrame: data with added price ratio features
    """
    data_features = data.copy()
    
    if 'Price_beef_kilo' in data_features.columns and 'Price_rice_kilo' in data_features.columns:
        data_features['Beef_Rice_Ratio'] = data_features['Price_beef_kilo'] / (data_features['Price_rice_kilo'] + 1e-8)
    
    if 'Price_beef_kilo' in data_features.columns and 'Price_coffee_kilo' in data_features.columns:
        data_features['Beef_Coffee_Ratio'] = data_features['Price_beef_kilo'] / (data_features['Price_coffee_kilo'] + 1e-8)
    
    if 'Price_rice_kilo' in data_features.columns and 'Price_coffee_kilo' in data_features.columns:
        data_features['Rice_Coffee_Ratio'] = data_features['Price_rice_kilo'] / (data_features['Price_coffee_kilo'] + 1e-8)
    
    print("Price ratio features created")
    
    return data_features


def normalize_features(data, columns, method='standard'):
    """
    Normalize or standardize features

    Parameters:
        data: DataFrame
        columns: list of column names to normalize
        method: method ('standard' or 'minmax')

    Returns:
        tuple: (normalized data, scaler object)
    """
    data_normalized = data.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        print("Unknown normalization method")
        return data_normalized, None
    
    data_normalized[columns] = scaler.fit_transform(data_normalized[columns])
    
    print(f"Features normalized using {method} method")
    
    return data_normalized, scaler


def prepare_features_for_modeling(data, target_columns=None, drop_columns=None):
    """
    Prepare features for modeling

    Parameters:
        data: DataFrame
        target_columns: list of target column names
        drop_columns: list of column names to drop

    Returns:
        tuple: (features, targets)
    """
    data_prepared = data.copy()
    
    default_drop_columns = ['Year', 'Month', 'Price_rice_infl', 'Price_beef_infl', 'Price_coffee_infl']
    
    if drop_columns is None:
        drop_columns = default_drop_columns
    else:
        drop_columns = list(set(drop_columns + default_drop_columns))
    
    data_prepared = data_prepared.drop(columns=drop_columns, errors='ignore')
    
    if target_columns is None:
        target_columns = ['Price_beef_kilo', 'Price_rice_kilo', 'Price_coffee_kilo']
    
    X = data_prepared.drop(columns=target_columns, errors='ignore')
    y = data_prepared[target_columns]
    
    X = X.select_dtypes(include=[np.number])
    
    print(f"Feature shape: {X.shape}")
    print(f"Number of feature columns: {X.shape[1]}")
    print(f"Target shape: {y.shape}")
    
    return X, y


def remove_nan_rows(data):
    """
    Remove rows containing NaN values

    Parameters:
        data: DataFrame

    Returns:
        DataFrame: data after removing NaN rows
    """
    data_clean = data.dropna()
    print(f"Data shape after removing NaN: {data_clean.shape}")
    return data_clean


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file_path = os.path.join(project_root, "rice_beef_coffee_price_changes.csv")
    
    data = load_data(data_file_path)
    if data is not None:
        data_clean = clean_data(data, method='interpolate')
        
        data_features = create_temporal_features(data_clean)
        
        price_columns = ['Price_beef_kilo', 'Price_rice_kilo', 'Price_coffee_kilo']
        data_features = create_lag_features(data_features, price_columns, lags=[1, 3, 6])
        data_features = create_rolling_features(data_features, price_columns, windows=[3, 6])
        data_features = create_price_ratio_features(data_features)
        
        data_features = remove_nan_rows(data_features)
        
        X, y = prepare_features_for_modeling(data_features)
        
        print("Feature engineering completed")