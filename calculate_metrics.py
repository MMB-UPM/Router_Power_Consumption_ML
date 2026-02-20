#!/usr/bin/env python3
"""
Script to calculate MAPE, SMAPE, and relative MSE for RB and RA routers
using polynomial regression models on concatenated test data.
"""

import os
import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.preprocessing import PolynomialFeatures
from tabulate import tabulate


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Avoid division by zero
    mask = denominator != 0
    return np.mean(numerator[mask] / denominator[mask]) * 100


def calculate_relative_mse(y_true, y_pred):
    """Calculate Relative MSE as (MSE / target_mean^2) * 100"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    target_mean = np.mean(y_true)
    return (mse / (target_mean ** 2)) * 100


def load_model_and_metadata(router_type):
    """Load polynomial regression model and metadata for a router"""
    model_type = "polynomial_regression"
    model_version = "1.0.0"
    
    base_path = f"ml_inference/models/{router_type}/{model_type}"
    model_path = f"{base_path}/model_v{model_version}.pkl"
    metadata_path = f"{base_path}/metadata_v{model_version}.yaml"
    
    # Load model
    model = joblib.load(model_path)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
    
    # Get features and polynomial degree
    features = metadata['features']['input']
    target = metadata['features']['output']
    degree = metadata['model']['hyperparameters'].get('degree', 2)
    
    return model, features, target, degree


def load_test_data(router_type):
    """Load and concatenate all test data files for a router"""
    test_data_path = f"telemetry_mock/data/test_data/{router_type}"
    
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(test_data_path) if f.endswith('.csv')]
    csv_files.sort()  # Sort for consistent order
    
    print(f"\n{router_type.upper()} - Loading test data files:")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    
    # Load and concatenate all files
    dfs = []
    for csv_file in csv_files:
        file_path = os.path.join(test_data_path, csv_file)
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Drop any rows with NaN values
    original_len = len(combined_df)
    combined_df = combined_df.dropna()
    dropped = original_len - len(combined_df)
    
    print(f"  Total records: {len(combined_df)}")
    if dropped > 0:
        print(f"  (Dropped {dropped} rows with NaN values)")
    
    return combined_df


def make_predictions(model, data, features, degree):
    """Make predictions using polynomial regression model"""
    # Extract feature columns
    X = data[features].values
    
    # Transform features using PolynomialFeatures
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    # Make predictions
    predictions = model.predict(X_poly)
    
    return predictions


def calculate_metrics_for_router(router_type):
    """Calculate all metrics for a specific router"""
    print(f"\n{'='*60}")
    print(f"Processing {router_type.upper()} Router")
    print(f"{'='*60}")
    
    # Load model and metadata
    model, features, target, degree = load_model_and_metadata(router_type)
    print(f"Model: Polynomial Regression (degree={degree})")
    print(f"Features: {features}")
    print(f"Target: {target}")
    
    # Load test data
    test_data = load_test_data(router_type)
    
    # Get actual values
    y_true = test_data[target].values
    
    # Make predictions
    print(f"\nMaking predictions...")
    y_pred = make_predictions(model, test_data, features, degree)
    
    # Calculate metrics
    mape = calculate_mape(y_true, y_pred)
    smape = calculate_smape(y_true, y_pred)
    relative_mse = calculate_relative_mse(y_true, y_pred)
    
    # Calculate additional statistics for context
    mse = np.mean((y_true - y_pred) ** 2)
    target_mean = np.mean(y_true)
    
    print(f"\nStatistics:")
    print(f"  Mean target value: {target_mean:.2f}")
    print(f"  Standard deviation of target: {np.std(y_true):.2f}")
    print(f"  Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
    print(f"  Symmetric Mean Absolute Percentage Error (SMAPE): {smape:.4f}%")
    print(f"  Relative MSE: {relative_mse:.4f}%")
    
    return {
        'Router': router_type.upper(),
        'MAPE (%)': mape,
        'SMAPE (%)': smape,
        'Relative MSE (%)': relative_mse,
        'Records': len(test_data),
        'Mean Target': target_mean
    }


def main():
    """Main function to calculate metrics for both routers"""
    print("="*60)
    print("ML Model Performance Metrics Calculator")
    print("Model: Polynomial Regression")
    print("="*60)
    
    results = []
    
    # Calculate metrics for RB
    rb_results = calculate_metrics_for_router('rb')
    results.append(rb_results)
    
    # Calculate metrics for RA
    ra_results = calculate_metrics_for_router('ra')
    results.append(ra_results)
    
    # Display results in a formatted table
    print("\n" + "="*60)
    print("SUMMARY - PERFORMANCE METRICS BY ROUTER")
    print("="*60)
    print("\nMetrics Table:")
    
    # Create table for main metrics
    table_data = []
    for result in results:
        table_data.append([
            result['Router'],
            f"{result['MAPE (%)']:.4f}",
            f"{result['SMAPE (%)']:.4f}",
            f"{result['Relative MSE (%)']:.4f}"
        ])
    
    headers = ['Router', 'MAPE (%)', 'SMAPE (%)', 'Relative MSE (%)']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Display additional context
    print("\nAdditional Information:")
    context_data = []
    for result in results:
        context_data.append([
            result['Router'],
            result['Records'],
            f"{result['Mean Target']:.2f}"
        ])
    
    context_headers = ['Router', 'Total Records', 'Mean Energy (W)']
    print(tabulate(context_data, headers=context_headers, tablefmt='grid'))
    
    print("\n" + "="*60)
    print("Calculation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
