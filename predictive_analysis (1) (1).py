"""
Predictive Analytics and Trend Forecasting Tool
Author: Data Analytics Team
Description: Preprocesses historical sequence metrics and fits a trend forecasting model.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Suppress background execution warnings
warnings.filterwarnings("ignore")

def generate_historical_trends():
    """Generates synthetic historical business metrics tracking a growth trend."""
    print("[INFO] Simulating historical time-series baseline parameters...")
    np.random.seed(42)
    
    # Simulating 36 months of sales data
    months = np.arange(1, 37)
    dates = pd.date_range(start="2023-01-01", periods=36, freq="ME")
    
    # Upward linear trend + seasonal variation + random noise
    base_trend = 150 + (months * 4.5)
    seasonal_effect = np.sin(months * (2 * np.pi / 12)) * 15
    random_noise = np.random.normal(0, 8, size=36)
    
    sales_volume = base_trend + seasonal_effect + random_noise
    
    return pd.DataFrame({
        'Month_Index': months,
        'Date': dates,
        'Sales_Volume': np.round(sales_volume, 2)
    })

def build_predictive_model(df):
    """Preprocesses variables and fits a regression pipeline to project trends."""
    print("[INFO] Preprocessing data splits and training regression matrix...")
    
    # Reshaping features for Scikit-Learn compliance
    X = df[['Month_Index']].values
    y = df['Sales_Volume'].values
    
    # Fit the predictive model
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate historical predictions (Fitted Values)
    df['Predicted_Sales'] = model.predict(X)
    
    # Evaluate model accuracy metrics
    mse = mean_squared_error(y, df['Predicted_Sales'])
    rmse = np.sqrt(mse)
    r2 = r2_score(y, df['Predicted_Sales'])
    
    # Forecast future trends (Next 12 months)
    print("[INFO] Projecting future metrics across a 12-month forward horizon...")
    future_months = np.arange(37, 49).reshape(-1, 1)
    future_dates = pd.date_range(start="2026-01-01", periods=12, freq="ME")
    future_predictions = model.predict(future_months)
    
    future_df = pd.DataFrame({
        'Month_Index': future_months.flatten(),
        'Date': future_dates,
        'Sales_Volume': np.nan,  # Future actuals are unknown
        'Predicted_Sales': np.round(future_predictions, 2)
    })
    
    return df, future_df, rmse, r2

def visualize_forecast_trends(historical_df, forecast_df):
    """Plots historical factual distributions alongside predictive forward horizons."""
    print("[INFO] Generating high-fidelity trend visualization matrix...")
    plt.figure(figsize=(12, 6.5))
    sns.set_theme(style="whitegrid")
    
    # Plot Historical Actual Data points
    plt.plot(historical_df['Date'], historical_df['Sales_Volume'], 
             label='Historical Actuals', color='#1f77b4', marker='o', linewidth=2)
    
    # Plot Model Linear Trend Line
    plt.plot(historical_df['Date'], historical_df['Predicted_Sales'], 
             label='Model Fit (Regression Line)', color='#ff7f0e', linestyle='--')
    
    # Plot 12-Month Future Forecast
    plt.plot(forecast_df['Date'], forecast_df['Predicted_Sales'], 
             label='12-Month Forward Forecast', color='#2ca02c', linestyle=':', marker='s', linewidth=2)
    
    # Formatting presentation space
    plt.title('Business Performance Forecasting & Trend Line Analytics', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Timeline Horizon', fontsize=11)
    plt.ylabel('Sales Volume (Units)', fontsize=11)
    plt.legend(loc='upper left', frameon=True)
    
    plt.tight_layout()
    output_filename = 'predictive_trend_forecast.png'
    plt.savefig(output_filename, dpi=300)
    print(f"[SUCCESS] Analytical chart successfully exported as '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    print("=== STARTING PREDICTIVE ANALYTICAL CORE ENGINE ===")
    
    # 1. Gather historical baseline metric frames
    historical_data = generate_historical_trends()
    
    # 2. Process features and map forecasting loops
    historical_mapped, future_forecast, rmse_score, r2_score_val = build_predictive_model(historical_data)
    
    print("\n=== MODEL PERFORMANCE ACCURACY METRICS ===")
    print(f"Root Mean Squared Error (RMSE): {rmse_score.round(4)}")
    print(f"R-Squared Score (Variance Explanations): {round(r2_score_val, 4)} ({round(r2_score_val * 100, 2)}%)")    
    print("==========================================\n")
    
    print("=== FUTURE FORECAST TARGETS (NEXT 12 MONTHS) ===")
    print(future_forecast[['Date', 'Predicted_Sales']].to_string(index=False))
    print("================================================\n")
    
    # 3. Output tracking trend visualization curves
    visualize_forecast_trends(historical_mapped, future_forecast)
    print("=== PIPELINE RUN COMPLETE AND READY FOR SUBMISSION ===")