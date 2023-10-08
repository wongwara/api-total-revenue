from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
from datetime import datetime, timedelta

app = FastAPI()

project_description = {
    "description": "This project aims to forecast sales volume for stores and items.",
    "endpoints": {
        "/": "Display a brief description of the project objectives, list of endpoints, expected input parameters, and output format of the model.",
        "/health/": "Return status code 200 with a welcome message.",
        "/sales/national/": "Return the next 7 days total revenue forecast for an input date.",
        "/sales/stores/items/": "Return predicted total revenue for an input item, store, and date."
    },
    "github_repo": "https://github.com/wongwara/api-total-revenue"
}

@app.get("/")
async def read_root():
    return project_description

def load_total_revenue(filepath):
    # Load sales data from a CSV file
    total_revenue = pd.read_csv(filepath)
    total_revenue['date'] = pd.to_datetime(total_revenue['date'])
    return total_revenue

@app.get('/health', status_code=200)
async def healthcheck():
    return 'The prediction and forecasting are all ready to go! :0'

def format_features(store_id: str, item_id: str, date: str, revenue: float):
    """
    Format the features, including date transformations.

    Args:
        store_id (str): Store ID.
        item_id (str): Item ID.
        date (str): Date in a string format (e.g., 'yyyy-mm-dd').
        revenue (float): Revenue value.

    Returns:
        pd.Series: A Pandas Series with formatted features.
    """
    # Convert the date to a datetime object
    date = pd.to_datetime(date)
    # Extract date features
    month = date.month
    weekday = date.weekday()
    year = date.year
    quarter = date.quarter
    week = date.isocalendar().week
    dayofyear = date.dayofyear

    # Create a Pandas Series with the formatted features
    formatted_features = pd.Series({
        'store_id': store_id,
        'item_id': item_id,
        'date': date,
        'revenue': revenue,
        'month': month,
        'weekday': weekday,
        'year': year,
        'quarter': quarter,
        'week': week,
        'dayofyear': dayofyear
    })

    return formatted_features

def add_date_features(df, date_column='date'):
    """
    Adds date-related features to a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the date column.
        date_column (str): Name of the date column. Default is 'date'.

    Returns:
        pd.DataFrame: DataFrame with added date-related features.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    df['month'] = df[date_column].dt.month
    df['weekday'] = df[date_column].dt.weekday
    df['year'] = df[date_column].dt.year
    df['quarter'] = df[date_column].dt.quarter
    df['week'] = df[date_column].dt.isocalendar().week
    df['dayofyear'] = df[date_column].dt.dayofyear

    return df

def predict(store_id: str, item_id: str, revenue: float, date: str):
    # Create a DataFrame with the formatted features
    df = pd.DataFrame({
        'store_id': [store_id],
        'item_id': [item_id],
        'revenue': [revenue],
        'date': [date]  # Date should be in 'yyyy-mm-dd' format
    })

    # Add date-related features to the DataFrame
    df = add_date_features(df)

    # Load the pre-fitted XGBoost model with pipeline
    xgb_pipeline_nolags = joblib.load("../models/xgb_pipeline_nolags.pkl")

    # Make predictions using the loaded model
    prediction = xgb_pipeline_nolags.predict(df)[0]

    return prediction

@app.get('/sales/national', response_model=dict, tags=["Sales"])
async def get_sales_forecast(input_date: str = Query(..., description="Input date in 'yyyy-mm-dd' format")):
    try:
        # Convert the input date to a datetime object
        input_date = datetime.strptime(input_date, '%Y-%m-%d')
        forecast_data = {}

        # Calculate the sales volume forecast for the next 7 days
        for day in range(1, 8):
            forecast_date = input_date + timedelta(days=day)
            formatted_features = format_features('national', 'all', forecast_date, 0.0)  # You can use 0.0 for revenue since it's a forecast

            # This ensures that the model is loaded only once and reused for predictions
            if 'xgb_pipeline_nolags' not in get_sales_forecast.__dict__:
                get_sales_forecast.xgb_pipeline_nolags = joblib.load("../models/xgb_pipeline_nolags.pkl")

            # Make predictions using the loaded model
            prediction = get_sales_forecast.xgb_pipeline_nolags.predict(formatted_features.to_frame().T)[0]

            forecast_data[forecast_date.strftime('%Y-%m-%d')] = prediction

        return {'message': 'Sales Volume Forecast for Next 7 Days', 'data': forecast_data}
    except ValueError:
        return JSONResponse(content={'message': 'Invalid date format. Use YYYY-MM-DD.'}, status_code=400)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
