import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import os

# Configuration
DATA_DIR = r"C:\Users\d501434\OneDrive - Drew Marine\Desktop\Pytrials\data"
OUTPUT_DIR = r"C:\Users\d501434\OneDrive - Drew Marine\Desktop\Pytrials\model\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load transactional data
def load_transactions():
    try:
        print("Processing: Loading transactions...")
        transactions_path = os.path.join(DATA_DIR, "transactions.csv")
        transactions = pd.read_csv(transactions_path, low_memory=False)

        # Rename columns to match corrected names
        transactions.rename(columns={
            'Tracking #': 'Tracking_#',
            'Due Date': 'Due_Date',
            'w/h number': 'wh_number',
            'Sales (USD)': 'Sales_(USD)',
            'COS (USD)': 'COS_(USD)',
            'Product Description': 'Product_Description',
            'Business Unit': 'Business_Unit',  # Renamed to match the dataset structure
            'Business Segment': 'Business_Segment'
        }, inplace=True)

        # Convert Due_Date to datetime and extract YearMonth
        transactions['Due_Date'] = pd.to_datetime(transactions['Due_Date'], errors='coerce')
        transactions['YearMonth'] = transactions['Due_Date'].dt.to_period('M')

        # Drop lead_time if present
        transactions = transactions.drop(columns=['lead_time'], errors='ignore')

        print("Completed: Transactions loaded successfully.")
        return transactions
    except Exception as e:
        print(f"Error loading transactions: {e}")
        return None

# Load item master data
def load_item_master():
    try:
        print("Processing: Loading item master...")
        item_master_path = os.path.join(DATA_DIR, "item_master.csv")
        item_master = pd.read_csv(item_master_path)

        # Rename columns to match corrected names
        item_master.rename(columns={
            'Product description': 'Product_Description',
            'Lead_time': 'lead_time',  # Retain for now but drop later if not needed
            'Business Unit': 'Business_Unit',  # Renamed to match the dataset structure
            'Business Segment': 'Business_Segment'
        }, inplace=True)

        # Drop lead_time if present
        item_master = item_master.drop(columns=['lead_time'], errors='ignore')

        print("Completed: Item master loaded successfully.")
        return item_master
    except Exception as e:
        print(f"Error loading item master: {e}")
        return None

# Load warehouses data
def load_warehouses():
    try:
        print("Processing: Loading warehouses...")
        warehouses_path = os.path.join(DATA_DIR, "warehouses.csv")
        warehouses = pd.read_csv(warehouses_path)

        # Drop lead_time if present
        warehouses = warehouses.drop(columns=['lead_time'], errors='ignore')

        print("Completed: Warehouses loaded successfully.")
        return warehouses
    except Exception as e:
        print(f"Error loading warehouses: {e}")
        return None

# Preprocess historical data
def preprocess_historical_data(transactions, item_master, warehouses):
    try:
        print("Processing: Preprocessing historical data...")

        # Debugging: Display transactions and item_master structure before merge
        print("Transactions Data (Before Merge):")
        print(transactions[['Product', 'wh_number']].head())
        print("Item Master Data (Before Merge):")
        print(item_master[['Product', 'wh_number', 'Product_Description']].head())

        # Drop Product_Description from transactions to avoid duplication
        transactions = transactions.drop(columns=['Product_Description'], errors='ignore')

        # Ensure merge keys are consistent
        transactions['Product'] = transactions['Product'].astype(str).str.strip()
        transactions['wh_number'] = transactions['wh_number'].astype(str).str.strip()

        item_master['Product'] = item_master['Product'].astype(str).str.strip()
        item_master['wh_number'] = item_master['wh_number'].astype(str).str.strip()

        warehouses['wh_number'] = warehouses['wh_number'].astype(str).str.strip()

        # Merge transactions with item_master
        data = transactions.merge(item_master, on=['Product', 'wh_number'], how='left')

        # Debugging: Check if Product_Description exists after merge
        if 'Product_Description' not in data.columns or data['Product_Description'].isnull().all():
            raise KeyError("Product_Description is missing after merging transactions with item_master.")

        # Merge with warehouses
        data = data.merge(warehouses, on='wh_number', how='left')

        # Drop irrelevant columns
        data = data.drop(columns=['Tracking_#', 'Customer_Name', 'Vessel_Name'], errors='ignore')

        # Clean numeric columns
        numeric_cols = ['Quantity', 'Sales_(USD)', 'COS_(USD)', 'unit_price']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

        # Ensure text columns remain as strings
        text_cols = ['Product_Description', 'wh_name', 'City', 'Business_Unit', 'Business_Segment']
        for col in text_cols:
            if col in data.columns:
                data[col] = data[col].astype(str)

        print("Completed: Historical data preprocessed successfully.")
        return data
    except Exception as e:
        print(f"Error preprocessing historical data: {e}")
        return None

# Aggregate monthly data
def aggregate_data(data):
    try:
        print("Processing: Aggregating data...")

        # Define required columns for aggregation
        required_columns = ['YearMonth', 'Product', 'Product_Description', 'wh_number', 'wh_name', 'Quantity', 'Sales_(USD)', 'COS_(USD)', 'unit_price', 'Business_Unit', 'Business_Segment']
        if not all(col in data.columns for col in required_columns):
            raise KeyError(f"Missing required columns for aggregation: {', '.join(set(required_columns) - set(data.columns))}")

        # Aggregate measures by YearMonth, Product, and wh_name
        monthly_data = data.groupby(['YearMonth', 'Product', 'Product_Description', 'wh_number', 'wh_name', 'Business_Unit', 'Business_Segment']).agg({
            'Quantity': 'sum',
            'Sales_(USD)': 'sum',
            'COS_(USD)': 'sum',
            'unit_price': 'mean'
        }).reset_index()

        print("Completed: Data aggregated successfully.")
        return monthly_data
    except Exception as e:
        print(f"Error aggregating data: {e}")
        return None
    
from sklearn.preprocessing import LabelEncoder

# Label encode and feature engineering
def label_encode_and_feature_engineering(monthly_data):
    try:
        print("Processing: Label encoding and feature engineering...")

        # Define required columns for feature engineering
        required_columns = ['Product', 'Product_Description', 'wh_number', 'wh_name', 'Quantity', 'unit_price', 'Business_Unit', 'Business_Segment']
        if not all(col in monthly_data.columns for col in required_columns):
            raise KeyError(f"Missing required columns for feature engineering: {', '.join(set(required_columns) - set(monthly_data.columns))}")

        # Label encode categorical columns
        product_encoder = LabelEncoder()
        warehouse_encoder = LabelEncoder()
        monthly_data['Product_Encoded'] = product_encoder.fit_transform(monthly_data['Product'])
        monthly_data['Warehouse_Encoded'] = warehouse_encoder.fit_transform(monthly_data['wh_name'])

        # Feature engineering
        monthly_data['Month'] = monthly_data['YearMonth'].dt.month
        monthly_data['Quantity_Lag1'] = monthly_data.groupby(['Product_Encoded', 'Warehouse_Encoded'])['Quantity'].shift(1)
        monthly_data['Quantity_Lag2'] = monthly_data.groupby(['Product_Encoded', 'Warehouse_Encoded'])['Quantity'].shift(2)
        monthly_data['Quantity_RollingMean_3'] = monthly_data.groupby(['Product_Encoded', 'Warehouse_Encoded'])['Quantity'].transform(lambda x: x.rolling(window=3).mean())

        # Identify records with insufficient historical data
        insufficient_data = monthly_data[monthly_data[['Quantity_Lag1', 'Quantity_Lag2']].isnull().any(axis=1)].copy()

        # Calculate additional metrics for insufficient data
        insufficient_data['Tracking_Count_Past_24_Months'] = insufficient_data.apply(
            lambda row: monthly_data[
                (monthly_data['Product'] == row['Product']) &
                (monthly_data['wh_name'] == row['wh_name']) &
                (monthly_data['YearMonth'] >= monthly_data['YearMonth'].max() - 24)
            ]['Product'].nunique(), axis=1
        )
        insufficient_data['Total_Quantity_Past_24_Months'] = insufficient_data.apply(
            lambda row: monthly_data[
                (monthly_data['Product'] == row['Product']) &
                (monthly_data['wh_name'] == row['wh_name']) &
                (monthly_data['YearMonth'] >= monthly_data['YearMonth'].max() - 24)
            ]['Quantity'].sum(), axis=1
        )
        insufficient_data['No_History_Periods'] = insufficient_data.apply(
            lambda row: monthly_data[
                (monthly_data['Product'] == row['Product']) &
                (monthly_data['wh_name'] == row['wh_name']) &
                (monthly_data['YearMonth'] >= monthly_data['YearMonth'].max() - 24)
            ]['Quantity'].eq(0).sum(), axis=1
        )
        insufficient_data['Total_Sales_Past_24_Months'] = insufficient_data.apply(
            lambda row: monthly_data[
                (monthly_data['Product'] == row['Product']) &
                (monthly_data['wh_name'] == row['wh_name']) &
                (monthly_data['YearMonth'] >= monthly_data['YearMonth'].max() - 24)
            ]['Sales_(USD)'].sum(), axis=1
        )
        insufficient_data['Total_COS_Past_24_Months'] = insufficient_data.apply(
            lambda row: monthly_data[
                (monthly_data['Product'] == row['Product']) &
                (monthly_data['wh_name'] == row['wh_name']) &
                (monthly_data['YearMonth'] >= monthly_data['YearMonth'].max() - 24)
            ]['COS_(USD)'].sum(), axis=1
        )

        # Save insufficient data to a separate file
        insufficient_data.to_csv(os.path.join(OUTPUT_DIR, "insufficient_data.csv"), index=False)

        # Drop rows with insufficient historical data
        monthly_data = monthly_data.dropna(subset=['Quantity_Lag1', 'Quantity_Lag2'])

        monthly_data = monthly_data.fillna(0)
        print("Completed: Label encoding and feature engineering completed.")
        return monthly_data, product_encoder, warehouse_encoder
    except Exception as e:
        print(f"Error during label encoding and feature engineering: {e}")
        return None, None, None
    
# Train model
def train_model(train_X, train_y):
    try:
        print("Processing: Training model...")
        from sklearn.ensemble import RandomForestRegressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(train_X, train_y)
        print("Completed: Model trained successfully.")
        return rf_model
    except Exception as e:
        print(f"Error during model training: {e}")
        return None

# Generate future predictions
def generate_future_predictions(model, last_historical_data, product_warehouse_pairs, months_to_forecast=12):
    try:
        print("Processing: Generating future forecasts...")

        # Create future date range
        last_historical_date = last_historical_data['YearMonth'].max()
        future_months = pd.period_range(start=last_historical_date + 1, periods=months_to_forecast, freq='M')
        future_df = pd.DataFrame({'YearMonth': future_months})
        future_df['Month'] = future_df['YearMonth'].dt.month

        # Add Product/Warehouse info
        future_df = future_df.merge(product_warehouse_pairs, how='cross')

        # Merge with last historical data
        last_historical_data = last_historical_data[last_historical_data['YearMonth'] == last_historical_date].drop('YearMonth', axis=1)
        future_df = future_df.merge(
            last_historical_data,
            on=['Product_Encoded', 'Warehouse_Encoded'],
            how='left',
            suffixes=('', '_last_hist')
        )

        # Populate lag features
        future_df['Quantity_Lag1'] = future_df['Quantity']
        future_df['Quantity_Lag2'] = future_df['Quantity_Lag1']
        future_df['Quantity_RollingMean_3'] = future_df[['Quantity', 'Quantity_Lag1', 'Quantity_Lag2']].mean(axis=1)

        # Fill NaN with 0 (avoiding chained assignment)
        future_df = future_df.fillna(0)

        # Ensure columns match training data
        future_features = future_df[model.feature_names_in_]

        # Predict
        future_predictions = model.predict(future_features)
        future_df['Predicted_Quantity'] = future_predictions
        future_df['Predicted_Amount'] = future_df['Predicted_Quantity'] * future_df['unit_price']

        # Simplify forecast results
        future_df = future_df[['YearMonth', 'Product', 'Product_Description', 'wh_number', 'wh_name', 'Predicted_Quantity', 'Predicted_Amount']]
        print("Forecast Results (Before Saving):")
        print(future_df.head())

        # Save forecast results
        future_df.to_csv(os.path.join(OUTPUT_DIR, "forecast_results.csv"), index=False)
        print("Completed: Future forecasts saved successfully.")
        return future_df
    except Exception as e:
        print(f"Error generating forecasts: {e}")
        return None

# Main function for forecasting
def main():
    print("Starting forecasting process...")

    # Load data
    transactions = load_transactions()
    if transactions is None:
        print("Error occurred during data loading. Exiting.")
        return

    item_master = load_item_master()
    if item_master is None:
        print("Error occurred during data loading. Exiting.")
        return

    warehouses = load_warehouses()
    if warehouses is None:
        print("Error occurred during data loading. Exiting.")
        return

    # Preprocess historical data
    data = preprocess_historical_data(transactions, item_master, warehouses)
    if data is None:
        print("Error occurred during preprocessing. Exiting.")
        return

    # Aggregate data
    monthly_data = aggregate_data(data)
    if monthly_data is None:
        print("Error occurred during aggregation. Exiting.")
        return

    # Save aggregated data
    monthly_data.to_csv(os.path.join(OUTPUT_DIR, "aggregated_data.csv"), index=False)
    print("Completed: Aggregated data saved successfully.")

    # Label encode and feature engineering
    feature_engineered_data, product_encoder, warehouse_encoder = label_encode_and_feature_engineering(monthly_data)
    if feature_engineered_data is None:
        print("Error occurred during feature engineering. Exiting.")
        return

    feature_engineered_data.to_csv(os.path.join(OUTPUT_DIR, "feature_engineered_data.csv"), index=False)
    print("Completed: Feature-engineered data saved successfully.")

    # Prepare training data
    train_data = feature_engineered_data[feature_engineered_data['Quantity'] > 0]
    features = ['Month', 'Quantity_Lag1', 'Quantity_Lag2', 'Quantity_RollingMean_3']
    target = 'Quantity'

    if not all(col in train_data.columns for col in features):
        print(f"Required features ({', '.join(features)}) are missing in the feature-engineered data. Exiting.")
        return

    train_X = train_data[features]
    train_y = train_data[target]

    # Train model
    model = train_model(train_X, train_y)
    if model is None:
        print("Error occurred during model training. Exiting.")
        return

    # Generate future predictions
    product_warehouse_pairs = feature_engineered_data[['Product_Encoded', 'Warehouse_Encoded']].drop_duplicates()
    future_data = generate_future_predictions(model, feature_engineered_data, product_warehouse_pairs)

    print("Forecasting process completed successfully.")

if __name__ == "__main__":
    main()
