import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import joblib

def prepare_churn_features(df, analysis_date=None):
    """
    Prepare features for churn prediction
    
    Args:
        df: Transaction DataFrame
        analysis_date: Date to use as reference, defaults to max date
        
    Returns:
        DataFrame with customer features for churn prediction
    """
    # If no analysis date provided, use the max date in the dataset
    if analysis_date is None:
        analysis_date = df['InvoiceDate'].max()
    
    # Copy data
    df_copy = df.copy()
    
    # Convert InvoiceDate to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df_copy['InvoiceDate']):
        df_copy['InvoiceDate'] = pd.to_datetime(df_copy['InvoiceDate'])
    
    # Calculate days since analysis date for each transaction
    df_copy['DaysSinceAnalysis'] = (analysis_date - df_copy['InvoiceDate']).dt.days
    
    # Calculate average time between purchases
    purchase_dates = df_copy.groupby(['CustomerID', 'InvoiceNo'])['InvoiceDate'].min().reset_index()
    purchase_dates = purchase_dates.sort_values(['CustomerID', 'InvoiceDate'])
    
    # Calculate days between purchases for each customer
    customer_purchase_intervals = []
    
    for customer, group in purchase_dates.groupby('CustomerID'):
        dates = group['InvoiceDate'].tolist()
        if len(dates) > 1:
            intervals = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
            avg_interval = sum(intervals) / len(intervals)
        else:
            avg_interval = None
        
        customer_purchase_intervals.append({
            'CustomerID': customer,
            'AvgPurchaseInterval': avg_interval,
            'LastPurchaseDate': dates[-1],
            'DaysSinceLastPurchase': (analysis_date - dates[-1]).days
        })
    
    purchase_intervals_df = pd.DataFrame(customer_purchase_intervals)
    
    # Define features
    customer_features = df_copy.groupby('CustomerID').agg({
        'InvoiceNo': 'nunique',                # Number of orders
        'TotalAmount': 'sum',                  # Total amount spent
        'DaysSinceAnalysis': 'min',            # Days since most recent purchase
        'ProductID': lambda x: x.nunique(),    # Number of unique products purchased
        'InvoiceDate': lambda x: (x.max() - x.min()).days  # Tenure (days between first and last purchase)
    }).rename(columns={
        'InvoiceNo': 'NumberOfOrders',
        'TotalAmount': 'TotalSpent',
        'DaysSinceAnalysis': 'Recency',
        'ProductID': 'UniqueProducts',
        'InvoiceDate': 'Tenure'
    })
    
    # Calculate average order value
    customer_features['AvgOrderValue'] = customer_features['TotalSpent'] / customer_features['NumberOfOrders']
    
    # Calculate purchase frequency (orders per month)
    customer_features['PurchaseFrequency'] = customer_features['NumberOfOrders'] / (customer_features['Tenure'] / 30 + 1)  # +1 to avoid division by zero
    
    # Merge with purchase intervals
    customer_features = customer_features.merge(purchase_intervals_df[['CustomerID', 'AvgPurchaseInterval', 'DaysSinceLastPurchase']], 
                                               left_index=True, right_on='CustomerID')
    
    # Set CustomerID as index
    customer_features.set_index('CustomerID', inplace=True)
    
    # If average purchase interval is None, replace with median value
    median_interval = purchase_intervals_df['AvgPurchaseInterval'].median()
    customer_features['AvgPurchaseInterval'].fillna(median_interval, inplace=True)
    
    # Define churn (if days since last purchase is more than 2x the avg purchase interval)
    customer_features['ChurnRisk'] = (customer_features['DaysSinceLastPurchase'] > 
                                     2 * customer_features['AvgPurchaseInterval']).astype(int)
    
    # For customers with only 1 purchase, consider them as churn risk if it's been more than 60 days
    single_purchase_mask = customer_features['NumberOfOrders'] == 1
    customer_features.loc[single_purchase_mask, 'ChurnRisk'] = (
        customer_features.loc[single_purchase_mask, 'DaysSinceLastPurchase'] > 60).astype(int)
    
    return customer_features

def train_churn_model(customer_features):
    """
    Train a model to predict customer churn risk
    
    Args:
        customer_features: DataFrame with customer features
        
    Returns:
        Trained model and feature importance
    """
    # Prepare features and target
    features = [
        'NumberOfOrders', 'TotalSpent', 'Recency', 
        'UniqueProducts', 'Tenure', 'AvgOrderValue', 
        'PurchaseFrequency', 'AvgPurchaseInterval', 
        'DaysSinceLastPurchase'
    ]
    
    X = customer_features[features]
    y = customer_features['ChurnRisk']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    print("Churn Model Performance:")
    print(classification_report(y_test, y_pred))
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Feature Importance:")
    print(feature_importance)
    
    # Save model and scaler
    joblib.dump(model, 'churn_model.pkl')
    joblib.dump(scaler, 'churn_scaler.pkl')
    
    return model, scaler, feature_importance

def predict_churn_probability(customer_features, model, scaler):
    """
    Predict churn probability for customers
    
    Args:
        customer_features: DataFrame with customer features
        model: Trained churn prediction model
        scaler: Fitted feature scaler
        
    Returns:
        DataFrame with customers and their churn probabilities
    """
    # Prepare features
    features = [
        'NumberOfOrders', 'TotalSpent', 'Recency', 
        'UniqueProducts', 'Tenure', 'AvgOrderValue', 
        'PurchaseFrequency', 'AvgPurchaseInterval', 
        'DaysSinceLastPurchase'
    ]
    
    X = customer_features[features]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict probabilities
    churn_proba = model.predict_proba(X_scaled)[:, 1]  # Probability of class 1 (churn)
    
    # Create results DataFrame
    churn_predictions = customer_features.copy()
    churn_predictions['ChurnProbability'] = churn_proba
    
    # Sort by churn probability (highest risk first)
    churn_predictions = churn_predictions.sort_values('ChurnProbability', ascending=False)
    
    return churn_predictions

def identify_churn_factors(customer_id, churn_predictions, feature_importance):
    """
    Identify key factors contributing to churn risk for a specific customer
    
    Args:
        customer_id: Customer ID to analyze
        churn_predictions: DataFrame with churn predictions
        feature_importance: DataFrame with feature importance
        
    Returns:
        List of key factors and their values
    """
    if customer_id not in churn_predictions.index:
        return None
    
    # Get customer data
    customer_data = churn_predictions.loc[customer_id]
    
    # Get top features
    top_features = feature_importance['Feature'].tolist()
    
    # Create factors list
    factors = []
    
    for feature in top_features[:3]:  # Top 3 most important features
        value = customer_data[feature]
        
        if feature == 'Recency':
            factor = f"{feature}: {value:.0f} days since last purchase"
        elif feature == 'DaysSinceLastPurchase':
            factor = f"{feature}: {value:.0f} days"
        elif feature == 'TotalSpent' or feature == 'AvgOrderValue':
            factor = f"{feature}: ${value:.2f}"
        elif feature == 'AvgPurchaseInterval':
            factor = f"{feature}: {value:.0f} days between purchases"
        else:
            factor = f"{feature}: {value:.2f}"
            
        factors.append(factor)
    
    return factors