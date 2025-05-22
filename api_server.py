from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import uvicorn
import json
from datetime import datetime
import yaml
import os

from src.data.preprocess import preprocess_data, generate_sample_data
from src.models.customer_segmentation import perform_rfm_analysis, perform_customer_clustering
from src.models.recommendation import hybrid_recommendation, collaborative_filtering, content_based_recommendation
from src.models.churn_prediction import prepare_churn_features, train_churn_model, predict_churn_probability

# Load configuration
def load_config():
    """Load configuration from config.yaml or use defaults"""
    default_config = {
        'data_generation': {
            'n_customers': 1000,
            'n_products': 200,
            'n_transactions': 10000,
            'start_date': '2024-01-01',
            'end_date': '2025-05-01',
            'product_categories': ['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports', 'Books', 'Toys', 'Food', 'Health'],
            'min_price': 5,
            'max_price': 200,
            'min_quantity': 1,
            'max_quantity': 5,
            'min_items_per_invoice': 1,
            'max_items_per_invoice': 5,
            'price_variation_range': [0.95, 1.05]
        },
        'preprocessing': {
            'required_columns': ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'Quantity', 'UnitPrice'],
            'min_quantity': 0,
            'min_price': 0,
            'type_mapping': {
                'CustomerID': 'str',
                'InvoiceNo': 'str',
                'Quantity': 'float',
                'UnitPrice': 'float',
                'TotalAmount': 'float'
            }
        },
        'churn_analysis': {
            'high_risk_threshold': 0.7,
            'top_risk_customers': 10
        }
    }
    
    try:
        if os.path.exists('config.yaml'):
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
                # Merge with defaults for any missing values
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
    except Exception as e:
        print(f"Error loading config: {str(e)}")
    
    return default_config

# Initialize FastAPI app
app = FastAPI(
    title="E-Commerce Analytics API",
    description="API for e-commerce data analysis and customer insights",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
config = load_config()

# Global variables to store data
data_store = {
    'data': None,
    'rfm_data': None,
    'cluster_data': None,
    'cluster_analysis': None,
    'cluster_names': None,
    'churn_features': None,
    'churn_model': None,
    'churn_scaler': None,
    'feature_importance': None,
    'churn_predictions': None
}

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    """
    Upload and process transaction data
    """
    try:
        # Read the uploaded file
        df = pd.read_csv(file.file)
        
        # Validate required columns
        required_columns = config['preprocessing']['required_columns']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {', '.join(missing_columns)}")
        
        # Preprocess the data
        df = preprocess_data(df, config['preprocessing'])
        
        # Store the processed data
        data_store['data'] = df
        
        return {"message": "Data uploaded and processed successfully", "rows": len(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-sample-data")
async def generate_sample():
    """
    Generate sample transaction data
    """
    try:
        df = generate_sample_data(**config['data_generation'])
        data_store['data'] = df
        return {"message": "Sample data generated successfully", "rows": len(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/customer-segmentation")
async def get_customer_segmentation():
    """
    Get customer segmentation analysis
    """
    try:
        if data_store['data'] is None:
            raise HTTPException(status_code=400, detail="No data available. Please upload or generate data first.")
        
        # Perform RFM analysis
        rfm_data = perform_rfm_analysis(data_store['data'])
        cluster_data, cluster_analysis, cluster_names = perform_customer_clustering(rfm_data)
        
        # Store results
        data_store['rfm_data'] = rfm_data
        data_store['cluster_data'] = cluster_data
        data_store['cluster_analysis'] = cluster_analysis
        data_store['cluster_names'] = cluster_names
        
        # Prepare response
        response = {
            "rfm_segments": rfm_data['RFM_Segment'].value_counts().to_dict(),
            "cluster_distribution": cluster_data['Cluster_Name'].value_counts().to_dict(),
            "cluster_analysis": cluster_analysis.to_dict(),
            "cluster_names": cluster_names
        }
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommendations/{customer_id}")
async def get_recommendations(customer_id: str, n_recommendations: int = 5):
    """
    Get product recommendations for a customer
    """
    try:
        if data_store['data'] is None:
            raise HTTPException(status_code=400, detail="No data available. Please upload or generate data first.")
        
        # Get recommendations using hybrid approach
        recommendations = hybrid_recommendation(data_store['data'], customer_id, n_recommendations)
        
        # Get product details
        product_details = []
        for product_id in recommendations:
            product_data = data_store['data'][data_store['data']['ProductID'] == product_id].iloc[0]
            product_details.append({
                'ProductID': product_id,
                'Description': product_data['Description'],
                'Category': product_data['Category'],
                'Price': float(product_data['UnitPrice'])
            })
        
        return {"recommendations": product_details}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/churn-analysis")
async def get_churn_analysis():
    """
    Get customer churn analysis
    """
    try:
        # Validate data availability
        if data_store['data'] is None:
            raise HTTPException(
                status_code=400, 
                detail="No data available. Please upload or generate data first."
            )
        
        # Validate required columns
        required_columns = config['preprocessing']['required_columns']
        missing_columns = [col for col in required_columns if col not in data_store['data'].columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns for churn analysis: {', '.join(missing_columns)}"
            )
        
        # Debug logging
        print("Debug - Input data shape:", data_store['data'].shape)
        print("Debug - Input data columns:", data_store['data'].columns.tolist())
        
        # Prepare churn features
        try:
            churn_features = prepare_churn_features(data_store['data'])
            if churn_features is None or churn_features.empty:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to generate churn features"
                )
            print("Debug - Churn features shape:", churn_features.shape)
            print("Debug - Churn features columns:", churn_features.columns.tolist())
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error preparing churn features: {str(e)}"
            )
        
        # Train model if not already trained
        if data_store['churn_model'] is None:
            try:
                model, scaler, feature_importance = train_churn_model(churn_features)
                if model is None or scaler is None:
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to train churn model"
                    )
                data_store['churn_model'] = model
                data_store['churn_scaler'] = scaler
                data_store['feature_importance'] = feature_importance
                print("Debug - Model trained successfully")
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error training churn model: {str(e)}"
                )
        
        # Get churn predictions
        try:
            churn_predictions = predict_churn_probability(
                churn_features, 
                data_store['churn_model'], 
                data_store['churn_scaler']
            )
            if churn_predictions is None or churn_predictions.empty:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to generate churn predictions"
                )
            print("Debug - Churn predictions shape:", churn_predictions.shape)
            print("Debug - Churn predictions columns:", churn_predictions.columns.tolist())
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating churn predictions: {str(e)}"
            )
        
        # Store results
        data_store['churn_features'] = churn_features
        data_store['churn_predictions'] = churn_predictions
        
        # Prepare response
        try:
            response = {
                "churn_rate": float(churn_predictions['ChurnProbability'].mean()),
                "high_risk_customers": int((churn_predictions['ChurnProbability'] > config['churn_analysis']['high_risk_threshold']).sum()),
                "feature_importance": data_store['feature_importance'].to_dict(),
                "top_risk_customers": churn_predictions.head(config['churn_analysis']['top_risk_customers']).to_dict()
            }
            print("Debug - Response prepared successfully")
            return response
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error preparing response: {str(e)}"
            )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Unexpected error in churn analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@app.get("/customer/{customer_id}")
async def get_customer_details(customer_id: str):
    """
    Get detailed analysis for a specific customer
    """
    try:
        if data_store['data'] is None:
            raise HTTPException(status_code=400, detail="No data available. Please upload or generate data first.")
        
        # Get customer data
        customer_data = data_store['data'][data_store['data']['CustomerID'] == customer_id]
        if customer_data.empty:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        # Get RFM data
        rfm_data = data_store['rfm_data']
        if rfm_data is not None and customer_id in rfm_data.index:
            rfm_info = rfm_data.loc[customer_id].to_dict()
        else:
            rfm_info = None
        
        # Get cluster data
        cluster_data = data_store['cluster_data']
        if cluster_data is not None and customer_id in cluster_data.index:
            cluster_info = cluster_data.loc[customer_id].to_dict()
        else:
            cluster_info = None
        
        # Get churn prediction
        churn_predictions = data_store['churn_predictions']
        if churn_predictions is not None and customer_id in churn_predictions.index:
            churn_info = churn_predictions.loc[customer_id].to_dict()
        else:
            churn_info = None
        
        # Prepare response
        response = {
            "customer_id": customer_id,
            "total_orders": int(customer_data['InvoiceNo'].nunique()),
            "total_spent": float(customer_data['TotalAmount'].sum()),
            "last_purchase": customer_data['InvoiceDate'].max().strftime('%Y-%m-%d'),
            "rfm_analysis": rfm_info,
            "cluster_analysis": cluster_info,
            "churn_analysis": churn_info
        }
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 