import os
import json
import hmac
import hashlib
import base64
from datetime import datetime, timedelta
import requests
from flask import Flask, request, jsonify, redirect, session
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key')

# Shopify API configuration
SHOPIFY_API_KEY = os.getenv('SHOPIFY_API_KEY')
SHOPIFY_API_SECRET = os.getenv('SHOPIFY_API_SECRET')
SHOPIFY_API_SCOPES = 'read_orders,read_customers,read_products'
SHOPIFY_API_VERSION = '2024-01'  # Use the latest stable version

# Analytics API configuration
ANALYTICS_API_URL = os.getenv('ANALYTICS_API_URL', 'http://api:8000')

def verify_webhook(data, hmac_header):
    """Verify Shopify webhook"""
    calculated_hmac = base64.b64encode(
        hmac.new(
            SHOPIFY_API_SECRET.encode('utf-8'),
            data,
            hashlib.sha256
        ).digest()
    ).decode('utf-8')
    return hmac.compare_digest(calculated_hmac, hmac_header)

@app.route('/install')
def install():
    """Handle Shopify app installation"""
    shop = request.args.get('shop')
    if not shop:
        return 'Shop parameter is required', 400

    # Generate authorization URL
    auth_url = f"https://{shop}/admin/oauth/authorize"
    params = {
        'client_id': SHOPIFY_API_KEY,
        'scope': SHOPIFY_API_SCOPES,
        'redirect_uri': f"{request.host_url}auth/callback",
        'state': hmac.new(
            SHOPIFY_API_SECRET.encode('utf-8'),
            shop.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    }
    
    return redirect(f"{auth_url}?{requests.compat.urlencode(params)}")

@app.route('/auth/callback')
def auth_callback():
    """Handle OAuth callback from Shopify"""
    shop = request.args.get('shop')
    code = request.args.get('code')
    state = request.args.get('state')
    
    # Verify state
    expected_state = hmac.new(
        SHOPIFY_API_SECRET.encode('utf-8'),
        shop.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    if not hmac.compare_digest(state, expected_state):
        return 'Invalid state parameter', 400
    
    # Exchange code for access token
    response = requests.post(
        f"https://{shop}/admin/oauth/access_token",
        json={
            'client_id': SHOPIFY_API_KEY,
            'client_secret': SHOPIFY_API_SECRET,
            'code': code
        }
    )
    
    if response.status_code != 200:
        return 'Failed to get access token', 400
    
    access_token = response.json()['access_token']
    
    # Store access token securely (implement your storage solution)
    store_access_token(shop, access_token)
    
    return 'App installed successfully!'

@app.route('/webhooks/orders/create', methods=['POST'])
def handle_order_webhook():
    """Handle new order webhook from Shopify"""
    # Verify webhook
    hmac_header = request.headers.get('X-Shopify-Hmac-Sha256')
    if not verify_webhook(request.get_data(), hmac_header):
        return 'Invalid webhook signature', 401
    
    # Process order data
    order_data = request.json
    
    # Transform order data to match our analytics format
    analytics_data = transform_order_data(order_data)
    
    # Send to analytics API
    response = requests.post(
        f"{ANALYTICS_API_URL}/upload-data",
        json=analytics_data
    )
    
    if response.status_code != 200:
        return 'Failed to process order data', 500
    
    return 'Order processed successfully'

def transform_order_data(order_data):
    """Transform Shopify order data to analytics format"""
    transformed_data = []
    
    for item in order_data['line_items']:
        transformed_data.append({
            'CustomerID': str(order_data['customer']['id']),
            'InvoiceNo': str(order_data['order_number']),
            'ProductID': str(item['product_id']),
            'Description': item['title'],
            'Category': item.get('product_type', 'Unknown'),
            'Quantity': float(item['quantity']),
            'UnitPrice': float(item['price']),
            'InvoiceDate': order_data['created_at'],
            'TotalAmount': float(item['quantity']) * float(item['price'])
        })
    
    return transformed_data

def store_access_token(shop, access_token):
    """Store access token securely (implement your storage solution)"""
    # TODO: Implement secure token storage
    # This could be a database, encrypted file, or secure key-value store
    pass

def get_access_token(shop):
    """Retrieve access token for a shop"""
    # TODO: Implement token retrieval from your storage solution
    pass

def sync_shopify_data(shop):
    """Sync historical data from Shopify"""
    access_token = get_access_token(shop)
    if not access_token:
        return False
    
    # Get orders from the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    orders = []
    next_page = f"https://{shop}/admin/api/{SHOPIFY_API_VERSION}/orders.json"
    
    while next_page:
        response = requests.get(
            next_page,
            headers={'X-Shopify-Access-Token': access_token}
        )
        
        if response.status_code != 200:
            return False
        
        data = response.json()
        orders.extend(data['orders'])
        
        # Get next page link from headers
        next_page = response.links.get('next', {}).get('url')
    
    # Transform and send data to analytics API
    for order in orders:
        analytics_data = transform_order_data(order)
        requests.post(
            f"{ANALYTICS_API_URL}/upload-data",
            json=analytics_data
        )
    
    return True

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 