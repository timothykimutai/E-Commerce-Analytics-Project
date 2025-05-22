import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)


def generate_sample_data(n_customers=1000, n_products=200, n_transactions=10000, start_date="2024-01-01", end_date="2025-05-01"):
    """
    Generate sample e-commerce transaction data.
    
    Args:
        n_customers: Number of unique customers
        n_products: Number of unique products
        n_transactions: Number of transactions to generate
        start_date: Starting date for transactions
        end_date: End date for transactions
        
    Returns:
        DataFrame with columns: CustomerID, InvoiceNo, ProductID, Description, Quantity, UnitPrice, InvoiceDate
    """
    # Create customer IDs
    customer_ids = [f'C{i:05d}' for i in range(1, n_customers + 1)]
    
    # Create product info
    product_categories = ['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports', 'Books', 'Toys', 'Food', 'Health']
    products = []
    
    for i in range(1, n_products + 1):
        cat = np.random.choice(product_categories)
        name = f"{cat} Item {i}"
        price = round(np.random.uniform(5, 200), 2)
        product_id = f"P{i:05d}"
        products.append((product_id, name, cat, price))
    
    product_df = pd.DataFrame(products, columns=['ProductID', 'Description', 'Category', 'BasePrice'])
    
    # Generate transactions
    transactions = []
    invoice_counter = 1
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    date_range = (end_date - start_date).days
    
    # Create skewed customer distribution (some customers buy more frequently)
    customer_weights = np.random.exponential(scale=1.0, size=len(customer_ids))
    customer_weights = customer_weights / np.sum(customer_weights)
    
    # Some products are more popular than others
    product_weights = np.random.exponential(scale=1.0, size=len(products))
    product_weights = product_weights / np.sum(product_weights)
    
    for _ in range(n_transactions):
        # Select customer (weighted)
        customer = np.random.choice(customer_ids, p=customer_weights)
        
        # Create invoice date
        random_days = np.random.randint(0, date_range)
        invoice_date = start_date + pd.Timedelta(days=random_days)
        
        # Create invoice number
        invoice_no = f"INV{invoice_counter:06d}"
        invoice_counter += 1
        
        # Determine how many items in this invoice (1-10)
        n_items = np.random.randint(1, 6)
        
        # Select products for this invoice (weighted)
        selected_products = np.random.choice(range(len(products)), size=n_items, replace=False, p=product_weights)
        
        for prod_idx in selected_products:
            product_id, description, category, base_price = products[prod_idx]
            
            # Add small price variations
            unit_price = round(base_price * np.random.uniform(0.95, 1.05), 2)
            
            # Determine quantity (usually 1-5)
            quantity = np.random.randint(1, 6)
            
            transactions.append([
                customer, 
                invoice_no,
                product_id,
                description,
                category,
                quantity,
                unit_price,
                invoice_date
            ])
    
    # Create DataFrame with correct column names
    df = pd.DataFrame(transactions, columns=[
        'CustomerID', 'InvoiceNo', 'ProductID', 'Description', 'Category', 'Quantity', 'UnitPrice', 'InvoiceDate'
    ])
    
    # Calculate TotalAmount
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    
    # Ensure all required columns are present
    required_columns = ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'TotalAmount']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in generated data: {', '.join(missing_columns)}")
    
    return df

def load_data():
    """
    Load data or generate sample data if file doesn't exist
    
    Returns:
        DataFrame with transaction data
    """
    # For this project, we'll generate sample data
    
    df = generate_sample_data()
    print(f"Generated sample data with {df['CustomerID'].nunique()} customers and {df['InvoiceNo'].nunique()} transactions")
    return df

def preprocess_data(df):
    """
    Clean and preprocess the transaction data
    
    Args:
        df: Raw transaction DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Convert InvoiceDate to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df_clean['InvoiceDate']):
        df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    
    # Filter out any potential negative quantities or returns
    df_clean = df_clean[df_clean['Quantity'] > 0]
    
    # Filter out any potential zero or negative prices
    df_clean = df_clean[df_clean['UnitPrice'] > 0]
    
    # Calculate TotalAmount if it doesn't exist
    if 'TotalAmount' not in df_clean.columns:
        df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']
    
    # Add Year-Month for time-based analysis
    df_clean['YearMonth'] = df_clean['InvoiceDate'].dt.to_period('M')
    
    return df_clean

def exploratory_data_analysis(df):
    """
    Perform exploratory data analysis on the transaction data
    
    Args:
        df: Preprocessed transaction DataFrame
        
    Returns:
        Dictionary of analysis results
    """
    results = {}
    
    # Basic statistics
    results['n_transactions'] = df['InvoiceNo'].nunique()
    results['n_customers'] = df['CustomerID'].nunique()
    results['n_products'] = df['ProductID'].nunique()
    results['date_range'] = (df['InvoiceDate'].min(), df['InvoiceDate'].max())
    results['total_revenue'] = df['TotalAmount'].sum()
    
    # Customer metrics
    customer_stats = df.groupby('CustomerID').agg(
        total_spent=('TotalAmount', 'sum'),
        avg_order_value=('TotalAmount', lambda x: x.sum() / df.loc[x.index, 'InvoiceNo'].nunique()),
        n_orders=('InvoiceNo', lambda x: x.nunique()),
        n_products=('ProductID', lambda x: x.nunique())
    )
    
    results['avg_customer_lifetime_value'] = customer_stats['total_spent'].mean()
    results['avg_orders_per_customer'] = customer_stats['n_orders'].mean()
    
    # Product metrics
    product_stats = df.groupby('ProductID').agg(
        total_quantity=('Quantity', 'sum'),
        total_revenue=('TotalAmount', 'sum'),
        n_customers=('CustomerID', lambda x: x.nunique())
    )
    
    results['top_products'] = product_stats.sort_values('total_revenue', ascending=False).head(10)
    
    # Time metrics
    time_stats = df.groupby(df['InvoiceDate'].dt.to_period('M')).agg(
        total_revenue=('TotalAmount', 'sum'),
        n_orders=('InvoiceNo', lambda x: x.nunique()),
        n_customers=('CustomerID', lambda x: x.nunique())
    )
    
    results['monthly_stats'] = time_stats
    
    return results