import pandas as pd  
import numpy as np  
from scipy.sparse import csr_matrix  
from sklearn.neighbors import NearestNeighbors 

def create_user_item_matrix(df):
    """
    Create user-item matrix for collaborative filtering
    
    Args:
        df: Transaction DataFrame
        
    Returns:
        User-item matrix and mappings for users and items
    """
    # Create user-item purchase matrix
    user_item = df.pivot_table(
        index='CustomerID',
        columns='ProductID',
        values='Quantity',
        aggfunc='sum',
        fill_value=0
    )
    
    # Create mappings for users and items
    user_mapping = {user: i for i, user in enumerate(user_item.index)}
    item_mapping = {item: i for i, item in enumerate(user_item.columns)}
    
    # Inverse mappings
    user_inv_mapping = {i: user for user, i in user_mapping.items()}
    item_inv_mapping = {i: item for item, i in item_mapping.items()}
    
    # Convert to sparse matrix for efficiency
    user_item_matrix = csr_matrix(user_item.values)
    
    return user_item_matrix, user_mapping, item_mapping, user_inv_mapping, item_inv_mapping

def collaborative_filtering(df, user_id, n_recommendations=5):
    """
    Implement collaborative filtering for product recommendations
    
    Args:
        df: Transaction DataFrame
        user_id: CustomerID to generate recommendations for
        n_recommendations: Number of products to recommend
        
    Returns:
        List of recommended product IDs
    """
    # Create user-item matrix
    user_item_matrix, user_mapping, item_mapping, user_inv_mapping, item_inv_mapping = create_user_item_matrix(df)
    
    # Initialize model
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
    model_knn.fit(user_item_matrix)
    
    # Check if user exists in our data
    if user_id not in user_mapping:
        print(f"User {user_id} not found in the dataset.")
        return []
    
    # Get user index
    user_idx = user_mapping[user_id]
    
    # Get user's purchases
    user_purchases = df[df['CustomerID'] == user_id]['ProductID'].unique()
    
    # Find similar users
    distances, indices = model_knn.kneighbors(user_item_matrix[user_idx].reshape(1, -1), n_neighbors=6)
    
    # Get recommendations from similar users
    recommended_products = []
    
    # Skip the first index as it's the user itself
    for idx in indices.flatten()[1:]:
        similar_user_id = user_inv_mapping[idx]
        
        # Get products purchased by similar user
        similar_user_purchases = df[df['CustomerID'] == similar_user_id]['ProductID'].unique()
        
        # Add products user hasn't purchased yet
        for product in similar_user_purchases:
            if product not in user_purchases and product not in recommended_products:
                recommended_products.append(product)
                
                if len(recommended_products) >= n_recommendations:
                    break
    
    return recommended_products

def content_based_recommendation(df, user_id, n_recommendations=5):
    """
    Implement content-based filtering for product recommendations
    
    Args:
        df: Transaction DataFrame with product categories
        user_id: CustomerID to generate recommendations for
        n_recommendations: Number of products to recommend
        
    Returns:
        List of recommended product IDs
    """
    # Get user's purchase history
    user_purchases = df[df['CustomerID'] == user_id]
    
    if user_purchases.empty:
        print(f"User {user_id} not found in the dataset.")
        return []
    
    # Calculate category preferences for the user
    category_preferences = user_purchases.groupby('Category').agg({
        'TotalAmount': 'sum'
    }).reset_index()
    
    # Normalize to get preference weights
    total_spent = category_preferences['TotalAmount'].sum()
    category_preferences['Weight'] = category_preferences['TotalAmount'] / total_spent
    
    # Get products already purchased by user
    purchased_products = user_purchases['ProductID'].unique()
    
    # Find products in preferred categories that user hasn't purchased yet
    recommended_products = []
    
    # Sort categories by weight (preference)
    for category in category_preferences.sort_values('Weight', ascending=False)['Category']:
        # Get products in this category that user hasn't purchased
        category_products = df[
            (df['Category'] == category) & 
            (~df['ProductID'].isin(purchased_products))
        ]['ProductID'].unique()
        
        # Add to recommendations
        for product in category_products:
            if product not in recommended_products:
                recommended_products.append(product)
                
                if len(recommended_products) >= n_recommendations:
                    break
        
        if len(recommended_products) >= n_recommendations:
            break
    
    return recommended_products[:n_recommendations]

def hybrid_recommendation(df, user_id, n_recommendations=5, collab_weight=0.7):
    """
    Combine collaborative and content-based recommendations
    
    Args:
        df: Transaction DataFrame
        user_id: CustomerID to generate recommendations for
        n_recommendations: Number of products to recommend
        collab_weight: Weight for collaborative filtering (0-1)
        
    Returns:
        List of recommended product IDs
    """
    # Get recommendations from both methods
    collab_recs = collaborative_filtering(df, user_id, n_recommendations=n_recommendations*2)
    content_recs = content_based_recommendation(df, user_id, n_recommendations=n_recommendations*2)
    
    # Create weighted recommendations
    collab_score = {prod: collab_weight * (len(collab_recs) - i) / len(collab_recs) 
                    for i, prod in enumerate(collab_recs)}
    
    content_score = {prod: (1 - collab_weight) * (len(content_recs) - i) / len(content_recs) 
                     for i, prod in enumerate(content_recs)}
    
    # Combine scores
    combined_score = {}
    for prod in set(collab_recs + content_recs):
        combined_score[prod] = collab_score.get(prod, 0) + content_score.get(prod, 0)
    
    # Sort by score and return top N
    recommended_products = sorted(combined_score.items(), key=lambda x: x[1], reverse=True)
    return [prod for prod, score in recommended_products[:n_recommendations]]