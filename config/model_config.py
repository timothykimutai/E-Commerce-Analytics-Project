# Customer Segmentation
SEGMENTATION_CONFIG = {
    'n_clusters': 5,
    'rfm_weights': {'recency': 0.35, 'frequency': 0.35, 'monetary': 0.3},
    'random_state': 42,
    'standardize': True
}

# Recommendation System
RECOMMENDATION_CONFIG = {
    'algorithm': 'collaborative',
    'n_recommendations': 5,
    'similarity_metric': 'cosine',
    'min_interactions': 3,
    'cold_start_strategy': 'popular_items'
}

# Churn Prediction
CHURN_CONFIG = {
    'model_type': 'random_forest',
    'features': [
        'days_since_last_purchase', 
        'purchase_frequency', 
        'avg_order_value', 
        'total_orders',
        'return_rate',
        'cart_abandonment_rate',
        'email_engagement_score'
    ],
    'test_size': 0.2,
    'threshold': 0.7,
    'class_weight': 'balanced'
}