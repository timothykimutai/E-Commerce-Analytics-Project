import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union
from streamlit.runtime.uploaded_file_manager import UploadedFile

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.data.preprocess import preprocess_data, generate_sample_data
from src.models.customer_segmentation import perform_customer_clustering, perform_rfm_analysis
from src.models.recommendation import hybrid_recommendation, collaborative_filtering, content_based_recommendation
from src.models.churn_prediction import prepare_churn_features, train_churn_model, predict_churn_probability, identify_churn_factors

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_and_preprocess_data(uploaded_file: Optional[UploadedFile] = None) -> pd.DataFrame:
    """
    Load and preprocess data from either uploaded file or sample data.
    
    Args:
        uploaded_file: Optional uploaded CSV file
        
    Returns:
        Preprocessed DataFrame
        
    Raises:
        ValueError: If data loading or preprocessing fails
    """
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # Check if required columns exist
            required_columns = ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'TotalAmount']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        else:
            # Generate sample data
            df = generate_sample_data()
            print("Debug - Sample data columns:", df.columns.tolist())
            print("Debug - Sample data shape:", df.shape)
        
        # Ensure column names are correct
        df = df.rename(columns={
            'customer_id': 'CustomerID',
            'invoice_date': 'InvoiceDate',
            'invoice_no': 'InvoiceNo',
            'total_amount': 'TotalAmount',
            'product_id': 'ProductID',
            'description': 'Description',
            'category': 'Category',
            'quantity': 'Quantity',
            'unit_price': 'UnitPrice'
        })
        
        # Preprocess the data
        df = preprocess_data(df)
        
        # Validate required columns after preprocessing
        required_columns = ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'TotalAmount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns after preprocessing: {', '.join(missing_columns)}")
        
        print("Debug - Final DataFrame columns:", df.columns.tolist())
        print("Debug - Final DataFrame shape:", df.shape)
            
        return df
    except Exception as e:
        raise ValueError(f"Error loading or preprocessing data: {str(e)}")

@st.cache_data(ttl=3600)
def compute_rfm_and_clusters(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Compute RFM analysis and customer clusters.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        Tuple of (rfm_data, cluster_data, cluster_analysis, cluster_names)
    """
    # Debug logging
    print("Debug - Input DataFrame columns:", df.columns.tolist())
    print("Debug - Input DataFrame shape:", df.shape)
    print("Debug - Input DataFrame sample:", df.head())
    
    # Validate input data
    required_columns = ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'TotalAmount']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns for RFM analysis: {', '.join(missing_columns)}")
    
    # Perform RFM analysis
    rfm_data = perform_rfm_analysis(df)
    
    # Debug logging
    print("Debug - RFM data columns:", rfm_data.columns.tolist())
    print("Debug - RFM data shape:", rfm_data.shape)
    print("Debug - RFM data sample:", rfm_data.head())
    
    # Perform clustering
    cluster_data, cluster_analysis, cluster_names = perform_customer_clustering(rfm_data)
    
    # Debug logging
    print("Debug - Cluster data columns:", cluster_data.columns.tolist())
    print("Debug - Cluster data shape:", cluster_data.shape)
    print("Debug - Cluster data sample:", cluster_data.head())
    
    return rfm_data, cluster_data, cluster_analysis, cluster_names

@st.cache_data(ttl=3600)
def compute_churn_analysis(df: pd.DataFrame) -> Tuple[pd.DataFrame, RandomForestClassifier, StandardScaler, pd.DataFrame, pd.DataFrame]:
    """
    Compute churn analysis and predictions.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        Tuple of (churn_features, model, scaler, feature_importance, churn_predictions)
    """
    churn_features = prepare_churn_features(df)
    model, scaler, feature_importance = train_churn_model(churn_features)
    churn_predictions = predict_churn_probability(churn_features, model, scaler)
    return churn_features, model, scaler, feature_importance, churn_predictions

def create_streamlit_app():
    """
    Create Streamlit dashboard for e-commerce analytics.
    
    This dashboard provides three main functionalities:
    1. Customer Segmentation: RFM analysis and clustering
    2. Product Recommendations: Collaborative, content-based, and hybrid recommendations
    3. Churn Prediction: Customer churn risk analysis and retention strategies
    """
    st.set_page_config(
        page_title="E-Commerce Analytics Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("E-Commerce Analytics Dashboard")
    st.markdown("""
    This dashboard provides insights on customer segmentation, product recommendations, 
    and churn prediction for e-commerce businesses.
    """)
    
    # Initialize session state for data persistence
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.data = None
        st.session_state.rfm_data = None
        st.session_state.cluster_data = None
        st.session_state.cluster_analysis = None
        st.session_state.cluster_names = None
    
    # File uploader or sample data option
    st.sidebar.header("Data Options")
    data_option = st.sidebar.radio(
        "Choose data source:",
        ("Use sample data", "Upload your own data")
    )
    
    try:
        if data_option == "Upload your own data":
            uploaded_file = st.sidebar.file_uploader("Upload your transaction data CSV", type="csv")
            if uploaded_file is not None:
                with st.spinner("Loading and preprocessing data..."):
                    try:
                        df = load_and_preprocess_data(uploaded_file)
                        if df is not None and not df.empty:
                            st.session_state.data = df
                            st.session_state.data_loaded = True
                            st.sidebar.success("Data successfully loaded!")
                        else:
                            st.error("Failed to load data. Please check your file format.")
                            return
                    except ValueError as e:
                        st.error(str(e))
                        return
            else:
                if not st.session_state.data_loaded:
                    st.info("Please upload a CSV file with transaction data or use the sample data.")
                    return
        else:
            # Generate sample data
            with st.spinner("Generating sample data..."):
                try:
                    df = load_and_preprocess_data()
                    if df is not None and not df.empty:
                        st.session_state.data = df
                        st.session_state.data_loaded = True
                        st.sidebar.success("Sample data generated successfully!")
                    else:
                        st.error("Failed to generate sample data.")
                        return
                except ValueError as e:
                    st.error(str(e))
                    return
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return
    
    # Main dashboard tabs
    tab1, tab2, tab3 = st.tabs(["Customer Segmentation", "Product Recommendations", "Churn Prediction"])
    
    # Get data
    df = st.session_state.data
    
    # Customer Segmentation Tab
    with tab1:
        st.header("Customer Segmentation")
        
        # Validate data
        if df is None or df.empty:
            st.error("No data available for analysis. Please load or generate data first.")
            return
            
        # Check required columns
        required_columns = ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'TotalAmount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return
        
        # Compute RFM and clustering if not already in session state
        if 'rfm_data' not in st.session_state or st.session_state.rfm_data is None:
            with st.spinner("Analyzing customer segments..."):
                try:
                    # Debug logging
                    st.write("Debug - DataFrame columns:", df.columns.tolist())
                    st.write("Debug - DataFrame shape:", df.shape)
                    st.write("Debug - DataFrame sample:", df.head())
                    
                    # RFM Analysis
                    rfm_data, cluster_data, cluster_analysis, cluster_names = compute_rfm_and_clusters(df)
                    
                    # Validate RFM data
                    if rfm_data is None or rfm_data.empty:
                        st.error("Failed to compute RFM analysis. Please check your data.")
                        return
                        
                    st.session_state.rfm_data = rfm_data
                    st.session_state.cluster_data = cluster_data
                    st.session_state.cluster_analysis = cluster_analysis
                    st.session_state.cluster_names = cluster_names
                except Exception as e:
                    st.error(f"Error computing customer segments: {str(e)}")
                    import logging
                    logging.error(f"RFM analysis error: {str(e)}", exc_info=True)
                    return
        
        # Get data from session state with validation
        try:
            rfm_data = st.session_state.get('rfm_data')
            cluster_data = st.session_state.get('cluster_data')
            cluster_analysis = st.session_state.get('cluster_analysis')
            cluster_names = st.session_state.get('cluster_names')
            
            if any(x is None for x in [rfm_data, cluster_data, cluster_analysis, cluster_names]):
                st.error("Customer segmentation data is not available. Please try refreshing the page.")
                return
                
            if rfm_data.empty or cluster_data.empty:
                st.error("No customer segmentation data available. Please check your data.")
                return
            
            # Display segment metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("RFM Segments")
                
                # RFM segment distribution
                rfm_segment_counts = rfm_data['RFM_Segment'].value_counts().reset_index()
                rfm_segment_counts.columns = ['Segment', 'Count']
                
                fig_rfm = px.bar(
                    rfm_segment_counts, 
                    x='Count', 
                    y='Segment',
                    orientation='h',
                    title='Customer Distribution by RFM Segment',
                    color='Count',
                    color_continuous_scale='Blues'
                )
                
                st.plotly_chart(fig_rfm, use_container_width=True)
            
            with col2:
                st.subheader("Cluster Segments")
                
                # Cluster distribution
                cluster_counts = cluster_data['Cluster_Name'].value_counts().reset_index()
                cluster_counts.columns = ['Cluster', 'Count']
                
                fig_cluster = px.bar(
                    cluster_counts, 
                    x='Count', 
                    y='Cluster',
                    orientation='h',
                    title='Customer Distribution by Cluster',
                    color='Count',
                    color_continuous_scale='Greens'
                )
                
                st.plotly_chart(fig_cluster, use_container_width=True)
            
            # Cluster visualization
            st.subheader("Cluster Visualization")
            
            fig_pca = px.scatter(
                cluster_data, 
                x='PCA1', 
                y='PCA2',
                color='Cluster_Name',
                hover_data=['Recency', 'Frequency', 'Monetary'],
                title='Customer Clusters (PCA Visualization)'
            )
            
            st.plotly_chart(fig_pca, use_container_width=True)
            
            # Cluster details
            st.subheader("Cluster Details")
            
            # Show cluster metrics
            cluster_metrics = cluster_analysis.copy()
            cluster_metrics['Cluster_Name'] = cluster_metrics.index.map(cluster_names)
            cluster_metrics = cluster_metrics.reset_index()
            
            fig_metrics = make_subplots(
                rows=1, 
                cols=3,
                subplot_titles=('Average Recency (Days)', 'Average Purchase Frequency', 'Average Monetary Value ($)')
            )
            
            # Add bars for each metric
            fig_metrics.add_trace(
                go.Bar(
                    x=cluster_metrics['Cluster_Name'],
                    y=cluster_metrics['Recency'],
                    name='Recency',
                    marker_color='skyblue'
                ),
                row=1, col=1
            )
            
            fig_metrics.add_trace(
                go.Bar(
                    x=cluster_metrics['Cluster_Name'],
                    y=cluster_metrics['Frequency'],
                    name='Frequency',
                    marker_color='lightgreen'
                ),
                row=1, col=2
            )
            
            fig_metrics.add_trace(
                go.Bar(
                    x=cluster_metrics['Cluster_Name'],
                    y=cluster_metrics['Monetary'],
                    name='Monetary',
                    marker_color='salmon'
                ),
                row=1, col=3
            )
            
            fig_metrics.update_layout(
                height=400,
                showlegend=False,
                title_text="Cluster Characteristics"
            )
            
            st.plotly_chart(fig_metrics, use_container_width=True)
            
            # Customer lookup
            st.subheader("Customer Lookup")
            
            # Get list of customers
            customers = cluster_data.index.tolist()
            selected_customer = st.selectbox("Select a customer:", customers)
            
            if selected_customer:
                customer_data = cluster_data.loc[selected_customer]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("RFM Segment", customer_data['RFM_Segment'])
                    st.metric("Cluster", customer_data['Cluster_Name'])
                    st.metric("Recency (Days)", f"{customer_data['Recency']:.0f}")
                
                with col2:
                    st.metric("Frequency (Orders)", f"{customer_data['Frequency']:.0f}")
                    st.metric("Monetary Value ($)", f"${customer_data['Monetary']:.2f}")
                    st.metric("RFM Score", f"{customer_data['RFM_Score']}")
        except Exception as e:
            st.error(f"Error displaying customer segmentation: {str(e)}")
            import logging
            logging.error(f"Customer segmentation error: {str(e)}", exc_info=True)
    
    # Product Recommendations Tab
    with tab2:
        st.header("Product Recommendations")
        
        # Customer selection
        customers = df['CustomerID'].unique().tolist()
        selected_customer = st.selectbox("Select a customer for recommendations:", customers, key="rec_customer")
        
        # Method selection
        rec_method = st.radio(
            "Select recommendation method:",
            ("Collaborative Filtering", "Content-Based Filtering", "Hybrid Approach")
        )
        
        # Number of recommendations
        n_recommendations = st.slider("Number of recommendations:", 1, 10, 5)
        
        # Get recommendations
        if st.button("Generate Recommendations"):
            with st.spinner("Generating recommendations..."):
                try:
                    if rec_method == "Collaborative Filtering":
                        recommendations = collaborative_filtering(df, selected_customer, n_recommendations)
                        method_description = """
                        **Collaborative Filtering** recommends products based on similar customers' purchasing patterns.
                        It finds customers with similar preferences and suggests products they've purchased that the target customer hasn't tried yet.
                        """
                    elif rec_method == "Content-Based Filtering":
                        recommendations = content_based_recommendation(df, selected_customer, n_recommendations)
                        method_description = """
                        **Content-Based Filtering** recommends products based on the customer's own preferences.
                        It analyzes product categories the customer has purchased and recommends similar products from those categories.
                        """
                    else:  # Hybrid
                        recommendations = hybrid_recommendation(df, selected_customer, n_recommendations)
                        method_description = """
                        **Hybrid Approach** combines both collaborative and content-based filtering to provide more balanced recommendations.
                        It leverages both similar customers' patterns and the customer's own preferences.
                        """
                    
                    st.info(method_description)
                    
                    if recommendations:
                        # Get product details
                        product_details = {}
                        
                        for product_id in recommendations:
                            product_data = df[df['ProductID'] == product_id].iloc[0]
                            product_details[product_id] = {
                                'Description': product_data['Description'],
                                'Category': product_data['Category'],
                                'Price': product_data['UnitPrice']
                            }
                        
                        # Display recommendations
                        st.subheader("Recommended Products")
                        
                        # Show as cards
                        cols = st.columns(min(3, len(recommendations)))
                        for i, product_id in enumerate(recommendations):
                            col_idx = i % len(cols)
                            with cols[col_idx]:
                                details = product_details[product_id]
                                st.markdown(f"""
                                **{details['Description']}**  
                                Category: {details['Category']}  
                                Price: ${details['Price']:.2f}  
                                Product ID: {product_id}
                                """)
                                st.write("---")
                        
                        # Customer purchase history
                        st.subheader("Customer Purchase History")
                        
                        # Get customer's past purchases
                        past_purchases = df[df['CustomerID'] == selected_customer]
                        past_products = past_purchases.groupby('ProductID').agg({
                            'Description': 'first',
                            'Category': 'first',
                            'Quantity': 'sum',
                            'TotalAmount': 'sum',
                            'InvoiceDate': 'max'
                        }).reset_index()
                        
                        past_products = past_products.sort_values('InvoiceDate', ascending=False)
                        
                        # Show past purchases table
                        st.dataframe(
                            past_products[['ProductID', 'Description', 'Category', 'Quantity', 'TotalAmount']],
                            use_container_width=True
                        )
                    else:
                        st.warning("No recommendations found for this customer.")
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
        
        # Recommendation explanation
        with st.expander("How do recommendation systems work?"):
            st.markdown("""
            ### Recommendation Systems Explanation
            
            #### Collaborative Filtering
            This approach finds customers with similar purchasing patterns and recommends products they've bought that the target customer hasn't tried yet. It's based on the idea that people who agreed in the past will agree in the future.
            
            #### Content-Based Filtering
            This approach recommends products similar to what the customer has purchased before. It analyzes product categories and attributes to suggest similar items without requiring data from other customers.
            
            #### Hybrid Approach
            This combines both methods above to provide more balanced recommendations, leveraging both the wisdom of the crowd (collaborative) and personal preferences (content-based).
            """)
    
    # Churn Prediction Tab
    with tab3:
        st.header("Churn Prediction")
        
        # Compute churn features and model if not already in session state
        if 'churn_features' not in st.session_state or 'churn_model' not in st.session_state:
            with st.spinner("Analyzing churn risk..."):
                try:
                    # Prepare churn features
                    churn_features, churn_model, churn_scaler, feature_importance, churn_predictions = compute_churn_analysis(df)
                    st.session_state.churn_features = churn_features
                    st.session_state.churn_model = churn_model
                    st.session_state.churn_scaler = churn_scaler
                    st.session_state.feature_importance = feature_importance
                    st.session_state.churn_predictions = churn_predictions
                except Exception as e:
                    st.error(f"Error computing churn analysis: {str(e)}")
                    return
        
        # Get data from session state with validation
        try:
            churn_features = st.session_state.get('churn_features')
            churn_model = st.session_state.get('churn_model')
            churn_scaler = st.session_state.get('churn_scaler')
            feature_importance = st.session_state.get('feature_importance')
            churn_predictions = st.session_state.get('churn_predictions')
            
            if any(x is None for x in [churn_features, churn_model, churn_scaler, feature_importance, churn_predictions]):
                st.error("Churn analysis data is not available. Please try refreshing the page.")
                return
            
            # Display overall churn metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                churn_rate = churn_features['ChurnRisk'].mean() * 100
                st.metric("Average Churn Risk", f"{churn_rate:.1f}%")
            
            with col2:
                high_risk = (churn_predictions['ChurnProbability'] > 0.7).sum()
                st.metric("High Risk Customers", f"{high_risk}")
            
            with col3:
                medium_risk = ((churn_predictions['ChurnProbability'] > 0.3) & 
                              (churn_predictions['ChurnProbability'] <= 0.7)).sum()
                st.metric("Medium Risk Customers", f"{medium_risk}")
            
            # Churn distribution
            st.subheader("Churn Risk Distribution")
            
            fig_churn_dist = px.histogram(
                churn_predictions, 
                x='ChurnProbability',
                nbins=20,
                title='Distribution of Churn Probability',
                color_discrete_sequence=['indianred']
            )
            
            fig_churn_dist.update_layout(
                xaxis_title="Churn Probability",
                yaxis_title="Number of Customers"
            )
            
            st.plotly_chart(fig_churn_dist, use_container_width=True)
            
            # Display top churn factors
            st.subheader("Key Churn Factors")
            
            # Plot feature importance
            fig_features = px.bar(
                feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance for Churn Prediction',
                color='Importance',
                color_continuous_scale='Reds'
            )
            
            st.plotly_chart(fig_features, use_container_width=True)
            
            # High-risk customers
            st.subheader("Customers at Risk")
            
            # Show top 10 customers at risk
            high_risk_customers = churn_predictions.sort_values('ChurnProbability', ascending=False).head(10)
            
            # Table of high risk customers
            st.dataframe(
                high_risk_customers[['ChurnProbability', 'DaysSinceLastPurchase', 'Recency', 'NumberOfOrders', 'TotalSpent']],
                use_container_width=True
            )
            
            # Customer lookup
            st.subheader("Customer Churn Analysis")
            
            # Get list of customers
            customers = churn_predictions.index.tolist()
            selected_customer = st.selectbox("Select a customer:", customers, key="churn_customer")
            
            if selected_customer:
                customer_data = churn_predictions.loc[selected_customer]
                
                # Calculate risk level
                if customer_data['ChurnProbability'] > 0.7:
                    risk_level = "High Risk"
                    risk_color = "🔴"
                elif customer_data['ChurnProbability'] > 0.3:
                    risk_level = "Medium Risk"
                    risk_color = "🟡"
                else:
                    risk_level = "Low Risk"
                    risk_color = "🟢"
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Churn Risk", f"{risk_color} {risk_level}")
                    st.metric("Churn Probability", f"{customer_data['ChurnProbability']:.2f}")
                    st.metric("Days Since Last Purchase", f"{customer_data['DaysSinceLastPurchase']:.0f}")
                
                with col2:
                    st.metric("Total Orders", f"{customer_data['NumberOfOrders']:.0f}")
                    st.metric("Total Spent", f"${customer_data['TotalSpent']:.2f}")
                    st.metric("Average Order Value", f"${customer_data['AvgOrderValue']:.2f}")
                
                # Get churn factors
                churn_factors = identify_churn_factors(selected_customer, churn_predictions, feature_importance)
                
                st.subheader("Key Risk Factors")
                
                for factor in churn_factors:
                    st.write(f"• {factor}")
                
                # Retention recommendations
                st.subheader("Retention Recommendations")
                
                if customer_data['ChurnProbability'] > 0.5:
                    if customer_data['TotalSpent'] > churn_predictions['TotalSpent'].median():
                        st.write("• High-value customer at risk - consider personalized outreach")
                        st.write("• Offer exclusive discount on products in their preferred category")
                    else:
                        st.write("• Send re-engagement email with personalized product recommendations")
                        st.write("• Offer discount on next purchase")
                    
                    if customer_data['NumberOfOrders'] > 1:
                        st.write("• Remind of past positive experiences with your store")
                        st.write("• Loyalty program invitation with immediate benefits")
                    
                    days_inactive = customer_data['DaysSinceLastPurchase']
                    if days_inactive > 60:
                        st.write(f"• Customer inactive for {days_inactive:.0f} days - critical to re-engage soon")
                else:
                    st.write("• Continue regular engagement - customer shows healthy purchase patterns")
                    st.write("• Consider loyalty program to increase purchase frequency")
        except Exception as e:
            st.error(f"Error displaying churn analysis: {str(e)}")
            import logging
            logging.error(f"Churn analysis error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    try:
        # Initialize Streamlit session state if not already initialized
        if not st.session_state:
            st.session_state.update({
                'data_loaded': False,
                'rfm_data': None,
                'cluster_data': None,
                'cluster_analysis': None,
                'cluster_names': None,
                'churn_features': None,
                'churn_model': None,
                'churn_scaler': None,
                'feature_importance': None,
                'churn_predictions': None
            })
        
        # Run the Streamlit app
        create_streamlit_app()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        # Log the error for debugging
        import logging
        logging.error(f"Dashboard error: {str(e)}", exc_info=True)