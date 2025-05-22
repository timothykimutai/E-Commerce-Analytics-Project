import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union
from streamlit.runtime.uploaded_file_manager import UploadedFile
import time

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.data.preprocess import preprocess_data, generate_sample_data
from src.models.customer_segmentation import perform_customer_clustering, perform_rfm_analysis
from src.models.recommendation import hybrid_recommendation, collaborative_filtering, content_based_recommendation
from src.models.churn_prediction import prepare_churn_features, train_churn_model, predict_churn_probability, identify_churn_factors

# Custom CSS for enhanced styling
def load_custom_css():
    st.markdown("""
    <style>
    /* Main background and styling */
    .main {
        padding-top: 1rem;
    }
    
    /* Custom metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        color: white;
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Custom headers */
    .custom-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        color: #2C3E50;
        font-size: 1.5rem;
        font-weight: 600;
        border-bottom: 2px solid #3498DB;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 20px;
        background-color: white;
        border-radius: 10px;
        color: #2C3E50;
        font-weight: 600;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 2px solid #4ECDC4;
    }
    
    /* Custom info boxes */
    .info-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        border-left: 5px solid #00b894;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        border-left: 5px solid #d63031;
    }
    
    .success-box {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        border-left: 5px solid #00b894;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        transition: border-color 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

def create_animated_metric_card(title, value, delta=None, delta_color="normal"):
    """Create an animated metric card with custom styling"""
    delta_html = ""
    if delta:
        color = "#00b894" if delta_color == "normal" else "#d63031"
        delta_html = f'<div style="font-size: 0.8rem; color: {color}; margin-top: 0.5rem;">📈 {delta}</div>'
    
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.9rem; opacity: 0.8;">{title}</div>
        <div style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def create_info_box(message, box_type="info"):
    """Create styled info boxes"""
    box_class = f"{box_type}-box"
    st.markdown(f'<div class="{box_class}">{message}</div>', unsafe_allow_html=True)

def create_progress_animation():
    """Create an animated progress bar"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(1, 101):
        progress_bar.progress(i)
        status_text.text(f'Loading... {i}%')
        time.sleep(0.01)
    
    status_text.text('Complete!')
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()

@st.cache_data(ttl=3600)
def load_and_preprocess_data(uploaded_file: Optional[UploadedFile] = None) -> pd.DataFrame:
    """Load and preprocess data with enhanced error handling"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            required_columns = ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'TotalAmount']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        else:
            df = generate_sample_data()
        
        # Column renaming
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
        
        df = preprocess_data(df)
        
        # Final validation
        required_columns = ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'TotalAmount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns after preprocessing: {', '.join(missing_columns)}")
            
        return df
    except Exception as e:
        raise ValueError(f"Error loading or preprocessing data: {str(e)}")

@st.cache_data(ttl=3600)
def compute_rfm_and_clusters(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Compute RFM analysis and customer clusters"""
    required_columns = ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'TotalAmount']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns for RFM analysis: {', '.join(missing_columns)}")
    
    rfm_data = perform_rfm_analysis(df)
    cluster_data, cluster_analysis, cluster_names = perform_customer_clustering(rfm_data)
    
    return rfm_data, cluster_data, cluster_analysis, cluster_names

@st.cache_data(ttl=3600)
def compute_churn_analysis(df: pd.DataFrame) -> Tuple[pd.DataFrame, RandomForestClassifier, StandardScaler, pd.DataFrame, pd.DataFrame]:
    """Compute churn analysis and predictions"""
    churn_features = prepare_churn_features(df)
    model, scaler, feature_importance = train_churn_model(churn_features)
    churn_predictions = predict_churn_probability(churn_features, model, scaler)
    return churn_features, model, scaler, feature_importance, churn_predictions

def create_dashboard_header():
    """Create an attractive dashboard header"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 class="custom-header">🚀 E-Commerce Analytics Dashboard</h1>
        <p style="font-size: 1.2rem; color: #7f8c8d; margin-top: -1rem;">
            Unlock the power of your data with AI-driven insights
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_streamlit_app():
    """Enhanced Streamlit dashboard with modern UI"""
    st.set_page_config(
        page_title="E-Commerce Analytics Dashboard",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Dashboard header
    create_dashboard_header()
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.update({
            'data_loaded': False,
            'data': None,
            'rfm_data': None,
            'cluster_data': None,
            'cluster_analysis': None,
            'cluster_names': None
        })
    
    # Enhanced sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        
        # Data source selection with styling
        st.markdown("#### 📊 Data Source")
        data_option = st.radio(
            "Choose your data source:",
            ("🧪 Use sample data", "📁 Upload your own data"),
            help="Sample data is great for testing, upload your own for real insights!"
        )
        
        # Theme selector
        st.markdown("#### 🎨 Dashboard Theme")
        theme = st.selectbox(
            "Select theme:",
            ["🌊 Ocean Blue", "🌸 Sunset Pink", "🌱 Nature Green", "🌙 Dark Mode"],
            help="Choose your preferred color scheme"
        )
    
    # Data loading section
    try:
        if "Upload your own data" in data_option:
            uploaded_file = st.sidebar.file_uploader(
                "📤 Upload your transaction data CSV", 
                type="csv",
                help="Your CSV should contain CustomerID, InvoiceDate, InvoiceNo, and TotalAmount columns"
            )
            
            if uploaded_file is not None:
                with st.spinner("🔄 Processing your data..."):
                    try:
                        create_progress_animation()
                        df = load_and_preprocess_data(uploaded_file)
                        if df is not None and not df.empty:
                            st.session_state.data = df
                            st.session_state.data_loaded = True
                            create_info_box("✅ Data successfully loaded and processed!", "success")
                        else:
                            st.error("❌ Failed to load data. Please check your file format.")
                            return
                    except ValueError as e:
                        st.error(f"❌ {str(e)}")
                        return
            else:
                if not st.session_state.data_loaded:
                    create_info_box("📋 Please upload a CSV file or use sample data to get started.", "info")
                    return
        else:
            with st.spinner("🎲 Generating sample data..."):
                try:
                    df = load_and_preprocess_data()
                    if df is not None and not df.empty:
                        st.session_state.data = df
                        st.session_state.data_loaded = True
                        create_info_box("🎯 Sample data generated successfully! Ready for analysis.", "success")
                    else:
                        st.error("❌ Failed to generate sample data.")
                        return
                except ValueError as e:
                    st.error(f"❌ {str(e)}")
                    return
    except Exception as e:
        st.error(f"💥 An unexpected error occurred: {str(e)}")
        return
    
    # Enhanced main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "👥 Customer Segmentation", 
        "🎯 Product Recommendations", 
        "⚠️ Churn Prediction",
        "📈 Dashboard Overview"
    ])
    
    df = st.session_state.data
    
    # Dashboard Overview Tab (New)
    with tab4:
        st.markdown('<h2 class="sub-header">📊 Executive Dashboard</h2>', unsafe_allow_html=True)
        
        if df is not None and not df.empty:
            # Key metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_customers = df['CustomerID'].nunique()
                create_animated_metric_card("Total Customers", f"{total_customers:,}", "📈 Active user base")
            
            with col2:
                total_revenue = df['TotalAmount'].sum()
                create_animated_metric_card("Total Revenue", f"${total_revenue:,.2f}", "💰 Lifetime value")
            
            with col3:
                avg_order_value = df['TotalAmount'].mean()
                create_animated_metric_card("Avg Order Value", f"${avg_order_value:.2f}", "📊 Per transaction")
            
            with col4:
                total_orders = df['InvoiceNo'].nunique()
                create_animated_metric_card("Total Orders", f"{total_orders:,}", "🛒 Completed transactions")
            
            # Interactive charts
            st.markdown("### 📈 Sales Trends")
            
            # Time series analysis
            df_time = df.copy()
            df_time['InvoiceDate'] = pd.to_datetime(df_time['InvoiceDate'])
            daily_sales = df_time.groupby(df_time['InvoiceDate'].dt.date)['TotalAmount'].sum().reset_index()
            daily_sales.columns = ['Date', 'Revenue']
            
            fig_sales = px.line(
                daily_sales, 
                x='Date', 
                y='Revenue',
                title='📈 Daily Revenue Trend',
                color_discrete_sequence=['#667eea']
            )
            fig_sales.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2C3E50'),
                title_font_size=20
            )
            st.plotly_chart(fig_sales, use_container_width=True)
            
            # Category analysis
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Category' in df.columns:
                    category_sales = df.groupby('Category')['TotalAmount'].sum().reset_index()
                    fig_cat = px.pie(
                        category_sales, 
                        values='TotalAmount', 
                        names='Category',
                        title='🍕 Revenue by Category',
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_cat.update_layout(font=dict(color='#2C3E50'))
                    st.plotly_chart(fig_cat, use_container_width=True)
            
            with col2:
                # Top products
                if 'Description' in df.columns:
                    top_products = df.groupby('Description')['TotalAmount'].sum().sort_values(ascending=False).head(10)
                    fig_products = px.bar(
                        x=top_products.values,
                        y=top_products.index,
                        orientation='h',
                        title='🏆 Top 10 Products by Revenue',
                        color=top_products.values,
                        color_continuous_scale='Viridis'
                    )
                    fig_products.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#2C3E50')
                    )
                    st.plotly_chart(fig_products, use_container_width=True)
    
    # Enhanced Customer Segmentation Tab
    with tab1:
        st.markdown('<h2 class="sub-header">👥 Customer Segmentation Analysis</h2>', unsafe_allow_html=True)
        
        if df is None or df.empty:
            create_info_box("❌ No data available for analysis. Please load data first.", "warning")
            return
        
        # Compute RFM and clustering
        if 'rfm_data' not in st.session_state or st.session_state.rfm_data is None:
            with st.spinner("🧠 Analyzing customer segments..."):
                try:
                    create_progress_animation()
                    rfm_data, cluster_data, cluster_analysis, cluster_names = compute_rfm_and_clusters(df)
                    
                    if rfm_data is None or rfm_data.empty:
                        st.error("❌ Failed to compute RFM analysis.")
                        return
                        
                    st.session_state.rfm_data = rfm_data
                    st.session_state.cluster_data = cluster_data
                    st.session_state.cluster_analysis = cluster_analysis
                    st.session_state.cluster_names = cluster_names
                    
                    create_info_box("✅ Customer segmentation completed successfully!", "success")
                except Exception as e:
                    st.error(f"❌ Error computing customer segments: {str(e)}")
                    return
        
        try:
            rfm_data = st.session_state.get('rfm_data')
            cluster_data = st.session_state.get('cluster_data')
            cluster_analysis = st.session_state.get('cluster_analysis')
            cluster_names = st.session_state.get('cluster_names')
            
            if any(x is None for x in [rfm_data, cluster_data, cluster_analysis, cluster_names]):
                st.error("❌ Customer segmentation data is not available.")
                return
            
            # Interactive segment selector
            st.markdown("### 🎛️ Segment Explorer")
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                segment_type = st.selectbox(
                    "Select Analysis Type:",
                    ["RFM Segments", "Cluster Analysis", "Combined View"],
                    help="Choose how you want to view your customer segments"
                )
                
                show_details = st.checkbox("Show detailed metrics", value=True)
            
            with col1:
                if segment_type == "RFM Segments":
                    rfm_segment_counts = rfm_data['RFM_Segment'].value_counts().reset_index()
                    rfm_segment_counts.columns = ['Segment', 'Count']
                    
                    fig_rfm = px.sunburst(
                        rfm_segment_counts,
                        path=['Segment'],
                        values='Count',
                        title='🎯 Customer Distribution by RFM Segment',
                        color='Count',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_rfm, use_container_width=True)
                
                elif segment_type == "Cluster Analysis":
                    cluster_counts = cluster_data['Cluster_Name'].value_counts().reset_index()
                    cluster_counts.columns = ['Cluster', 'Count']
                    
                    fig_cluster = px.treemap(
                        cluster_counts,
                        path=['Cluster'],
                        values='Count',
                        title='🗂️ Customer Clusters Distribution',
                        color='Count',
                        color_continuous_scale='Greens'
                    )
                    st.plotly_chart(fig_cluster, use_container_width=True)
            
            # Interactive cluster visualization
            st.markdown("### 🎨 Interactive Cluster Map")
            
            # Add filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_monetary = st.slider(
                    "Min Monetary Value ($)",
                    min_value=float(cluster_data['Monetary'].min()),
                    max_value=float(cluster_data['Monetary'].max()),
                    value=float(cluster_data['Monetary'].min())
                )
            
            with col2:
                max_recency = st.slider(
                    "Max Recency (Days)",
                    min_value=int(cluster_data['Recency'].min()),
                    max_value=int(cluster_data['Recency'].max()),
                    value=int(cluster_data['Recency'].max())
                )
            
            with col3:
                selected_clusters = st.multiselect(
                    "Select Clusters",
                    options=cluster_data['Cluster_Name'].unique(),
                    default=cluster_data['Cluster_Name'].unique()
                )
            
            # Filter data based on selections
            filtered_data = cluster_data[
                (cluster_data['Monetary'] >= min_monetary) &
                (cluster_data['Recency'] <= max_recency) &
                (cluster_data['Cluster_Name'].isin(selected_clusters))
            ]
            
            # Enhanced scatter plot
            fig_pca = px.scatter(
                filtered_data,
                x='PCA1',
                y='PCA2',
                color='Cluster_Name',
                size='Monetary',
                hover_data=['Recency', 'Frequency', 'Monetary'],
                title='🎯 Customer Clusters (Interactive PCA Visualization)',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            
            fig_pca.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2C3E50'),
                title_font_size=20
            )
            
            st.plotly_chart(fig_pca, use_container_width=True)
            
            # Customer search and analysis
            st.markdown("### 🔍 Customer Deep Dive")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                search_customer = st.text_input(
                    "🔎 Search Customer ID:",
                    placeholder="Enter customer ID to analyze...",
                    help="Type a customer ID to get detailed insights"
                )
            
            with col2:
                if st.button("🚀 Generate Customer Profile", use_container_width=True):
                    if search_customer and search_customer in cluster_data.index:
                        customer_data = cluster_data.loc[search_customer]
                        
                        # Create customer profile card
                        st.markdown("#### 📋 Customer Profile")
                        
                        profile_col1, profile_col2, profile_col3 = st.columns(3)
                        
                        with profile_col1:
                            create_animated_metric_card(
                                "RFM Segment", 
                                customer_data['RFM_Segment'],
                                "Customer category"
                            )
                        
                        with profile_col2:
                            create_animated_metric_card(
                                "Cluster", 
                                customer_data['Cluster_Name'],
                                "Behavioral group"
                            )
                        
                        with profile_col3:
                            create_animated_metric_card(
                                "RFM Score", 
                                customer_data['RFM_Score'],
                                "Overall rating"
                            )
                        
                        # Detailed metrics
                        st.markdown("#### 📊 Detailed Metrics")
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric(
                                "🕐 Recency (Days)", 
                                f"{customer_data['Recency']:.0f}",
                                help="Days since last purchase"
                            )
                        
                        with metric_col2:
                            st.metric(
                                "🔄 Frequency", 
                                f"{customer_data['Frequency']:.0f}",
                                help="Number of purchases"
                            )
                        
                        with metric_col3:
                            st.metric(
                                "💰 Monetary ($)", 
                                f"${customer_data['Monetary']:.2f}",
                                help="Total spending"
                            )
                    else:
                        create_info_box("❌ Customer not found. Please check the ID.", "warning")
        
        except Exception as e:
            st.error(f"❌ Error displaying customer segmentation: {str(e)}")
    
    # Enhanced Product Recommendations Tab
    with tab2:
        st.markdown('<h2 class="sub-header">🎯 AI-Powered Product Recommendations</h2>', unsafe_allow_html=True)
        
        # Enhanced recommendation interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            customers = df['CustomerID'].unique().tolist()
            selected_customer = st.selectbox(
                "🎭 Select Customer:",
                customers,
                key="rec_customer",
                help="Choose a customer to generate personalized recommendations"
            )
        
        with col2:
            rec_method = st.selectbox(
                "🧠 AI Method:",
                ["🤝 Collaborative Filtering", "📊 Content-Based", "🎯 Hybrid AI"],
                help="Different AI approaches for recommendations"
            )
        
        # Advanced options in expander
        with st.expander("⚙️ Advanced Options"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_recommendations = st.slider("📊 Number of recommendations:", 1, 15, 8)
            
            with col2:
                confidence_threshold = st.slider("🎯 Confidence threshold:", 0.1, 1.0, 0.5)
            
            with col3:
                include_purchased = st.checkbox("Include previously purchased items", False)
        
        # Generate recommendations button with enhanced styling
        if st.button("🚀 Generate AI Recommendations", use_container_width=True):
            with st.spinner("🤖 AI is analyzing customer preferences..."):
                try:
                    create_progress_animation()
                    
                    # Generate recommendations based on method
                    if "Collaborative" in rec_method:
                        recommendations = collaborative_filtering(df, selected_customer, n_recommendations)
                        method_description = """
                        🤝 **Collaborative Filtering**: Our AI analyzes purchasing patterns of similar customers 
                        to recommend products that customers with similar tastes have enjoyed.
                        """
                    elif "Content-Based" in rec_method:
                        recommendations = content_based_recommendation(df, selected_customer, n_recommendations)
                        method_description = """
                        📊 **Content-Based Filtering**: Our AI studies this customer's purchase history 
                        to recommend similar products based on their personal preferences.
                        """
                    else:  # Hybrid
                        recommendations = hybrid_recommendation(df, selected_customer, n_recommendations)
                        method_description = """
                        🎯 **Hybrid AI**: Our most advanced approach combines multiple AI techniques 
                        for the most accurate and diverse recommendations.
                        """
                    
                    create_info_box(method_description, "info")
                    
                    if recommendations:
                        st.markdown("### 🎁 Personalized Recommendations")
                        
                        # Create recommendation cards
                        for i in range(0, len(recommendations), 3):
                            cols = st.columns(3)
                            for j, col in enumerate(cols):
                                if i + j < len(recommendations):
                                    product_id = recommendations[i + j]
                                    product_data = df[df['ProductID'] == product_id].iloc[0]
                                    
                                    with col:
                                        st.markdown(f"""
                                        <div class="card">
                                            <h4 style="color: #667eea; margin-bottom: 1rem;">🎯 {product_data['Description']}</h4>
                                            <p><strong>Category:</strong> {product_data['Category']}</p>
                                            <p><strong>Price:</strong> <span style="color: #00b894; font-size: 1.2rem; font-weight: bold;">${product_data['UnitPrice']:.2f}</span></p>
                                            <p><strong>Product ID:</strong> {product_id}</p>
                                            <div style="margin-top: 1rem;">
                                                <span style="background: linear-gradient(135deg, #00b894, #00cec9); color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem;">
                                                    ⭐ AI Recommended
                                                </span>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                        
                        # Purchase history analysis
                        st.markdown("### 📚 Customer Purchase History")
                        
                        past_purchases = df[df['CustomerID'] == selected_customer]
                        past_products = past_purchases.groupby('ProductID').agg({
                            'Description': 'first',
                            'Category': 'first',
                            'Quantity': 'sum',
                            'TotalAmount': 'sum',
                            'InvoiceDate': 'max'
                        }).reset_index()
                        
                        past_products = past_products.sort_values('InvoiceDate', ascending=False)
                        
                        # Interactive purchase history table
                        st.dataframe(
                            past_products[['ProductID', 'Description', 'Category', 'Quantity', 'TotalAmount']],
                            use_container_width=True,
                            height=300
                        )
                        
                        # Purchase pattern visualization
                        if len(past_products) > 0:
                            fig_history = px.bar(
                                past_products.head(10),
                                x='Description',
                                y='TotalAmount',
                                title='🛒 Top Purchases by Value',
                                color='TotalAmount',
                                color_continuous_scale='Blues'
                            )
                            fig_history.update_xaxes(tickangle=45)
                            fig_history.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#2C3E50')
                            )
                            st.plotly_chart(fig_history, use_container_width=True)
                    
                    else:
                        create_info_box("❌ No recommendations found for this customer. Try a different method or customer.", "warning")
                
                except Exception as e:
                    st.error(f"❌ Error generating recommendations: {str(e)}")
        
        # Educational section about recommendation systems
        with st.expander("🎓 Learn About AI Recommendation Systems"):
            tab_edu1, tab_edu2, tab_edu3 = st.tabs(["🤝 Collaborative", "📊 Content-Based", "🎯 Hybrid"])
            
            with tab_edu1:
                st.markdown("""
                ### 🤝 Collaborative Filtering
                
                **How it works:**
                - Analyzes purchase patterns of similar customers
                - Finds customers with similar tastes and preferences
                - Recommends products that similar customers have purchased
                
                **Best for:**
                - Discovering new products outside usual preferences
                - Leveraging community wisdom
                - Finding trending items among similar users
                
                **Example:** "Customers who bought A and B also bought C"
                """)
            
            with tab_edu2:
                st.markdown("""
                ### 📊 Content-Based Filtering
                
                **How it works:**
                - Analyzes the customer's own purchase history
                - Identifies patterns in product categories and attributes
                - Recommends similar products based on past preferences
                
                **Best for:**
                - Consistent with personal taste
                - Building on established preferences
                - Avoiding recommendation surprises
                
                **Example:** "Since you bought sports equipment, here are more sports items"
                """)
            
            with tab_edu3:
                st.markdown("""
                ### 🎯 Hybrid AI Approach
                
                **How it works:**
                - Combines multiple recommendation techniques
                - Balances personal preferences with community insights
                - Uses advanced algorithms to weight different factors
                
                **Best for:**
                - Most accurate recommendations
                - Balanced discovery and personalization
                - Overcoming individual method limitations
                
                **Example:** Advanced AI considers both personal history AND similar customer behavior
                """)
    
    # Enhanced Churn Prediction Tab
    with tab3:
        st.markdown('<h2 class="sub-header">⚠️ Customer Churn Prediction & Retention</h2>', unsafe_allow_html=True)
        
        # Compute churn analysis
        if 'churn_features' not in st.session_state or 'churn_model' not in st.session_state:
            with st.spinner("🔮 Analyzing customer churn risk..."):
                try:
                    create_progress_animation()
                    churn_features, churn_model, churn_scaler, feature_importance, churn_predictions = compute_churn_analysis(df)
                    st.session_state.churn_features = churn_features
                    st.session_state.churn_model = churn_model
                    st.session_state.churn_scaler = churn_scaler
                    st.session_state.feature_importance = feature_importance
                    st.session_state.churn_predictions = churn_predictions
                    
                    create_info_box("✅ Churn analysis completed successfully!", "success")
                except Exception as e:
                    st.error(f"❌ Error computing churn analysis: {str(e)}")
                    return
        
        try:
            churn_features = st.session_state.get('churn_features')
            churn_model = st.session_state.get('churn_model')
            churn_scaler = st.session_state.get('churn_scaler')
            feature_importance = st.session_state.get('feature_importance')
            churn_predictions = st.session_state.get('churn_predictions')
            
            if any(x is None for x in [churn_features, churn_model, churn_scaler, feature_importance, churn_predictions]):
                st.error("❌ Churn analysis data is not available.")
                return
            
            # Key churn metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_churn_risk = churn_features['ChurnRisk'].mean() * 100
                create_animated_metric_card("Avg Churn Risk", f"{avg_churn_risk:.1f}%", "📊 Overall risk level")
            
            with col2:
                high_risk = (churn_predictions['ChurnProbability'] > 0.7).sum()
                create_animated_metric_card("High Risk", f"{high_risk}", "🔴 Immediate attention")
            
            with col3:
                medium_risk = ((churn_predictions['ChurnProbability'] > 0.3) & 
                              (churn_predictions['ChurnProbability'] <= 0.7)).sum()
                create_animated_metric_card("Medium Risk", f"{medium_risk}", "🟡 Monitor closely")
            
            with col4:
                low_risk = (churn_predictions['ChurnProbability'] <= 0.3).sum()
                create_animated_metric_card("Low Risk", f"{low_risk}", "🟢 Stable customers")
            
            # Interactive churn analysis
            st.markdown("### 📊 Churn Risk Distribution")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Enhanced histogram with risk zones
                fig_churn_dist = go.Figure()
                
                fig_churn_dist.add_trace(go.Histogram(
                    x=churn_predictions['ChurnProbability'],
                    nbinsx=30,
                    name='Churn Probability',
                    marker_color='rgba(102, 126, 234, 0.7)',
                    hovertemplate='Probability: %{x:.2f}<br>Count: %{y}<extra></extra>'
                ))
                
                # Add risk zone indicators
                fig_churn_dist.add_vline(x=0.3, line_dash="dash", line_color="orange", 
                                        annotation_text="Medium Risk Threshold")
                fig_churn_dist.add_vline(x=0.7, line_dash="dash", line_color="red", 
                                        annotation_text="High Risk Threshold")
                
                fig_churn_dist.update_layout(
                    title='🎯 Customer Churn Probability Distribution',
                    xaxis_title="Churn Probability",
                    yaxis_title="Number of Customers",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2C3E50')
                )
                
                st.plotly_chart(fig_churn_dist, use_container_width=True)
            
            with col2:
                # Risk level pie chart
                risk_levels = []
                for prob in churn_predictions['ChurnProbability']:
                    if prob > 0.7:
                        risk_levels.append('🔴 High Risk')
                    elif prob > 0.3:
                        risk_levels.append('🟡 Medium Risk')
                    else:
                        risk_levels.append('🟢 Low Risk')
                
                risk_counts = pd.Series(risk_levels).value_counts()
                
                fig_risk_pie = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title='🍰 Risk Level Distribution',
                    color_discrete_map={
                        '🔴 High Risk': '#e74c3c',
                        '🟡 Medium Risk': '#f39c12',
                        '🟢 Low Risk': '#27ae60'
                    }
                )
                st.plotly_chart(fig_risk_pie, use_container_width=True)
            
            # Feature importance analysis
            st.markdown("### 🎯 Key Churn Factors")
            
            fig_features = px.bar(
                feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title='🔍 AI-Identified Churn Factors',
                color='Importance',
                color_continuous_scale='Reds',
                text='Importance'
            )
            
            fig_features.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_features.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2C3E50'),
                height=400
            )
            
            st.plotly_chart(fig_features, use_container_width=True)
            
            # At-risk customer management
            st.markdown("### 🚨 Customer Risk Management")
            
            # Risk level filter
            col1, col2 = st.columns([1, 3])
            
            with col1:
                risk_filter = st.selectbox(
                    "Filter by Risk Level:",
                    ["All Customers", "High Risk (>70%)", "Medium Risk (30-70%)", "Low Risk (<30%)"]
                )
                
                show_top_n = st.slider("Show top N customers:", 5, 50, 20)
            
            with col2:
                # Filter data based on selection
                if risk_filter == "High Risk (>70%)":
                    filtered_predictions = churn_predictions[churn_predictions['ChurnProbability'] > 0.7]
                elif risk_filter == "Medium Risk (30-70%)":
                    filtered_predictions = churn_predictions[
                        (churn_predictions['ChurnProbability'] > 0.3) & 
                        (churn_predictions['ChurnProbability'] <= 0.7)
                    ]
                elif risk_filter == "Low Risk (<30%)":
                    filtered_predictions = churn_predictions[churn_predictions['ChurnProbability'] <= 0.3]
                else:
                    filtered_predictions = churn_predictions
                
                # Show filtered results
                display_data = filtered_predictions.sort_values('ChurnProbability', ascending=False).head(show_top_n)
                
                # Add risk level column for display
                display_data = display_data.copy()
                display_data['Risk_Level'] = display_data['ChurnProbability'].apply(
                    lambda x: '🔴 High' if x > 0.7 else ('🟡 Medium' if x > 0.3 else '🟢 Low')
                )
                
                st.dataframe(
                    display_data[['Risk_Level', 'ChurnProbability', 'DaysSinceLastPurchase', 
                                'NumberOfOrders', 'TotalSpent', 'AvgOrderValue']],
                    use_container_width=True,
                    height=400
                )
            
            # Individual customer analysis
            st.markdown("### 🔍 Individual Customer Analysis")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                search_customer_churn = st.text_input(
                    "🔎 Analyze Customer:",
                    placeholder="Enter customer ID for detailed churn analysis...",
                    key="churn_search"
                )
            
            with col2:
                analyze_button = st.button("🚀 Generate Retention Strategy", use_container_width=True)
            
            if analyze_button and search_customer_churn and search_customer_churn in churn_predictions.index:
                customer_data = churn_predictions.loc[search_customer_churn]
                
                # Risk assessment
                risk_prob = customer_data['ChurnProbability']
                if risk_prob > 0.7:
                    risk_level = "🔴 High Risk"
                    risk_color = "#e74c3c"
                elif risk_prob > 0.3:
                    risk_level = "🟡 Medium Risk"
                    risk_color = "#f39c12"
                else:
                    risk_level = "🟢 Low Risk"
                    risk_color = "#27ae60"
                
                # Customer profile
                st.markdown("#### 📋 Customer Risk Profile")
                
                profile_col1, profile_col2, profile_col3 = st.columns(3)
                
                with profile_col1:
                    st.markdown(f"""
                    <div style="background: {risk_color}; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                        <h3>{risk_level}</h3>
                        <p>Probability: {risk_prob:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with profile_col2:
                    create_animated_metric_card(
                        "Days Inactive", 
                        f"{customer_data['DaysSinceLastPurchase']:.0f}",
                        "Since last purchase"
                    )
                
                with profile_col3:
                    create_animated_metric_card(
                        "Total Value", 
                        f"${customer_data['TotalSpent']:.2f}",
                        "Customer lifetime value"
                    )
                
                # Detailed metrics
                detail_col1, detail_col2, detail_col3 = st.columns(3)
                
                with detail_col1:
                    st.metric("📦 Total Orders", f"{customer_data['NumberOfOrders']:.0f}")
                
                with detail_col2:
                    st.metric("💰 Avg Order Value", f"${customer_data['AvgOrderValue']:.2f}")
                
                with detail_col3:
                    recency_score = customer_data.get('Recency', 'N/A')
                    st.metric("📅 Recency Score", f"{recency_score}")
                
                # AI-generated retention strategy
                st.markdown("#### 🎯 AI-Generated Retention Strategy")
                
                churn_factors = identify_churn_factors(search_customer_churn, churn_predictions, feature_importance)
                
                strategy_col1, strategy_col2 = st.columns(2)
                
                with strategy_col1:
                    st.markdown("**🚨 Key Risk Factors:**")
                    for factor in churn_factors[:3]:  # Show top 3 factors
                        st.write(f"• {factor}")
                
                with strategy_col2:
                    st.markdown("**💡 Recommended Actions:**")
                    
                    if risk_prob > 0.7:
                        st.write("🎯 **Immediate Action Required:**")
                        if customer_data['TotalSpent'] > churn_predictions['TotalSpent'].median():
                            st.write("• Personal call from account manager")
                            st.write("• Exclusive VIP discount (15-20%)")
                            st.write("• Priority customer service")
                        else:
                            st.write("• Automated re-engagement email")
                            st.write("• Limited-time discount offer")
                            st.write("• Product recommendation based on history")
                    
                    elif risk_prob > 0.3:
                        st.write("🔔 **Proactive Engagement:**")
                        st.write("• Newsletter with personalized content")
                        st.write("• Loyalty program invitation")
                        st.write("• Survey for feedback and preferences")
                    
                    else:
                        st.write("✅ **Maintain Engagement:**")
                        st.write("• Regular promotional emails")
                        st.write("• New product announcements")
                        st.write("• Referral program invitation")
            
            elif analyze_button and search_customer_churn:
                create_info_box("❌ Customer not found. Please check the customer ID.", "warning")
        
        except Exception as e:
            st.error(f"❌ Error displaying churn analysis: {str(e)}")

if __name__ == "__main__":
    try:
        # Initialize session state
        if not hasattr(st.session_state, 'initialized'):
            st.session_state.update({
                'initialized': True,
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
        
        create_streamlit_app()
    except Exception as e:
        st.error(f"💥 An unexpected error occurred: {str(e)}")
        import logging
        logging.error(f"Dashboard error: {str(e)}", exc_info=True)