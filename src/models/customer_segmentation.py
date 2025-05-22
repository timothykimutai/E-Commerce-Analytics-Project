import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import joblib


# -------------------------------
# RFM Analysis
# -------------------------------
def perform_rfm_analysis(df, analysis_date=None):
    required_columns = ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'TotalAmount']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for RFM analysis: {', '.join(missing)}")

    print("[INFO] DataFrame Columns:", df.columns.tolist())
    print("[INFO] Sample Data:\n", df.head())

    # Set analysis date
    if analysis_date is None:
        analysis_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    # Calculate RFM metrics
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (analysis_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',                                   # Frequency
        'TotalAmount': 'sum'                                      # Monetary
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm['Recency'] = rfm['Recency'].clip(lower=0)

    # RFM Scoring
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=range(5, 0, -1)).astype(int)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=range(1, 6)).astype(int)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, labels=range(1, 6)).astype(int)
    rfm['RFM_Score'] = rfm['R_Score'] * 100 + rfm['F_Score'] * 10 + rfm['M_Score']

    # Segment customers
    def segment_customer(r, f, m):
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3 and m >= 3:
            return 'Loyal Customers'
        elif r >= 3 and f >= 1 and m >= 2:
            return 'Potential Loyalists'
        elif r >= 4 and f <= 1 and m <= 2:
            return 'New Customers'
        elif r >= 3 and f <= 2 and m <= 2:
            return 'Promising'
        elif r <= 2 and f >= 3 and m >= 3:
            return 'Needs Attention'
        elif r <= 2 and f >= 2 and m >= 2:
            return 'At Risk'
        elif r <= 1 and f >= 4 and m >= 4:
            return 'Cannot Lose Them'
        elif r <= 2 and f <= 2 and m <= 3:
            return 'About to Sleep'
        elif r <= 1 and f <= 1 and m >= 3:
            return 'Hibernating'
        else:
            return 'Lost'

    rfm['RFM_Segment'] = rfm.apply(lambda row: segment_customer(row['R_Score'], row['F_Score'], row['M_Score']), axis=1)

    print("[INFO] RFM Analysis Complete. Sample:\n", rfm.head())
    return rfm


# -------------------------------
# Customer Clustering
# -------------------------------
def perform_customer_clustering(rfm_df, method='kmeans', n_clusters=5):
    df_clustering = rfm_df.reset_index().copy()  # Restore CustomerID as column

    features = ['Recency', 'Frequency', 'Monetary']
    X = df_clustering[features].values

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply clustering
    if method == 'kmeans':
        if n_clusters is None:
            silhouette_scores = []
            for k in range(2, 11):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                score = silhouette_score(X_scaled, kmeans.fit_predict(X_scaled))
                silhouette_scores.append(score)
            n_clusters = np.argmax(silhouette_scores) + 2

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_clustering['Cluster'] = kmeans.fit_predict(X_scaled)

    elif method == 'dbscan':
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        df_clustering['Cluster'] = dbscan.fit_predict(X_scaled)
        n_outliers = (df_clustering['Cluster'] == -1).sum()
        print(f"[INFO] DBSCAN identified {n_outliers} outliers")

    # Save scaler
    joblib.dump(scaler, 'customer_scaler.pkl')

    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df_clustering['PCA1'] = pca_result[:, 0]
    df_clustering['PCA2'] = pca_result[:, 1]

    # Cluster summary
    cluster_analysis = df_clustering.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'CustomerID': 'count'
    }).rename(columns={'CustomerID': 'Count'})

    print("[INFO] Cluster Summary:\n", cluster_analysis)

    # Name clusters
    def name_cluster(cluster_id, summary):
        row = summary.loc[cluster_id]
        r, f, m = row['Recency'], row['Frequency'], row['Monetary']
        r_median = summary['Recency'].median()
        f_median = summary['Frequency'].median()
        m_median = summary['Monetary'].median()

        if r < r_median and f > f_median and m > m_median:
            return "VIP Customers"
        elif r < r_median and f > f_median:
            return "Loyal Customers"
        elif r < r_median and m > m_median:
            return "Big Spenders"
        elif r < r_median:
            return "New Customers"
        elif f > f_median and m > m_median:
            return "At-Risk High Value"
        elif f > f_median:
            return "Regular Customers"
        elif m > m_median:
            return "Dormant High Value"
        else:
            return "Churned Customers"

    cluster_names = {cid: name_cluster(cid, cluster_analysis) for cid in cluster_analysis.index}
    df_clustering['Cluster_Name'] = df_clustering['Cluster'].map(cluster_names)

    return df_clustering, cluster_analysis, cluster_names
