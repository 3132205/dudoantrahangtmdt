"""
Clustering Module
==================
Module phân cụm khách hàng và sản phẩm.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from typing import Optional, Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class ClusterAnalyzer:
    """
    Class phân tích và phân cụm dữ liệu.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Khởi tạo ClusterAnalyzer.
        
        Parameters:
        -----------
        config : dict, optional
            Cấu hình cho clustering
        """
        self.config = config or {}
        self.models = {}
        self.cluster_labels = None
        self.cluster_profiles = None
        self.clustering_log = []
    
    def prepare_data(self, df: pd.DataFrame,
                     features: List[str],
                     scaling: str = 'standard') -> np.ndarray:
        """
        Chuẩn bị dữ liệu cho clustering.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame chứa dữ liệu
        features : list
            Danh sách features dùng để clustering
        scaling : str
            Phương pháp scaling ('standard', 'minmax', 'none')
            
        Returns:
        --------
        np.ndarray
            Dữ liệu đã được chuẩn bị
        """
        # Lấy features
        X = df[features].copy()
        
        # Xử lý missing
        X = X.fillna(X.mean())
        
        # Scaling
        if scaling == 'standard':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        elif scaling == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values
        
        self.clustering_log.append(f"Prepared data with {len(features)} features, scaling={scaling}")
        
        return X_scaled
    
    def find_optimal_k(self, X: np.ndarray,
                       k_range: List[int] = range(2, 11),
                       random_state: int = 42) -> Dict:
        """
        Tìm số cụm tối ưu bằng Elbow method và Silhouette.
        
        Parameters:
        -----------
        X : np.ndarray
            Dữ liệu đã chuẩn bị
        k_range : list
            Dải giá trị k cần thử
        random_state : int
            Random seed
            
        Returns:
        --------
        dict
            Kết quả tìm k tối ưu
        """
        inertias = []
        sil_scores = []
        calinski_scores = []
        davies_scores = []
        
        for k in k_range:
            # Thực hiện KMeans
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            
            if k >= 2:
                sil = silhouette_score(X, labels)
                sil_scores.append(sil)
                
                cal = calinski_harabasz_score(X, labels)
                calinski_scores.append(cal)
                
                dav = davies_bouldin_score(X, labels)
                davies_scores.append(dav)
            else:
                sil_scores.append(0)
                calinski_scores.append(0)
                davies_scores.append(0)
        
        # Tìm k tối ưu theo silhouette
        optimal_k_idx = np.argmax(sil_scores[1:]) + 1
        optimal_k = list(k_range)[optimal_k_idx]
        
        results = {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': sil_scores,
            'calinski_scores': calinski_scores,
            'davies_scores': davies_scores,
            'optimal_k': optimal_k
        }
        
        self.clustering_log.append(f"Found optimal k={optimal_k} by silhouette score")
        
        return results
    
    def kmeans_clustering(self, X: np.ndarray,
                          n_clusters: int,
                          random_state: int = 42,
                          **kwargs) -> np.ndarray:
        """
        Thực hiện K-Means clustering.
        
        Parameters:
        -----------
        X : np.ndarray
            Dữ liệu đã chuẩn bị
        n_clusters : int
            Số cụm
        random_state : int
            Random seed
        **kwargs : dict
            Tham số bổ sung cho KMeans
            
        Returns:
        --------
        np.ndarray
            Nhãn cụm
        """
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=kwargs.get('n_init', 10),
            max_iter=kwargs.get('max_iter', 300)
        )
        
        self.cluster_labels = model.fit_predict(X)
        self.models['kmeans'] = model
        
        self.clustering_log.append(f"Performed K-Means with {n_clusters} clusters")
        
        return self.cluster_labels
    
    def hierarchical_clustering(self, X: np.ndarray,
                                n_clusters: int,
                                linkage_method: str = 'ward') -> np.ndarray:
        """
        Thực hiện Hierarchical clustering.
        
        Parameters:
        -----------
        X : np.ndarray
            Dữ liệu đã chuẩn bị
        n_clusters : int
            Số cụm
        linkage_method : str
            Phương pháp linkage ('ward', 'complete', 'average')
            
        Returns:
        --------
        np.ndarray
            Nhãn cụm
        """
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method
        )
        
        self.cluster_labels = model.fit_predict(X)
        self.models['hierarchical'] = model
        
        self.clustering_log.append(f"Performed Hierarchical clustering with {n_clusters} clusters, linkage={linkage_method}")
        
        return self.cluster_labels
    
    def dbscan_clustering(self, X: np.ndarray,
                          eps: float = 0.5,
                          min_samples: int = 5) -> np.ndarray:
        """
        Thực hiện DBSCAN clustering.
        
        Parameters:
        -----------
        X : np.ndarray
            Dữ liệu đã chuẩn bị
        eps : float
            Maximum distance
        min_samples : int
            Minimum samples in neighborhood
            
        Returns:
        --------
        np.ndarray
            Nhãn cụm (-1 là noise)
        """
        model = DBSCAN(eps=eps, min_samples=min_samples)
        
        self.cluster_labels = model.fit_predict(X)
        self.models['dbscan'] = model
        
        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)
        
        self.clustering_log.append(f"Performed DBSCAN: {n_clusters} clusters, {n_noise} noise points")
        
        return self.cluster_labels
    
    def analyze_clusters(self, df: pd.DataFrame,
                         features: List[str],
                         labels: np.ndarray) -> pd.DataFrame:
        """
        Phân tích đặc điểm các cụm.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame gốc
        features : list
            Danh sách features đã dùng
        labels : np.ndarray
            Nhãn cụm
            
        Returns:
        --------
        pd.DataFrame
            Profile của các cụm
        """
        # Thêm nhãn vào dataframe
        df_with_labels = df.copy()
        df_with_labels['cluster'] = labels
        
        # Tính trung bình các features theo cụm
        cluster_profiles = df_with_labels.groupby('cluster')[features].mean()
        
        # Thêm thông tin về size
        cluster_profiles['count'] = df_with_labels.groupby('cluster').size()
        cluster_profiles['percentage'] = cluster_profiles['count'] / len(df_with_labels) * 100
        
        self.cluster_profiles = cluster_profiles
        
        self.clustering_log.append("Analyzed cluster profiles")
        
        return cluster_profiles
    
    def get_cluster_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Tính các metrics đánh giá clustering.
        
        Parameters:
        -----------
        X : np.ndarray
            Dữ liệu đã chuẩn bị
        labels : np.ndarray
            Nhãn cụm
            
        Returns:
        --------
        dict
            Các metrics
        """
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters < 2:
            return {
                'n_clusters': n_clusters,
                'silhouette_score': None,
                'calinski_harabasz_score': None,
                'davies_bouldin_score': None
            }
        
        metrics = {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_score(X, labels),
            'calinski_harabasz_score': calinski_harabasz_score(X, labels),
            'davies_bouldin_score': davies_bouldin_score(X, labels)
        }
        
        return metrics
    
    def name_clusters(self, profiles: pd.DataFrame,
                      features: List[str]) -> Dict:
        """
        Đặt tên cho các cụm dựa trên đặc điểm.
        
        Parameters:
        -----------
        profiles : pd.DataFrame
            Profile của các cụm
        features : list
            Danh sách features
            
        Returns:
        --------
        dict
            Tên và mô tả các cụm
        """
        cluster_names = {}
        
        for cluster in profiles.index:
            profile = profiles.loc[cluster]
            
            # Xác định đặc điểm nổi bật
            characteristics = []
            
            # Dựa trên quantiles để phân loại
            for feat in features:
                if feat in profile:
                    feat_values = profiles[feat].values
                    q25, q75 = np.percentile(feat_values, [25, 75])
                    
                    if profile[feat] > q75:
                        characteristics.append(f"high_{feat}")
                    elif profile[feat] < q25:
                        characteristics.append(f"low_{feat}")
            
            # Đặt tên
            if 'return_rate' in features:
                if profile.get('return_rate', 0) > profiles['return_rate'].quantile(0.75):
                    if profile.get('frequency', 0) > profiles['frequency'].quantile(0.75):
                        name = "High Risk VIP"
                    else:
                        name = "High Risk"
                elif profile.get('return_rate', 0) < profiles['return_rate'].quantile(0.25):
                    if profile.get('monetary_total', 0) > profiles['monetary_total'].quantile(0.75):
                        name = "Safe VIP"
                    else:
                        name = "Safe"
                elif profile.get('frequency', 0) > profiles['frequency'].quantile(0.75):
                    name = "Frequent Buyers"
                else:
                    name = f"Cluster {cluster}"
            else:
                name = f"Cluster {cluster}"
            
            cluster_names[cluster] = {
                'name': name,
                'size': int(profile['count']),
                'percentage': float(profile['percentage']),
                'characteristics': characteristics,
                'profile': profile.to_dict()
            }
        
        self.clustering_log.append(f"Named {len(cluster_names)} clusters")
        
        return cluster_names
    
    def get_clustering_log(self) -> List[str]:
        """
        Lấy log của quá trình clustering.
        
        Returns:
        --------
        list
            Danh sách các bước đã thực hiện
        """
        return self.clustering_log