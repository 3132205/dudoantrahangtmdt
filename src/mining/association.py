"""
Association Rule Mining Module
===============================
Module khai phá luật kết hợp từ dữ liệu giao dịch.
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.frequent_patterns import association_rules
from typing import Optional, Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class AssociationMiner:
    """
    Class khai phá luật kết hợp.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Khởi tạo AssociationMiner.
        
        Parameters:
        -----------
        config : dict, optional
            Cấu hình cho association mining
        """
        self.config = config or {}
        self.frequent_itemsets = None
        self.rules = None
        self.mining_log = []
    
    def prepare_basket(self, df: pd.DataFrame,
                       transaction_id_col: str = 'order_id',
                       item_cols: List[str] = None) -> pd.DataFrame:
        """
        Chuẩn bị dữ liệu dạng basket cho association mining.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame chứa dữ liệu
        transaction_id_col : str
            Tên cột transaction ID
        item_cols : list
            Danh sách các cột item để tạo basket
            
        Returns:
        --------
        pd.DataFrame
            Basket format data
        """
        if item_cols is None:
            item_cols = ['product_category', 'payment_method', 'shipping_type']
        
        # Tạo transaction ID nếu chưa có
        if transaction_id_col not in df.columns:
            df = df.copy()
            df['transaction_id'] = range(len(df))
            transaction_id_col = 'transaction_id'
        
        # Tạo basket
        basket = pd.DataFrame()
        basket['transaction_id'] = df[transaction_id_col]
        
        for col in item_cols:
            if col in df.columns:
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col)
                basket = pd.concat([basket, dummies], axis=1)
        
        # Group by transaction_id
        basket = basket.groupby('transaction_id').max().reset_index(drop=True)
        
        self.mining_log.append(f"Created basket with {basket.shape[1]} items from {len(df)} transactions")
        
        return basket
    
    def find_frequent_itemsets(self, basket: pd.DataFrame,
                               method: str = 'apriori',
                               min_support: float = 0.01,
                               max_len: int = 4,
                               use_colnames: bool = True) -> pd.DataFrame:
        """
        Tìm frequent itemsets.
        
        Parameters:
        -----------
        basket : pd.DataFrame
            Basket format data
        method : str
            Phương pháp ('apriori', 'fpgrowth')
        min_support : float
            Ngưỡng support tối thiểu
        max_len : int
            Độ dài tối đa của itemset
        use_colnames : bool
            Sử dụng tên cột thay vì index
            
        Returns:
        --------
        pd.DataFrame
            Frequent itemsets
        """
        if method == 'apriori':
            self.frequent_itemsets = apriori(
                basket, 
                min_support=min_support, 
                use_colnames=use_colnames, 
                max_len=max_len,
                verbose=1
            )
        elif method == 'fpgrowth':
            self.frequent_itemsets = fpgrowth(
                basket, 
                min_support=min_support, 
                use_colnames=use_colnames, 
                max_len=max_len
            )
        else:
            raise ValueError(f"Method {method} not supported")
        
        # Thêm cột length
        self.frequent_itemsets['length'] = self.frequent_itemsets['itemsets'].apply(len)
        
        self.mining_log.append(f"Found {len(self.frequent_itemsets)} frequent itemsets with min_support={min_support}")
        
        return self.frequent_itemsets
    
    def generate_rules(self, 
                       metric: str = 'confidence',
                       min_threshold: float = 0.5,
                       min_lift: float = 1.0) -> pd.DataFrame:
        """
        Tạo luật kết hợp từ frequent itemsets.
        
        Parameters:
        -----------
        metric : str
            Metric để đánh giá ('confidence', 'lift', 'support')
        min_threshold : float
            Ngưỡng tối thiểu cho metric
        min_lift : float
            Ngưỡng lift tối thiểu
            
        Returns:
        --------
        pd.DataFrame
            Association rules
        """
        if self.frequent_itemsets is None:
            raise ValueError("No frequent itemsets found. Run find_frequent_itemsets first.")
        
        self.rules = association_rules(
            self.frequent_itemsets,
            metric=metric,
            min_threshold=min_threshold
        )
        
        # Filter by lift
        if min_lift > 1:
            self.rules = self.rules[self.rules['lift'] >= min_lift]
        
        # Add additional metrics
        self.rules['antecedent_len'] = self.rules['antecedents'].apply(len)
        self.rules['consequent_len'] = self.rules['consequents'].apply(len)
        self.rules['rule_length'] = self.rules['antecedent_len'] + self.rules['consequent_len']
        
        self.mining_log.append(f"Generated {len(self.rules)} rules with {metric}>={min_threshold}, lift>={min_lift}")
        
        return self.rules
    
    def filter_rules_by_consequent(self, consequent_pattern: str) -> pd.DataFrame:
        """
        Lọc các luật có consequent chứa pattern.
        
        Parameters:
        -----------
        consequent_pattern : str
            Pattern cần tìm trong consequent
            
        Returns:
        --------
        pd.DataFrame
            Các luật phù hợp
        """
        if self.rules is None:
            raise ValueError("No rules generated. Run generate_rules first.")
        
        filtered = self.rules[
            self.rules['consequents'].apply(
                lambda x: any(consequent_pattern in str(item) for item in x)
            )
        ].copy()
        
        self.mining_log.append(f"Filtered {len(filtered)} rules with consequent containing '{consequent_pattern}'")
        
        return filtered
    
    def get_top_rules(self, n: int = 10, by: str = 'lift') -> pd.DataFrame:
        """
        Lấy top N luật theo metric.
        
        Parameters:
        -----------
        n : int
            Số lượng luật cần lấy
        by : str
            Metric để xếp hạng ('lift', 'confidence', 'support')
            
        Returns:
        --------
        pd.DataFrame
            Top N rules
        """
        if self.rules is None:
            raise ValueError("No rules generated. Run generate_rules first.")
        
        return self.rules.nlargest(n, by)[
            ['antecedents', 'consequents', 'support', 'confidence', 'lift']
        ]
    
    def discretize_numerical(self, df: pd.DataFrame,
                             column: str,
                             bins: List[float],
                             labels: List[str]) -> pd.DataFrame:
        """
        Rời rạc hóa biến số.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame chứa dữ liệu
        column : str
            Tên cột cần rời rạc hóa
        bins : list
            Danh sách các bins
        labels : list
            Nhãn cho các bins
            
        Returns:
        --------
        pd.DataFrame
            DataFrame với cột đã được rời rạc hóa
        """
        df_disc = df.copy()
        new_col = f"{column}_bin"
        
        df_disc[new_col] = pd.cut(
            df_disc[column],
            bins=bins,
            labels=labels,
            right=False
        )
        
        self.mining_log.append(f"Discretized {column} into {new_col} with bins: {bins}")
        
        return df_disc
    
    def get_rule_stats(self) -> Dict:
        """
        Lấy thống kê về các luật.
        
        Returns:
        --------
        dict
            Thống kê các luật
        """
        if self.rules is None:
            return {}
        
        stats = {
            'total_rules': len(self.rules),
            'avg_support': self.rules['support'].mean(),
            'avg_confidence': self.rules['confidence'].mean(),
            'avg_lift': self.rules['lift'].mean(),
            'min_lift': self.rules['lift'].min(),
            'max_lift': self.rules['lift'].max(),
            'rules_by_length': self.rules['rule_length'].value_counts().to_dict()
        }
        
        return stats
    
    def get_mining_log(self) -> List[str]:
        """
        Lấy log của quá trình mining.
        
        Returns:
        --------
        list
            Danh sách các bước đã thực hiện
        """
        return self.mining_log