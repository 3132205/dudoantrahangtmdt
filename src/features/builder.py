"""
Feature Builder Module
======================
Module chịu trách nhiệm tạo các đặc trưng mới từ dữ liệu:
- RFM features (Recency, Frequency, Monetary)
- Return rate features
- Time-based features
- Interaction features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Union, Any
import logging

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Class xây dựng các đặc trưng cho dự án.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Khởi tạo FeatureBuilder.
        
        Parameters:
        -----------
        config : dict, optional
            Cấu hình feature engineering
        """
        self.config = config or {}
        self.feature_log = []
        self.created_features = []
    
    def build_rfm_features(self, df: pd.DataFrame,
                           customer_id_col: str = 'customer_id',
                           date_col: str = 'order_date',
                           value_col: str = 'order_value',
                           target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Tạo RFM features cho khách hàng.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame chứa dữ liệu giao dịch
        customer_id_col : str
            Tên cột customer ID
        date_col : str
            Tên cột ngày
        value_col : str
            Tên cột giá trị đơn hàng
        target_col : str, optional
            Tên cột target (return flag)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame với RFM features đã được merge
        """
        df_rfm = df.copy()
        
        # Đảm bảo date_col là datetime
        if not pd.api.types.is_datetime64_any_dtype(df_rfm[date_col]):
            df_rfm[date_col] = pd.to_datetime(df_rfm[date_col])
        
        # Tính toán RFM cho mỗi khách hàng
        current_date = df_rfm[date_col].max()
        
        # Group by customer_id
        rfm_raw = df_rfm.groupby(customer_id_col).agg({
            date_col: lambda x: (current_date - x.max()).days,  # Recency
            'order_id': 'count',  # Frequency
            value_col: ['sum', 'mean'],  # Monetary
        })
        
        if target_col and target_col in df_rfm.columns:
            rfm_raw[target_col] = df_rfm.groupby(customer_id_col)[target_col].agg(['mean', 'sum'])
        
        # Flatten column names
        rfm_raw.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                          for col in rfm_raw.columns.values]
        rfm_raw = rfm_raw.reset_index()
        
        # Rename columns
        rename_dict = {
            f'{date_col}_<lambda>': 'recency_days',
            'order_id_count': 'frequency',
            f'{value_col}_sum': 'monetary_total',
            f'{value_col}_mean': 'avg_order_value'
        }
        
        if target_col and target_col in df_rfm.columns:
            rename_dict.update({
                f'{target_col}_mean': 'customer_return_rate',
                f'{target_col}_sum': 'total_returns'
            })
        
        rfm_raw.rename(columns=rename_dict, inplace=True)
        
        # Merge vào dataframe gốc
        result_df = df_rfm.merge(rfm_raw, on=customer_id_col, how='left')
        
        self.feature_log.append(f"Created RFM features: {list(rename_dict.values())}")
        self.created_features.extend(rename_dict.values())
        
        return result_df
    
    def build_return_rate_features(self, df: pd.DataFrame,
                                   customer_id_col: str = 'customer_id',
                                   product_id_col: str = 'product_id',
                                   category_col: str = 'product_category',
                                   target_col: str = 'return_flag',
                                   min_samples: int = 5) -> pd.DataFrame:
        """
        Tạo các đặc trưng về tỷ lệ trả hàng.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame chứa dữ liệu
        customer_id_col : str
            Tên cột customer ID
        product_id_col : str
            Tên cột product ID
        category_col : str
            Tên cột category
        target_col : str
            Tên cột target
        min_samples : int
            Số lượng mẫu tối thiểu để tính tỷ lệ
            
        Returns:
        --------
        pd.DataFrame
            DataFrame với return rate features
        """
        df_return = df.copy()
        
        # 1. Product return rate
        if product_id_col in df_return.columns:
            product_stats = df_return.groupby(product_id_col).agg({
                target_col: ['count', 'mean']
            }).round(4)
            product_stats.columns = ['product_orders', 'product_return_rate']
            product_stats = product_stats.reset_index()
            
            # Xử lý các sản phẩm có ít orders
            overall_return_rate = df_return[target_col].mean()
            product_stats.loc[product_stats['product_orders'] < min_samples, 'product_return_rate'] = overall_return_rate
            product_stats['product_return_rate'].fillna(overall_return_rate, inplace=True)
            
            df_return = df_return.merge(product_stats[[product_id_col, 'product_return_rate']], 
                                        on=product_id_col, how='left')
            
            self.feature_log.append("Added product_return_rate")
            self.created_features.append('product_return_rate')
        
        # 2. Category return rate
        if category_col in df_return.columns:
            category_stats = df_return.groupby(category_col).agg({
                target_col: ['count', 'mean']
            }).round(4)
            category_stats.columns = ['category_orders', 'category_return_rate']
            category_stats = category_stats.reset_index()
            
            overall_return_rate = df_return[target_col].mean()
            category_stats.loc[category_stats['category_orders'] < min_samples, 'category_return_rate'] = overall_return_rate
            category_stats['category_return_rate'].fillna(overall_return_rate, inplace=True)
            
            df_return = df_return.merge(category_stats[[category_col, 'category_return_rate']], 
                                        on=category_col, how='left')
            
            self.feature_log.append("Added category_return_rate")
            self.created_features.append('category_return_rate')
        
        # 3. Customer historical return rate
        if customer_id_col in df_return.columns and 'order_date' in df_return.columns:
            df_return = df_return.sort_values(['customer_id', 'order_date'])
            
            def calc_historical_returns(group):
                group = group.sort_values('order_date')
                historical_rates = []
                
                for i in range(len(group)):
                    if i == 0:
                        historical_rates.append(0)
                    else:
                        rate = group.iloc[:i][target_col].mean()
                        historical_rates.append(rate)
                
                group['customer_hist_return_rate'] = historical_rates
                return group
            
            df_return = df_return.groupby(customer_id_col).apply(calc_historical_returns).reset_index(drop=True)
            
            self.feature_log.append("Added customer_hist_return_rate")
            self.created_features.append('customer_hist_return_rate')
        
        return df_return
    
    def build_time_features(self, df: pd.DataFrame,
                            date_col: str = 'order_date',
                            features: List[str] = None) -> pd.DataFrame:
        """
        Tạo các đặc trưng thời gian.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame chứa dữ liệu
        date_col : str
            Tên cột ngày
        features : list
            Danh sách các features cần tạo
            
        Returns:
        --------
        pd.DataFrame
            DataFrame với time features
        """
        df_time = df.copy()
        
        if features is None:
            features = ['day_of_week', 'month', 'quarter', 'is_weekend', 'hour', 'season']
        
        # Đảm bảo date_col là datetime
        if not pd.api.types.is_datetime64_any_dtype(df_time[date_col]):
            df_time[date_col] = pd.to_datetime(df_time[date_col])
        
        if 'day_of_week' in features:
            df_time['day_of_week'] = df_time[date_col].dt.dayofweek
            df_time['day_name'] = df_time[date_col].dt.day_name()
            self.created_features.extend(['day_of_week', 'day_name'])
        
        if 'month' in features:
            df_time['month'] = df_time[date_col].dt.month
            self.created_features.append('month')
        
        if 'quarter' in features:
            df_time['quarter'] = df_time[date_col].dt.quarter
            self.created_features.append('quarter')
        
        if 'is_weekend' in features:
            df_time['is_weekend'] = (df_time[date_col].dt.dayofweek >= 5).astype(int)
            self.created_features.append('is_weekend')
        
        if 'hour' in features:
            df_time['hour_of_day'] = df_time[date_col].dt.hour
            self.created_features.append('hour_of_day')
        
        if 'season' in features:
            def get_season(month):
                if month in [3, 4, 5]:
                    return 'Spring'
                elif month in [6, 7, 8]:
                    return 'Summer'
                elif month in [9, 10, 11]:
                    return 'Fall'
                else:
                    return 'Winter'
            
            df_time['season'] = df_time[date_col].dt.month.map(get_season)
            self.created_features.append('season')
        
        if 'week_of_year' in features:
            df_time['week_of_year'] = df_time[date_col].dt.isocalendar().week
            self.created_features.append('week_of_year')
        
        self.feature_log.append(f"Added time features: {features}")
        
        return df_time
    
    def build_interaction_features(self, df: pd.DataFrame,
                                   features: List[str] = None) -> pd.DataFrame:
        """
        Tạo các interaction features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame chứa dữ liệu
        features : list
            Danh sách các interaction cần tạo
            
        Returns:
        --------
        pd.DataFrame
            DataFrame với interaction features
        """
        df_inter = df.copy()
        
        if features is None:
            features = ['value_per_item', 'discount_rate', 'shipping_ratio']
        
        # Value per item
        if 'value_per_item' in features and 'order_value' in df_inter.columns and 'quantity' in df_inter.columns:
            df_inter['value_per_item'] = df_inter['order_value'] / (df_inter['quantity'] + 1e-6)
            self.created_features.append('value_per_item')
            self.feature_log.append("Added value_per_item = order_value / quantity")
        
        # Discount rate
        if 'discount_rate' in features and 'discount_amount' in df_inter.columns and 'order_value' in df_inter.columns:
            df_inter['discount_rate'] = df_inter['discount_amount'] / (df_inter['order_value'] + 1e-6)
            self.created_features.append('discount_rate')
            self.feature_log.append("Added discount_rate = discount_amount / order_value")
        
        # Shipping ratio
        if 'shipping_ratio' in features and 'shipping_cost' in df_inter.columns and 'order_value' in df_inter.columns:
            df_inter['shipping_ratio'] = df_inter['shipping_cost'] / (df_inter['order_value'] + 1e-6)
            self.created_features.append('shipping_ratio')
            self.feature_log.append("Added shipping_ratio = shipping_cost / order_value")
        
        # Return rate trend
        if 'return_rate_trend' in features and 'previous_returns_count' in df_inter.columns and 'previous_orders_count' in df_inter.columns:
            df_inter['return_rate_trend'] = df_inter['previous_returns_count'] / (df_inter['previous_orders_count'] + 1)
            self.created_features.append('return_rate_trend')
            self.feature_log.append("Added return_rate_trend = previous_returns_count / previous_orders_count")
        
        return df_inter
    
    def build_holiday_features(self, df: pd.DataFrame,
                               date_col: str = 'order_date',
                               holidays: List[str] = None) -> pd.DataFrame:
        """
        Tạo các đặc trưng về ngày lễ.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame chứa dữ liệu
        date_col : str
            Tên cột ngày
        holidays : list
            Danh sách các ngày lễ (định dạng 'MM-DD')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame với holiday features
        """
        df_holiday = df.copy()
        
        if holidays is None:
            holidays = ['11-11', '11-29', '12-24', '12-25', '01-01']
        
        # Đảm bảo date_col là datetime
        if not pd.api.types.is_datetime64_any_dtype(df_holiday[date_col]):
            df_holiday[date_col] = pd.to_datetime(df_holiday[date_col])
        
        # Tạo cột is_holiday
        df_holiday['is_holiday'] = 0
        
        for holiday in holidays:
            month, day = map(int, holiday.split('-'))
            holiday_mask = (df_holiday[date_col].dt.month == month) & (df_holiday[date_col].dt.day == day)
            df_holiday.loc[holiday_mask, 'is_holiday'] = 1
        
        # Thêm holiday season (tuần lễ Black Friday, Christmas)
        black_friday_week = (df_holiday[date_col].dt.month == 11) & (df_holiday[date_col].dt.day >= 20) & (df_holiday[date_col].dt.day <= 30)
        df_holiday.loc[black_friday_week, 'holiday_season'] = 'BlackFriday'
        
        christmas_week = (df_holiday[date_col].dt.month == 12) & (df_holiday[date_col].dt.day >= 20) & (df_holiday[date_col].dt.day <= 26)
        df_holiday.loc[christmas_week, 'holiday_season'] = 'Christmas'
        
        new_year_week = (df_holiday[date_col].dt.month == 12) & (df_holiday[date_col].dt.day >= 30) | (df_holiday[date_col].dt.month == 1) & (df_holiday[date_col].dt.day <= 2)
        df_holiday.loc[new_year_week, 'holiday_season'] = 'NewYear'
        
        df_holiday['holiday_season'] = df_holiday['holiday_season'].fillna('Normal')
        
        self.created_features.extend(['is_holiday', 'holiday_season'])
        self.feature_log.append(f"Added holiday features with {len(holidays)} holidays")
        
        return df_holiday
    
    def build_lag_features(self, df: pd.DataFrame,
                           customer_id_col: str = 'customer_id',
                           date_col: str = 'order_date',
                           features: List[str] = None,
                           lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """
        Tạo lag features cho mỗi khách hàng.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame chứa dữ liệu
        customer_id_col : str
            Tên cột customer ID
        date_col : str
            Tên cột ngày
        features : list
            Danh sách features cần tạo lag
        lags : list
            Các lag cần tạo
            
        Returns:
        --------
        pd.DataFrame
            DataFrame với lag features
        """
        df_lag = df.copy()
        
        if features is None:
            features = ['order_value', 'discount_amount', 'shipping_cost']
        
        # Sắp xếp theo customer và ngày
        df_lag = df_lag.sort_values([customer_id_col, date_col])
        
        for col in features:
            if col in df_lag.columns:
                for lag in lags:
                    lag_col = f'{col}_lag_{lag}'
                    df_lag[lag_col] = df_lag.groupby(customer_id_col)[col].shift(lag)
                    self.created_features.append(lag_col)
                
                self.feature_log.append(f"Added lag features for {col} with lags {lags}")
        
        return df_lag
    
    def build_rolling_features(self, df: pd.DataFrame,
                               customer_id_col: str = 'customer_id',
                               date_col: str = 'order_date',
                               features: List[str] = None,
                               windows: List[int] = [7, 30, 90]) -> pd.DataFrame:
        """
        Tạo rolling window features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame chứa dữ liệu
        customer_id_col : str
            Tên cột customer ID
        date_col : str
            Tên cột ngày
        features : list
            Danh sách features cần tạo rolling
        windows : list
            Các window size (ngày)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame với rolling features
        """
        df_rolling = df.copy()
        
        if features is None:
            features = ['order_value', 'discount_amount']
        
        # Đảm bảo có cột ngày và sắp xếp
        if not pd.api.types.is_datetime64_any_dtype(df_rolling[date_col]):
            df_rolling[date_col] = pd.to_datetime(df_rolling[date_col])
        
        df_rolling = df_rolling.sort_values([customer_id_col, date_col])
        
        for col in features:
            if col in df_rolling.columns:
                for window in windows:
                    # Rolling mean
                    roll_mean_col = f'{col}_roll_{window}_mean'
                    df_rolling[roll_mean_col] = (
                        df_rolling.groupby(customer_id_col)[col]
                        .transform(lambda x: x.rolling(window, min_periods=1).mean())
                    )
                    self.created_features.append(roll_mean_col)
                    
                    # Rolling std
                    roll_std_col = f'{col}_roll_{window}_std'
                    df_rolling[roll_std_col] = (
                        df_rolling.groupby(customer_id_col)[col]
                        .transform(lambda x: x.rolling(window, min_periods=1).std())
                    )
                    self.created_features.append(roll_std_col)
                
                self.feature_log.append(f"Added rolling features for {col} with windows {windows}")
        
        return df_rolling
    
    def get_created_features(self) -> List[str]:
        """
        Lấy danh sách các features đã tạo.
        
        Returns:
        --------
        list
            Danh sách features
        """
        return list(set(self.created_features))
    
    def get_feature_log(self) -> List[str]:
        """
        Lấy log của quá trình tạo features.
        
        Returns:
        --------
        list
            Danh sách các bước đã thực hiện
        """
        return self.feature_log