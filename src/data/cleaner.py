"""
Data Cleaner Module
===================
Module chịu trách nhiệm làm sạch dữ liệu:
- Xử lý missing values
- Xử lý outliers
- Xử lý duplicates
- Mã hóa biến phân loại
- Chuẩn hóa dữ liệu
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Optional, Dict, List, Tuple, Union, Any
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Class xử lý làm sạch dữ liệu.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Khởi tạo DataCleaner.
        
        Parameters:
        -----------
        config : dict, optional
            Cấu hình xử lý dữ liệu
        """
        self.config = config or {}
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}
        self.cleaning_log = []
    
    def handle_missing(self, df: pd.DataFrame, 
                       num_strategy: str = 'median',
                       cat_strategy: str = 'mode',
                       threshold_drop: float = 50,
                       constant_fill: Any = 0) -> pd.DataFrame:
        """
        Xử lý missing values.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame cần xử lý
        num_strategy : str
            Chiến lược cho numerical columns ('mean', 'median', 'constant', 'drop')
        cat_strategy : str
            Chiến lược cho categorical columns ('mode', 'constant', 'drop')
        threshold_drop : float
            Ngưỡng % missing để drop column
        constant_fill : any
            Giá trị constant để fill
            
        Returns:
        --------
        pd.DataFrame
            DataFrame đã xử lý missing
        """
        df_clean = df.copy()
        
        # Log missing trước khi xử lý
        missing_before = df_clean.isnull().sum().sum()
        logger.info(f"Missing values before: {missing_before}")
        
        # Xác định các cột có missing > threshold
        missing_percent = df_clean.isnull().sum() / len(df_clean) * 100
        high_missing_cols = missing_percent[missing_percent > threshold_drop].index.tolist()
        
        if high_missing_cols:
            logger.info(f"Dropping columns with >{threshold_drop}% missing: {high_missing_cols}")
            df_clean = df_clean.drop(columns=high_missing_cols)
            self.cleaning_log.append(f"Dropped columns: {high_missing_cols}")
        
        # Xử lý numerical columns
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        if num_strategy != 'drop':
            for col in num_cols:
                if df_clean[col].isnull().any():
                    if num_strategy == 'mean':
                        fill_value = df_clean[col].mean()
                    elif num_strategy == 'median':
                        fill_value = df_clean[col].median()
                    elif num_strategy == 'constant':
                        fill_value = constant_fill
                    else:
                        continue
                    
                    df_clean[col].fillna(fill_value, inplace=True)
                    self.cleaning_log.append(f"Filled {col} missing with {num_strategy}: {fill_value:.2f}")
        
        # Xử lý categorical columns
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        
        if cat_strategy != 'drop':
            for col in cat_cols:
                if df_clean[col].isnull().any():
                    if cat_strategy == 'mode':
                        fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                    elif cat_strategy == 'constant':
                        fill_value = 'Unknown'
                    else:
                        continue
                    
                    df_clean[col].fillna(fill_value, inplace=True)
                    self.cleaning_log.append(f"Filled {col} missing with {cat_strategy}: {fill_value}")
        
        # Log missing sau khi xử lý
        missing_after = df_clean.isnull().sum().sum()
        logger.info(f"Missing values after: {missing_after}")
        
        return df_clean
    
    def handle_outliers(self, df: pd.DataFrame,
                        method: str = 'iqr',
                        treatment: str = 'cap',
                        iqr_multiplier: float = 1.5,
                        zscore_threshold: float = 3) -> pd.DataFrame:
        """
        Xử lý outliers.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame cần xử lý
        method : str
            Phương pháp phát hiện outlier ('iqr', 'zscore')
        treatment : str
            Cách xử lý ('cap', 'remove')
        iqr_multiplier : float
            Hệ số nhân cho IQR
        zscore_threshold : float
            Ngưỡng Z-score
            
        Returns:
        --------
        pd.DataFrame
            DataFrame đã xử lý outliers
        """
        df_clean = df.copy()
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        outlier_stats = []
        
        for col in num_cols:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR
                
                outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
                n_outliers = len(outliers)
                
            elif method == 'zscore':
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                outliers = df_clean[z_scores > zscore_threshold]
                n_outliers = len(outliers)
                lower_bound, upper_bound = None, None
            
            else:
                raise ValueError(f"Method {method} not supported")
            
            outlier_pct = n_outliers / len(df_clean) * 100
            
            if n_outliers > 0:
                if treatment == 'cap' and method == 'iqr':
                    df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                    self.cleaning_log.append(f"Capped {col} outliers to [{lower_bound:.2f}, {upper_bound:.2f}]")
                    
                elif treatment == 'remove':
                    df_clean = df_clean.drop(outliers.index)
                    self.cleaning_log.append(f"Removed {n_outliers} outliers from {col}")
                
                outlier_stats.append({
                    'column': col,
                    'n_outliers': n_outliers,
                    'outlier_pct': outlier_pct,
                    'treatment': treatment
                })
        
        logger.info(f"Processed outliers in {len(outlier_stats)} columns")
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame, 
                          subset: Optional[List[str]] = None,
                          keep: str = 'first') -> pd.DataFrame:
        """
        Xử lý dữ liệu trùng lặp.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame cần xử lý
        subset : list, optional
            Danh sách cột để xác định duplicate
        keep : str
            Giữ lại dòng nào ('first', 'last', False)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame đã xử lý duplicates
        """
        df_clean = df.copy()
        
        n_duplicates = df_clean.duplicated(subset=subset).sum()
        
        if n_duplicates > 0:
            df_clean = df_clean.drop_duplicates(subset=subset, keep=keep)
            self.cleaning_log.append(f"Removed {n_duplicates} duplicate rows")
            logger.info(f"Removed {n_duplicates} duplicate rows")
        
        return df_clean
    
    def encode_categorical(self, df: pd.DataFrame,
                           method: str = 'onehot',
                           columns: Optional[List[str]] = None,
                           drop_first: bool = True,
                           handle_unknown: str = 'ignore') -> pd.DataFrame:
        """
        Mã hóa biến phân loại.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame cần xử lý
        method : str
            Phương pháp mã hóa ('onehot', 'label', 'target')
        columns : list, optional
            Danh sách cột cần mã hóa
        drop_first : bool
            Có drop cột đầu tiên trong one-hot không
        handle_unknown : str
            Cách xử lý giá trị unknown
            
        Returns:
        --------
        pd.DataFrame
            DataFrame đã mã hóa
        """
        df_encoded = df.copy()
        
        if columns is None:
            columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if method == 'onehot':
            for col in columns:
                # Tạo one-hot encoding
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=drop_first, dummy_na=False)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded = df_encoded.drop(columns=[col])
                
                self.cleaning_log.append(f"One-hot encoded {col} -> {len(dummies.columns)} columns")
        
        elif method == 'label':
            for col in columns:
                le = LabelEncoder()
                df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = le
                df_encoded = df_encoded.drop(columns=[col])
                
                self.cleaning_log.append(f"Label encoded {col}")
        
        elif method == 'target':
            # Target encoding cần y target, sẽ xử lý riêng
            logger.warning("Target encoding should be done with target variable")
            pass
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame,
                       method: str = 'standard',
                       columns: Optional[List[str]] = None,
                       exclude_binary: bool = True) -> pd.DataFrame:
        """
        Chuẩn hóa features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame cần xử lý
        method : str
            Phương pháp scaling ('standard', 'minmax', 'robust')
        columns : list, optional
            Danh sách cột cần scale
        exclude_binary : bool
            Có loại trừ cột binary không
            
        Returns:
        --------
        pd.DataFrame
            DataFrame đã scale
        """
        df_scaled = df.copy()
        
        if columns is None:
            columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        
        if exclude_binary:
            # Loại trừ cột chỉ có 2 giá trị
            columns = [col for col in columns if df_scaled[col].nunique() > 2]
        
        if not columns:
            logger.warning("No columns to scale")
            return df_scaled
        
        # Chọn scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Method {method} not supported")
        
        # Fit và transform
        scaled_values = scaler.fit_transform(df_scaled[columns])
        
        # Tạo tên cột mới
        scaled_cols = [f"{col}_scaled" for col in columns]
        
        # Gán giá trị scaled
        for i, col in enumerate(scaled_cols):
            df_scaled[col] = scaled_values[:, i]
        
        self.scalers[method] = scaler
        self.cleaning_log.append(f"Scaled {len(columns)} columns with {method}")
        
        return df_scaled
    
    def handle_imbalance(self, X: pd.DataFrame, y: pd.Series,
                         method: str = 'smote',
                         sampling_strategy: str = 'auto',
                         random_state: int = 42):
        """
        Xử lý mất cân bằng lớp.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target vector
        method : str
            Phương pháp xử lý ('smote', 'random_over', 'random_under')
        sampling_strategy : str
            Chiến lược sampling
        random_state : int
            Random seed
            
        Returns:
        --------
        X_resampled, y_resampled
            Dữ liệu đã được cân bằng
        """
        from imblearn.over_sampling import SMOTE, RandomOverSampler
        from imblearn.under_sampling import RandomUnderSampler
        
        if method == 'smote':
            sampler = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
        elif method == 'random_over':
            sampler = RandomOverSampler(random_state=random_state, sampling_strategy=sampling_strategy)
        elif method == 'random_under':
            sampler = RandomUnderSampler(random_state=random_state, sampling_strategy=sampling_strategy)
        else:
            raise ValueError(f"Method {method} not supported")
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        self.cleaning_log.append(f"Applied {method} to handle imbalance")
        logger.info(f"Before: {y.value_counts().to_dict()}")
        logger.info(f"After: {pd.Series(y_resampled).value_counts().to_dict()}")
        
        return X_resampled, y_resampled
    
    def get_cleaning_log(self) -> List[str]:
        """
        Lấy log của quá trình cleaning.
        
        Returns:
        --------
        list
            Danh sách các bước đã thực hiện
        """
        return self.cleaning_log
    
    def save_encoders(self, path: str):
        """Lưu encoders"""
        import joblib
        joblib.dump(self.encoders, f"{path}/encoders.pkl")
        
    def save_scalers(self, path: str):
        """Lưu scalers"""
        import joblib
        joblib.dump(self.scalers, f"{path}/scalers.pkl")