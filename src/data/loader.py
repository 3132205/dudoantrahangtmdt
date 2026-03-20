"""
Data Loader Module
==================
Module chịu trách nhiệm đọc dữ liệu từ các nguồn khác nhau
và kiểm tra schema dữ liệu.
"""

import pandas as pd
import numpy as np
import yaml
import os
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Union
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Class chịu trách nhiệm load dữ liệu từ các nguồn khác nhau
    và kiểm tra tính hợp lệ của dữ liệu.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Khởi tạo DataLoader.
        
        Parameters:
        -----------
        config_path : str, optional
            Đường dẫn đến file cấu hình params.yaml
        """
        self.config = None
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Đã load cấu hình từ {config_path}")
        
        self.data = None
        self.data_info = {}
    
    def load_from_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Đọc dữ liệu từ file CSV.
        
        Parameters:
        -----------
        file_path : str
            Đường dẫn đến file CSV
        **kwargs : dict
            Các tham số bổ sung cho pd.read_csv
            
        Returns:
        --------
        pd.DataFrame
            DataFrame chứa dữ liệu
        """
        try:
            self.data = pd.read_csv(file_path, **kwargs)
            logger.info(f"Đã đọc {len(self.data)} dòng từ {file_path}")
            self._update_info('csv', file_path)
            return self.data
        except Exception as e:
            logger.error(f"Lỗi khi đọc file CSV: {e}")
            raise
    
    def load_from_parquet(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Đọc dữ liệu từ file Parquet.
        
        Parameters:
        -----------
        file_path : str
            Đường dẫn đến file Parquet
        **kwargs : dict
            Các tham số bổ sung cho pd.read_parquet
            
        Returns:
        --------
        pd.DataFrame
            DataFrame chứa dữ liệu
        """
        try:
            self.data = pd.read_parquet(file_path, **kwargs)
            logger.info(f"Đã đọc {len(self.data)} dòng từ {file_path}")
            self._update_info('parquet', file_path)
            return self.data
        except Exception as e:
            logger.error(f"Lỗi khi đọc file Parquet: {e}")
            raise
    
    def load_from_excel(self, file_path: str, sheet_name: str = 0, **kwargs) -> pd.DataFrame:
        """
        Đọc dữ liệu từ file Excel.
        
        Parameters:
        -----------
        file_path : str
            Đường dẫn đến file Excel
        sheet_name : str or int
            Tên hoặc index của sheet cần đọc
        **kwargs : dict
            Các tham số bổ sung cho pd.read_excel
            
        Returns:
        --------
        pd.DataFrame
            DataFrame chứa dữ liệu
        """
        try:
            self.data = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            logger.info(f"Đã đọc {len(self.data)} dòng từ {file_path}")
            self._update_info('excel', file_path)
            return self.data
        except Exception as e:
            logger.error(f"Lỗi khi đọc file Excel: {e}")
            raise
    
    def _update_info(self, source_type: str, source_path: str):
        """Cập nhật thông tin về dữ liệu"""
        self.data_info = {
            'source_type': source_type,
            'source_path': source_path,
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum() / 1024**2,  # MB
            'load_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def check_schema(self, expected_columns: Optional[List[str]] = None) -> Dict:
        """
        Kiểm tra schema của dữ liệu.
        
        Parameters:
        -----------
        expected_columns : list, optional
            Danh sách các cột mong đợi
            
        Returns:
        --------
        dict
            Kết quả kiểm tra schema
        """
        if self.data is None:
            raise ValueError("Chưa có dữ liệu. Hãy load dữ liệu trước.")
        
        schema_check = {
            'missing_columns': [],
            'extra_columns': [],
            'data_types': {},
            'null_counts': {},
            'unique_counts': {},
            'is_valid': True
        }
        
        # Kiểm tra các cột mong đợi
        if expected_columns:
            missing = set(expected_columns) - set(self.data.columns)
            extra = set(self.data.columns) - set(expected_columns)
            
            schema_check['missing_columns'] = list(missing)
            schema_check['extra_columns'] = list(extra)
            
            if missing:
                schema_check['is_valid'] = False
                logger.warning(f"Thiếu các cột: {missing}")
        
        # Thông tin chi tiết về từng cột
        for col in self.data.columns:
            schema_check['data_types'][col] = str(self.data[col].dtype)
            schema_check['null_counts'][col] = int(self.data[col].isnull().sum())
            schema_check['unique_counts'][col] = int(self.data[col].nunique())
        
        return schema_check
    
    def get_basic_stats(self) -> Dict:
        """
        Lấy thống kê cơ bản về dữ liệu.
        
        Returns:
        --------
        dict
            Thống kê cơ bản
        """
        if self.data is None:
            raise ValueError("Chưa có dữ liệu. Hãy load dữ liệu trước.")
        
        stats = {
            'total_rows': len(self.data),
            'total_columns': len(self.data.columns),
            'memory_usage_mb': self.data_info.get('memory_usage', 0),
            'missing_cells': int(self.data.isnull().sum().sum()),
            'missing_percentage': float(self.data.isnull().sum().sum() / self.data.size * 100),
            'duplicate_rows': int(self.data.duplicated().sum()),
            'duplicate_percentage': float(self.data.duplicated().sum() / len(self.data) * 100),
            'numeric_columns': len(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(self.data.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': len(self.data.select_dtypes(include=['datetime64']).columns)
        }
        
        return stats
    
    def get_sample_data(self, n: int = 5) -> pd.DataFrame:
        """
        Lấy mẫu dữ liệu.
        
        Parameters:
        -----------
        n : int
            Số lượng dòng mẫu
            
        Returns:
        --------
        pd.DataFrame
            DataFrame chứa mẫu dữ liệu
        """
        if self.data is None:
            raise ValueError("Chưa có dữ liệu. Hãy load dữ liệu trước.")
        
        return self.data.head(n)
    
    def save_data(self, file_path: str, format: str = 'parquet', **kwargs):
        """
        Lưu dữ liệu ra file.
        
        Parameters:
        -----------
        file_path : str
            Đường dẫn file đầu ra
        format : str
            Định dạng file ('csv', 'parquet', 'excel')
        **kwargs : dict
            Các tham số bổ sung
        """
        if self.data is None:
            raise ValueError("Không có dữ liệu để lưu.")
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if format == 'csv':
            self.data.to_csv(file_path, index=False, **kwargs)
        elif format == 'parquet':
            self.data.to_parquet(file_path, index=False, **kwargs)
        elif format == 'excel':
            self.data.to_excel(file_path, index=False, **kwargs)
        else:
            raise ValueError(f"Định dạng {format} không được hỗ trợ")
        
        logger.info(f"Đã lưu dữ liệu vào {file_path}")
        
    
    def get_info(self) -> Dict:
        """
        Lấy thông tin về dữ liệu.
        
        Returns:
        --------
        dict
            Thông tin dữ liệu
        """
        
        return self.data_info
    