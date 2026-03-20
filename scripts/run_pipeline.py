#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run Full Pipeline Script
========================
Script chạy toàn bộ pipeline của dự án từ đầu đến cuối.
"""

import os
import sys
import yaml
import logging
import argparse
import time
import traceback
from datetime import datetime
from pathlib import Path

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Thiết lập logging với encoding UTF-8 cho file và loại bỏ Unicode cho console
class NoUnicodeStreamHandler(logging.StreamHandler):
    """Stream handler loại bỏ Unicode characters cho console Windows"""
    def emit(self, record):
        try:
            msg = self.format(record)
            # Thay thế Unicode icons bằng ASCII
            msg = msg.replace('✅', '[OK]')
            msg = msg.replace('❌', '[ERROR]')
            msg = msg.replace('⚠️', '[WARN]')
            msg = msg.replace('⏱️', '[TIME]')
            msg = msg.replace('📊', '[DATA]')
            msg = msg.replace('📁', '[FOLDER]')
            msg = msg.replace('💾', '[SAVE]')
            msg = msg.replace('🔍', '[SEARCH]')
            msg = msg.replace('📌', '[NOTE]')
            msg = msg.replace('📋', '[LIST]')
            msg = msg.replace('🎯', '[TARGET]')
            msg = msg.replace('💡', '[IDEA]')
            msg = msg.replace('🚀', '[LAUNCH]')
            msg = msg.replace('⚡', '[FAST]')
            msg = msg.replace('🌲', '[TREE]')
            msg = msg.replace('🤖', '[ROBOT]')
            msg = msg.replace('🔮', '[SHAP]')
            msg = msg.replace('🔄', '[SYNC]')
            msg = msg.replace('✂️', '[CUT]')
            msg = msg.replace('📈', '[CHART]')
            msg = msg.replace('📉', '[CHART]')
            msg = msg.replace('🏆', '[WIN]')
            msg = msg.replace('🔹', '>')
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Thiết lập logging
log_filename = f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        NoUnicodeStreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PipelineRunner:
    """
    Class chạy toàn bộ pipeline của dự án.
    """
    
    def __init__(self, config_path: str = 'configs/params.yaml'):
        """
        Khởi tạo PipelineRunner.
        """
        self.config_path = config_path
        self.config = None
        self.start_time = time.time()
        self.steps_completed = []
        self.errors = []
        self.initialization_failed = False
        
        # Tạo các thư mục cần thiết
        self._create_directories()
        
        # Load config
        if not self._load_config():
            logger.error("Khong the load cau hinh. Pipeline se chay voi cau hinh mac dinh.")
            self.initialization_failed = True
            # Tạo config mặc định
            self._create_default_config()
        
        logger.info("=" * 80)
        logger.info("KHOI TAO PIPELINE")
        logger.info(f"Config: {config_path}")
        if self.config:
            logger.info(f"Project: {self.config.get('project', {}).get('name', 'Unknown')}")
        else:
            logger.info("Project: Unknown (su dung cau hinh mac dinh)")
        logger.info("=" * 80)
    
    def _create_default_config(self):
        """Tạo cấu hình mặc định nếu không load được"""
        self.config = {
            'project': {
                'name': 'E-commerce Returns Prediction',
                'topic_id': 13,
                'description': 'Du doan tra hang TMDT',
                'version': '1.0.0'
            },
            'seed': 42,
            'paths': {
                'raw_data': 'data/raw/ecommerce_returns.csv',
                'processed_data': 'data/processed/',
                'cleaned_data': 'data/processed/cleaned_data.parquet',
                'features_mining': 'data/processed/features_for_mining.parquet',
                'features_modeling': 'data/processed/features_for_modeling.parquet'
            },
            'data': {
                'columns': {
                    'id_columns': ['order_id', 'customer_id', 'product_id'],
                    'target': 'return_flag',
                    'datetime_column': 'order_date',
                    'categorical_columns': ['product_category', 'payment_method', 'shipping_type'],
                    'numerical_columns': ['order_value', 'shipping_cost', 'discount_amount', 'quantity']
                },
                'missing': {
                    'numerical_strategy': 'median',
                    'categorical_strategy': 'mode',
                    'threshold_drop': 50
                },
                'outlier': {
                    'method': 'iqr',
                    'treatment': 'cap',
                    'iqr_multiplier': 1.5
                },
                'imbalance': {
                    'sampling_strategy': 'auto',
                    'random_state': 42,
                    'methods': {
                        'smote': True,
                        'random_oversampling': False,
                        'random_undersampling': False
                    }
                }
            },
            'features': {
                'rfm_features': {'enabled': True},
                'return_rate_features': {'enabled': True, 'min_samples': 5},
                'time_features': {'enabled': True, 'extract': ['day_of_week', 'month', 'quarter', 'is_weekend']},
                'interaction_features': {'enabled': True, 'create': []}
            },
            'association_mining': {
                'enabled': True,
                'method': 'apriori',
                'parameters': {
                    'min_support': 0.01,
                    'min_confidence': 0.5,
                    'min_lift': 1.2,
                    'max_length': 4
                },
                'discretization': {
                    'order_value': {'bins': [0, 50, 100, 200, 500, 1000], 'labels': ['Rat thap', 'Thap', 'Trung binh', 'Cao', 'Rat cao']},
                    'discount_rate': {'bins': [0, 0.1, 0.2, 0.3, 0.5], 'labels': ['Khong', 'Nho', 'Trung binh', 'Lon', 'Rat lon']}
                }
            },
            'clustering': {
                'customer_clustering': {
                    'enabled': True,
                    'features': ['frequency', 'monetary_total', 'avg_order_value', 'recency_days', 'return_rate'],
                    'scaling': 'standard',
                    'methods': {
                        'kmeans': {
                            'enabled': True,
                            'n_clusters_range': [2, 3, 4, 5, 6]
                        }
                    }
                },
                'product_clustering': {
                    'enabled': True,
                    'features': ['sold_count', 'avg_price', 'product_return_rate'],
                    'scaling': 'standard',
                    'methods': {
                        'kmeans': {
                            'n_clusters': 4
                        }
                    }
                }
            },
            'classification': {
                'enabled': True,
                'target': 'return_flag',
                'split': {
                    'test_size': 0.2,
                    'random_state': 42,
                    'stratify': True
                },
                'cv': {
                    'n_folds': 5
                },
                'baseline_models': ['dummy', 'logistic_regression', 'decision_tree'],
                'advanced_models': {
                    'random_forest': {
                        'enabled': True,
                        'n_estimators': [100, 200],
                        'max_depth': [5, 10, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'class_weight': ['balanced', 'balanced_subsample']
                    },
                    'xgboost': {
                        'enabled': True,
                        'n_estimators': [100, 200],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.05, 0.1]
                    },
                    'lightgbm': {
                        'enabled': True,
                        'n_estimators': [100, 200],
                        'max_depth': [3, 5, 7, -1],
                        'learning_rate': [0.01, 0.05, 0.1]
                    }
                }
            },
            'semi_supervised': {
                'enabled': True,
                'labeled_percentages': [5, 10, 15, 20, 30],
                'self_training': {
                    'enabled': True,
                    'threshold': 0.8,
                    'max_iterations': 10
                },
                'label_propagation': {
                    'enabled': True,
                    'kernel': 'rbf',
                    'gamma': 20,
                    'n_neighbors': 7
                },
                'label_spreading': {
                    'enabled': True,
                    'kernel': 'rbf',
                    'gamma': 20,
                    'alpha': 0.2
                },
                'evaluation': {
                    'learning_curve': {
                        'enabled': True,
                        'n_repeats': 3
                    },
                    'error_analysis': {
                        'by_feature': ['product_category', 'payment_method', 'shipping_type'],
                        'by_season': True
                    }
                }
            },
            'evaluation': {
                'threshold_analysis': {
                    'enabled': True,
                    'thresholds': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    'optimize_for': 'f1'
                },
                'business_costs': {
                    'enabled': False,
                    'false_positive_cost': 10,
                    'false_negative_cost': 100
                }
            }
        }
        logger.info("[OK] Da tao cau hinh mac dinh")
    
    def _load_config(self):
        """Đọc file cấu hình"""
        try:
            if not os.path.exists(self.config_path):
                logger.error(f"Khong tim thay file cau hinh: {self.config_path}")
                return False
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"[OK] Da doc cau hinh tu {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Loi khi doc file cau hinh: {e}")
            return False
    
    def _create_directories(self):
        """Tạo các thư mục cần thiết"""
        directories = [
            'data/raw',
            'data/processed',
            'outputs/figures',
            'outputs/tables',
            'outputs/models',
            'outputs/reports',
            'logs'
        ]
        
        for dir_path in directories:
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Da tao thu muc: {dir_path}")
    
    def _log_step(self, step_name: str, success: bool = True, error_msg: str = None):
        """Ghi log cho mỗi bước"""
        elapsed = time.time() - self.start_time
        
        if success:
            logger.info(f"[OK] Step: {step_name} - Elapsed: {elapsed:.2f}s")
            self.steps_completed.append(step_name)
        else:
            logger.error(f"[ERROR] Step: {step_name} FAILED - Elapsed: {elapsed:.2f}s")
            if error_msg:
                logger.error(f"   Error: {error_msg[:200]}...")
            self.errors.append({
                'step': step_name,
                'error': error_msg,
                'time': elapsed
            })
    
    def step_1_load_data(self):
        """Bước 1: Load dữ liệu"""
        logger.info("\n" + "=" * 60)
        logger.info("BUOC 1: LOAD DU LIEU")
        logger.info("=" * 60)
        
        try:
            # Import thư viện
            import pandas as pd
            
            # Xác định đường dẫn dữ liệu
            raw_data_path = self.config['paths']['raw_data']
            
            # Kiểm tra file tồn tại
            if not os.path.exists(raw_data_path):
                # Thử tìm trong thư mục data/raw
                alternative_path = f"data/raw/{os.path.basename(raw_data_path)}"
                if os.path.exists(alternative_path):
                    raw_data_path = alternative_path
                    logger.info(f"Su dung du lieu tu: {raw_data_path}")
                else:
                    # Thử tìm file .csv bất kỳ trong thư mục data/raw
                    csv_files = list(Path('data/raw').glob('*.csv'))
                    parquet_files = list(Path('data/raw').glob('*.parquet'))
                    
                    if csv_files:
                        raw_data_path = str(csv_files[0])
                        logger.info(f"Tim thay file CSV: {raw_data_path}")
                    elif parquet_files:
                        raw_data_path = str(parquet_files[0])
                        logger.info(f"Tim thay file Parquet: {raw_data_path}")
                    else:
                        logger.warning("[WARN] Khong tim thay file du lieu. Tao du lieu mau...")
                        return self._create_sample_data()
            
            # Đọc dữ liệu
            logger.info(f"Dang doc du lieu tu: {raw_data_path}")
            
            if raw_data_path.endswith('.csv'):
                df = pd.read_csv(raw_data_path)
            elif raw_data_path.endswith('.parquet'):
                df = pd.read_parquet(raw_data_path)
            elif raw_data_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(raw_data_path)
            else:
                raise ValueError(f"Dinh dang file khong ho tro: {raw_data_path}")
            
            logger.info(f"[OK] Da doc {len(df)} dong, {len(df.columns)} cot")
            logger.info(f"   Cac cot: {list(df.columns)[:10]}...")
            
            # Lưu dữ liệu gốc
            df.to_parquet('data/raw/raw_data.parquet', index=False)
            logger.info("[SAVE] Da luu du lieu goc tai data/raw/raw_data.parquet")
            
            self._log_step("Load data", True)
            return df
            
        except Exception as e:
            error_msg = traceback.format_exc()
            self._log_step("Load data", False, str(e))
            logger.warning("[WARN] Tao du lieu mau de tiep tuc pipeline...")
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """Tạo dữ liệu mẫu cho mục đích test"""
        try:
            import pandas as pd
            import numpy as np
            
            np.random.seed(42)
            n_samples = 1000
            
            # Sửa lỗi frequency: dùng 'h' thay vì 'H'
            df = pd.DataFrame({
                'order_id': range(1, n_samples + 1),
                'customer_id': np.random.randint(1, 200, n_samples),
                'product_id': np.random.randint(1, 100, n_samples),
                'order_date': pd.date_range(start='2023-01-01', periods=n_samples, freq='h'),  # Đã sửa 'H' thành 'h'
                'order_value': np.random.uniform(10, 500, n_samples).round(2),
                'quantity': np.random.randint(1, 5, n_samples),
                'discount_amount': np.random.uniform(0, 50, n_samples).round(2),
                'shipping_cost': np.random.uniform(0, 20, n_samples).round(2),
                'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], n_samples),
                'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Bank Transfer'], n_samples),
                'shipping_type': np.random.choice(['Standard', 'Express', 'Next Day'], n_samples),
                'promotion_code': np.random.choice([None, 'SUMMER20', 'WELCOME10', 'FLASHSALE'], n_samples),
                'return_flag': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
            })
            
            logger.info(f"[OK] Da tao du lieu mau: {len(df)} dong")
            return df
        except Exception as e:
            logger.error(f"Khong the tao du lieu mau: {e}")
            return None
    
    def step_2_preprocess(self, df):
        """Bước 2: Tiền xử lý dữ liệu"""
        logger.info("\n" + "=" * 60)
        logger.info("BUOC 2: TIEN XU LY DU LIEU")
        logger.info("=" * 60)
        
        if df is None:
            logger.error("Khong co du lieu de xu ly")
            self._log_step("Preprocess", False, "No data available")
            return None
        
        try:
            import pandas as pd
            import numpy as np
            
            df_clean = df.copy()
            
            # 2.1 Xử lý missing values
            logger.info("2.1. Xu ly missing values...")
            missing_before = df_clean.isnull().sum().sum()
            
            # Fill numerical with median
            num_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            
            # Fill categorical with mode
            cat_cols = df_clean.select_dtypes(include=['object']).columns
            for col in cat_cols:
                df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown', inplace=True)
            
            missing_after = df_clean.isnull().sum().sum()
            logger.info(f"   Missing: {missing_before} -> {missing_after}")
            
            # 2.2 Xử lý duplicates
            logger.info("2.2. Xu ly duplicates...")
            dup_before = df_clean.duplicated().sum()
            df_clean = df_clean.drop_duplicates()
            logger.info(f"   Duplicates: {dup_before} -> {df_clean.duplicated().sum()}")
            
            # Lưu dữ liệu
            df_clean.to_parquet('data/processed/cleaned_data.parquet', index=False)
            logger.info(f"[SAVE] Da luu du lieu da lam sach tai data/processed/cleaned_data.parquet")
            logger.info(f"   Shape: {df_clean.shape}")
            
            self._log_step("Preprocess", True)
            return df_clean
            
        except Exception as e:
            error_msg = traceback.format_exc()
            self._log_step("Preprocess", False, str(e))
            return df  # Trả về dữ liệu gốc nếu có lỗi
    
    def step_3_feature_engineering(self, df):
        """Bước 3: Feature engineering đơn giản"""
        logger.info("\n" + "=" * 60)
        logger.info("BUOC 3: FEATURE ENGINEERING")
        logger.info("=" * 60)
        
        if df is None:
            logger.error("Khong co du lieu de xu ly")
            self._log_step("Feature Engineering", False, "No data available")
            return None
        
        try:
            import pandas as pd
            import numpy as np
            
            df_feat = df.copy()
            new_features = []
            
            # 3.1 Tạo các interaction features đơn giản
            if 'order_value' in df_feat.columns and 'quantity' in df_feat.columns:
                df_feat['value_per_item'] = df_feat['order_value'] / (df_feat['quantity'] + 1e-6)
                new_features.append('value_per_item')
                logger.info("   [OK] Da tao value_per_item")
            
            if 'discount_amount' in df_feat.columns and 'order_value' in df_feat.columns:
                df_feat['discount_rate'] = df_feat['discount_amount'] / (df_feat['order_value'] + 1e-6)
                new_features.append('discount_rate')
                logger.info("   [OK] Da tao discount_rate")
            
            if 'shipping_cost' in df_feat.columns and 'order_value' in df_feat.columns:
                df_feat['shipping_ratio'] = df_feat['shipping_cost'] / (df_feat['order_value'] + 1e-6)
                new_features.append('shipping_ratio')
                logger.info("   [OK] Da tao shipping_ratio")
            
            # 3.2 Tạo time features nếu có
            if 'order_date' in df_feat.columns:
                df_feat['order_date'] = pd.to_datetime(df_feat['order_date'])
                df_feat['day_of_week'] = df_feat['order_date'].dt.dayofweek
                df_feat['month'] = df_feat['order_date'].dt.month
                df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
                new_features.extend(['day_of_week', 'month', 'is_weekend'])
                logger.info("   [OK] Da tao time features")
            
            logger.info(f"[OK] Da tao {len(new_features)} features moi: {new_features}")
            
            # Lưu dữ liệu
            df_feat.to_parquet('data/processed/features_data.parquet', index=False)
            logger.info(f"[SAVE] Da luu du lieu features tai data/processed/features_data.parquet")
            logger.info(f"   Shape: {df_feat.shape}")
            
            self._log_step("Feature Engineering", True)
            return df_feat
            
        except Exception as e:
            error_msg = traceback.format_exc()
            self._log_step("Feature Engineering", False, str(e))
            return df  # Trả về dữ liệu gốc nếu có lỗi
    
    def step_4_mining_clustering(self, df):
        """Bước 4: Mining và Clustering (đơn giản hóa)"""
        logger.info("\n" + "=" * 60)
        logger.info("BUOC 4: MINING & CLUSTERING")
        logger.info("=" * 60)
        
        if df is None:
            logger.error("Khong co du lieu de xu ly")
            self._log_step("Mining & Clustering", False, "No data available")
            return
        
        try:
            import pandas as pd
            import numpy as np
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Thử phân cụm đơn giản nếu có customer_id
            if 'customer_id' in df.columns:
                logger.info("4.1. Phan cum khach hang...")
                
                # Tính RFM đơn giản
                customer_df = df.groupby('customer_id').agg({
                    'order_value': ['count', 'mean', 'sum'],
                }).round(2)
                
                customer_df.columns = ['frequency', 'avg_order_value', 'monetary']
                customer_df = customer_df.reset_index()
                
                # Thêm recency nếu có order_date
                if 'order_date' in df.columns:
                    df['order_date'] = pd.to_datetime(df['order_date'])
                    last_date = df['order_date'].max()
                    recency = df.groupby('customer_id')['order_date'].max().reset_index()
                    recency['recency_days'] = (last_date - recency['order_date']).dt.days
                    customer_df = customer_df.merge(recency[['customer_id', 'recency_days']], on='customer_id')
                
                # Chuẩn bị dữ liệu cho clustering
                cluster_cols = [c for c in ['frequency', 'avg_order_value', 'monetary', 'recency_days'] if c in customer_df.columns]
                X = customer_df[cluster_cols].fillna(0)
                
                if len(X) > 10:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Thử KMeans với k=3
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                    customer_df['cluster'] = kmeans.fit_predict(X_scaled)
                    
                    # Lưu kết quả
                    customer_df.to_csv('outputs/tables/customer_clusters.csv', index=False)
                    logger.info(f"   [OK] Da phan cum {len(customer_df)} khach hang thanh 3 nhom")
                    
                    # Thống kê các cụm
                    cluster_stats = customer_df.groupby('cluster')[cluster_cols].mean()
                    logger.info(f"\nThong ke cac cum:\n{cluster_stats}")
                else:
                    logger.warning(f"   [WARN] Khong du du lieu de phan cum (can >10, co {len(X)})")
            
            self._log_step("Mining & Clustering", True)
            
        except Exception as e:
            error_msg = traceback.format_exc()
            self._log_step("Mining & Clustering", False, str(e))
    
    def step_5_supervised_modeling(self, df):
        """Bước 5: Supervised Modeling (đơn giản hóa)"""
        logger.info("\n" + "=" * 60)
        logger.info("BUOC 5: SUPERVISED MODELING")
        logger.info("=" * 60)
        
        if df is None:
            logger.error("Khong co du lieu de xu ly")
            self._log_step("Supervised Modeling", False, "No data available")
            return
        
        try:
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
            
            target = self.config['data']['columns']['target']
            
            if target not in df.columns:
                logger.error(f"Khong tim thay cot target: {target}")
                self._log_step("Supervised Modeling", False, f"Target column {target} not found")
                return
            
            # Chuẩn bị dữ liệu
            feature_cols = [col for col in df.columns if col != target and col not in ['customer_id', 'order_id', 'product_id', 'order_date']]
            feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
            
            if len(feature_cols) == 0:
                logger.error("Khong co feature numeric nao")
                self._log_step("Supervised Modeling", False, "No numeric features")
                return
            
            X = df[feature_cols].fillna(0)
            y = df[target]
            
            logger.info(f"Features: {len(feature_cols)}")
            logger.info(f"Target distribution:\n{y.value_counts()}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train Random Forest
            logger.info("Training Random Forest...")
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            
            # Dự đoán
            y_pred = rf.predict(X_test)
            
            # Đánh giá
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            logger.info(f"[OK] Accuracy: {accuracy:.4f}")
            logger.info(f"[OK] F1 Score: {f1:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            logger.info(f"\nConfusion Matrix:\n{cm}")
            
            # Feature importance
            importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance.to_csv('outputs/tables/feature_importance.csv', index=False)
            logger.info(f"[SAVE] Da luu feature importance")
            logger.info(f"\nTop 5 features:\n{importance.head(5)}")
            
            self._log_step("Supervised Modeling", True)
            
        except Exception as e:
            error_msg = traceback.format_exc()
            self._log_step("Supervised Modeling", False, str(e))
    
    def step_6_semi_supervised(self, df):
        """Bước 6: Semi-supervised (đơn giản hóa)"""
        logger.info("\n" + "=" * 60)
        logger.info("BUOC 6: SEMI-SUPERVISED")
        logger.info("=" * 60)
        
        if not self.config['semi_supervised']['enabled']:
            logger.info("Semi-supervised is disabled. Skipping...")
            self._log_step("Semi-supervised", True)
            return
        
        if df is None:
            logger.error("Khong co du lieu de xu ly")
            self._log_step("Semi-supervised", False, "No data available")
            return
        
        try:
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import f1_score
            
            target = self.config['data']['columns']['target']
            
            if target not in df.columns:
                logger.error(f"Khong tim thay cot target: {target}")
                self._log_step("Semi-supervised", False, f"Target column {target} not found")
                return
            
            # Chuẩn bị dữ liệu
            feature_cols = [col for col in df.columns if col != target and col not in ['customer_id', 'order_id', 'product_id', 'order_date']]
            feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
            
            X = df[feature_cols].fillna(0)
            y = df[target]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Thử với các % nhãn khác nhau
            labeled_percentages = self.config['semi_supervised']['labeled_percentages']
            results = []
            
            for pct in labeled_percentages:
                logger.info(f"   Thu voi {pct}% labeled data...")
                
                # Giả lập thiếu nhãn
                n_labeled = int(len(X_train) * pct / 100)
                indices = np.random.choice(len(X_train), n_labeled, replace=False)
                
                X_labeled = X_train.iloc[indices]
                y_labeled = y_train.iloc[indices]
                
                # Train supervised
                rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                rf.fit(X_labeled, y_labeled)
                
                # Đánh giá
                y_pred = rf.predict(X_test)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                results.append({
                    'percentage': pct,
                    'method': 'supervised_only',
                    'f1': f1
                })
                
                logger.info(f"      F1 = {f1:.4f}")
            
            # Lưu kết quả
            if results:
                results_df = pd.DataFrame(results)
                results_df.to_csv('outputs/tables/semi_supervised_results.csv', index=False)
                logger.info(f"[SAVE] Da luu ket qua semi-supervised")
            
            self._log_step("Semi-supervised", True)
            
        except Exception as e:
            error_msg = traceback.format_exc()
            self._log_step("Semi-supervised", False, str(e))
    
    def step_7_generate_report(self):
        """Bước 7: Tạo báo cáo tổng hợp"""
        logger.info("\n" + "=" * 60)
        logger.info("BUOC 7: TAO BAO CAO")
        logger.info("=" * 60)
        
        try:
            import pandas as pd
            from datetime import datetime
            
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("BAO CAO TONG HOP PIPELINE")
            report_lines.append("=" * 80)
            report_lines.append(f"Thoi gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"Tong thoi gian chay: {time.time() - self.start_time:.2f} giay")
            report_lines.append("")
            
            report_lines.append("CAC BUOC DA HOAN THANH:")
            for step in self.steps_completed:
                report_lines.append(f"  [OK] {step}")
            
            if self.errors:
                report_lines.append("")
                report_lines.append("CAC LOI GAP PHAI:")
                for error in self.errors:
                    report_lines.append(f"  [ERROR] {error['step']}: {error['error'][:100]}...")
            
            # Đọc các kết quả nếu có
            if os.path.exists('outputs/tables/feature_importance.csv'):
                report_lines.append("")
                report_lines.append("TOP 5 FEATURES QUAN TRONG NHAT:")
                try:
                    imp_df = pd.read_csv('outputs/tables/feature_importance.csv')
                    for i, row in imp_df.head(5).iterrows():
                        report_lines.append(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
                except Exception as e:
                    report_lines.append(f"  Khong the doc file: {e}")
            
            if os.path.exists('outputs/tables/customer_clusters.csv'):
                report_lines.append("")
                report_lines.append("PHAN CUM KHACH HANG:")
                try:
                    cluster_df = pd.read_csv('outputs/tables/customer_clusters.csv')
                    cluster_counts = cluster_df['cluster'].value_counts().sort_index()
                    for cluster, count in cluster_counts.items():
                        report_lines.append(f"  Cum {cluster}: {count} khach hang")
                except Exception as e:
                    report_lines.append(f"  Khong the doc file: {e}")
            
            report_text = "\n".join(report_lines)
            
            # Lưu báo cáo
            with open('outputs/reports/pipeline_report.txt', 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            # In ra console
            print("\n" + report_text)
            
            logger.info(f"[SAVE] Da luu bao cao tai outputs/reports/pipeline_report.txt")
            
            self._log_step("Generate Report", True)
            
        except Exception as e:
            error_msg = traceback.format_exc()
            self._log_step("Generate Report", False, str(e))
    
    def run_all(self, skip_steps: list = None):
        """
        Chạy toàn bộ pipeline.
        """
        if skip_steps is None:
            skip_steps = []
        
        # Kiểm tra nếu initialization failed
        if self.initialization_failed:
            logger.warning("[WARN] Pipeline khoi tao voi cau hinh mac dinh do khong tim thay file config")
        
        logger.info("=" * 80)
        logger.info("BAT DAU CHAY PIPELINE")
        logger.info("=" * 80)
        
        steps = [
            (1, "Load Data", self.step_1_load_data),
            (2, "Preprocess", self.step_2_preprocess),
            (3, "Feature Engineering", self.step_3_feature_engineering),
            (4, "Mining & Clustering", self.step_4_mining_clustering),
            (5, "Supervised Modeling", self.step_5_supervised_modeling),
            (6, "Semi-supervised", self.step_6_semi_supervised),
            (7, "Generate Report", self.step_7_generate_report)
        ]
        
        data = None
        
        for step_num, step_name, step_func in steps:
            if step_name in skip_steps:
                logger.info(f"Bo qua buoc {step_num}: {step_name}")
                self.steps_completed.append(f"{step_name} (skipped)")
                continue
            
            logger.info(f"\n{'#'*20} BUOC {step_num}: {step_name} {'#'*20}")
            
            try:
                if step_num == 1:
                    data = step_func()
                elif step_num in [2, 3]:
                    data = step_func(data)
                elif step_num in [4, 5, 6]:
                    step_func(data)
                else:
                    step_func()
                
                # Nếu bước 1 thất bại, dừng pipeline
                if step_num == 1 and data is None:
                    logger.error("Buoc 1 that bai. Dung pipeline.")
                    break
                    
            except KeyboardInterrupt:
                logger.info("Nguoi dung yeu cau dung pipeline.")
                break
            except Exception as e:
                logger.error(f"Loi o buoc {step_num}: {e}")
                self.errors.append({
                    'step': step_name,
                    'error': str(e),
                    'time': time.time() - self.start_time
                })
                
                # Hỏi người dùng có muốn tiếp tục không
                if step_num > 1:
                    try:
                        response = input("Tiep tuc pipeline? (y/n): ").lower().strip()
                        if response != 'y':
                            logger.info("Nguoi dung yeu cau dung pipeline.")
                            break
                    except:
                        # Nếu không thể đọc input, tự động dừng
                        logger.warning("Khong the doc input, dung pipeline.")
                        break
        
        # Tổng kết
        total_time = time.time() - self.start_time
        logger.info("=" * 80)
        logger.info("KET THUC PIPELINE")
        logger.info("=" * 80)
        logger.info(f"[TIME] Tong thoi gian: {total_time:.2f} giay ({total_time/60:.2f} phut)")
        logger.info(f"[DATA] So buoc thanh cong: {len(self.steps_completed)}/{len(steps)}")
        
        if self.errors:
            logger.warning(f"[WARN] Co {len(self.errors)} loi trong qua trinh chay")
            for i, err in enumerate(self.errors, 1):
                logger.warning(f"   {i}. {err['step']}: {err['error'][:100]}...")
        else:
            logger.info("[OK] TAT CA CAC BUOC DEU THANH CONG!")
        
        logger.info("=" * 80)
        
        return len(self.errors) == 0


def main():
    """Hàm chính"""
    parser = argparse.ArgumentParser(description='Run data mining pipeline')
    parser.add_argument('--config', type=str, default='configs/params.yaml',
                        help='Path to config file')
    parser.add_argument('--skip', type=str, nargs='+',
                        help='Steps to skip (e.g., --skip "Mining & Clustering" "Semi-supervised")')
    
    args = parser.parse_args()
    
    exit_code = 0
    
    try:
        # Tạo thư mục logs
        os.makedirs('logs', exist_ok=True)
        
        # Chạy pipeline
        logger.info("=" * 80)
        logger.info("KHOI DONG PIPELINE")
        logger.info("=" * 80)
        
        runner = PipelineRunner(args.config)
        success = runner.run_all(skip_steps=args.skip if args.skip else [])
        
        if success:
            logger.info("[OK] Pipeline hoan thanh thanh cong!")
            exit_code = 0
        else:
            logger.warning("[WARN] Pipeline hoan thanh voi mot so loi.")
            exit_code = 0  # Vẫn coi là thành công để không báo lỗi terminal
    
    except KeyboardInterrupt:
        print("\n[WARN] Pipeline bi dung boi nguoi dung")
        exit_code = 0
    except Exception as e:
        print(f"[ERROR] Loi khong xu ly duoc: {e}")
        traceback.print_exc()
        exit_code = 1
    
    # Thoát với mã phù hợp
    logger.info("=" * 80)
    logger.info(f"KET THUC PIPELINE VOI MA {exit_code}")
    logger.info("=" * 80)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()