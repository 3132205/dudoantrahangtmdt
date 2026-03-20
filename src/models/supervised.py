"""
Supervised Learning Module
===========================
Module huấn luyện các mô hình có giám sát.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from typing import Optional, Dict, List, Tuple, Any, Union
import logging
import time

logger = logging.getLogger(__name__)


class SupervisedModel:
    """
    Class huấn luyện và đánh giá các mô hình supervised learning.
    """
    
    def __init__(self, config: Optional[Dict] = None, random_state: int = 42):
        """
        Khởi tạo SupervisedModel.
        
        Parameters:
        -----------
        config : dict, optional
            Cấu hình cho mô hình
        random_state : int
            Random seed
        """
        self.config = config or {}
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.training_log = []
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series,
                     test_size: float = 0.2,
                     stratify: bool = True) -> Tuple:
        """
        Chuẩn bị dữ liệu train/test.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target vector
        test_size : float
            Tỷ lệ test
        stratify : bool
            Có stratify theo target không
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
        
        self.training_log.append(f"Split data: train={len(X_train)}, test={len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_baseline_models(self, X_train, y_train, X_test, y_test):
        """
        Huấn luyện các baseline models.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        X_test : array-like
            Test features
        y_test : array-like
            Test target
        """
        baseline_models = {
            'Dummy': DummyClassifier(strategy='most_frequent', random_state=self.random_state),
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000, class_weight='balanced'),
            'Decision Tree': DecisionTreeClassifier(random_state=self.random_state, max_depth=5, class_weight='balanced')
        }
        
        for name, model in baseline_models.items():
            start_time = time.time()
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            train_time = time.time() - start_time
            
            # Tính metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            metrics['train_time'] = train_time
            
            self.models[name] = model
            self.results[name] = metrics
            
            self.training_log.append(f"Trained {name} in {train_time:.2f}s")
    
    def train_random_forest(self, X_train, y_train, X_test, y_test,
                            param_grid: Optional[Dict] = None,
                            cv_folds: int = 5,
                            scoring: str = 'f1',
                            n_jobs: int = -1):
        """
        Huấn luyện Random Forest với hyperparameter tuning.
        
        Parameters:
        -----------
        X_train, y_train : Training data
        X_test, y_test : Test data
        param_grid : dict
            Grid of parameters to search
        cv_folds : int
            Number of cross-validation folds
        scoring : str
            Scoring metric
        n_jobs : int
            Number of parallel jobs
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', 'balanced_subsample']
            }
        
        start_time = time.time()
        
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=n_jobs)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(X_test)
        y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['train_time'] = train_time
        metrics['best_params'] = grid_search.best_params_
        metrics['cv_score'] = grid_search.best_score_
        
        self.models['Random Forest (Tuned)'] = best_rf
        self.results['Random Forest (Tuned)'] = metrics
        
        self.training_log.append(f"Trained Random Forest in {train_time:.2f}s")
        self.training_log.append(f"Best params: {grid_search.best_params_}")
        
        return best_rf
    
    def train_xgboost(self, X_train, y_train, X_test, y_test,
                      param_grid: Optional[Dict] = None,
                      cv_folds: int = 5,
                      scoring: str = 'f1',
                      n_jobs: int = -1):
        """
        Huấn luyện XGBoost với hyperparameter tuning.
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'scale_pos_weight': [1, 2, 5]
            }
        
        start_time = time.time()
        
        xgb_model = xgb.XGBClassifier(
            random_state=self.random_state,
            n_jobs=n_jobs,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        best_xgb = grid_search.best_estimator_
        y_pred = best_xgb.predict(X_test)
        y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['train_time'] = train_time
        metrics['best_params'] = grid_search.best_params_
        metrics['cv_score'] = grid_search.best_score_
        
        self.models['XGBoost (Tuned)'] = best_xgb
        self.results['XGBoost (Tuned)'] = metrics
        
        self.training_log.append(f"Trained XGBoost in {train_time:.2f}s")
        
        return best_xgb
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test,
                       param_grid: Optional[Dict] = None,
                       cv_folds: int = 5,
                       scoring: str = 'f1',
                       n_jobs: int = -1):
        """
        Huấn luyện LightGBM với hyperparameter tuning.
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7, -1],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [15, 31, 63],
                'class_weight': ['balanced', None]
            }
        
        start_time = time.time()
        
        lgb_model = lgb.LGBMClassifier(
            random_state=self.random_state,
            n_jobs=n_jobs,
            verbose=-1
        )
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            lgb_model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        best_lgb = grid_search.best_estimator_
        y_pred = best_lgb.predict(X_test)
        y_pred_proba = best_lgb.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['train_time'] = train_time
        metrics['best_params'] = grid_search.best_params_
        metrics['cv_score'] = grid_search.best_score_
        
        self.models['LightGBM (Tuned)'] = best_lgb
        self.results['LightGBM (Tuned)'] = metrics
        
        self.training_log.append(f"Trained LightGBM in {train_time:.2f}s")
        
        return best_lgb
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Tính các metrics đánh giá.
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
            except:
                metrics['roc_auc'] = 0
                metrics['pr_auc'] = 0
        
        return metrics
    
    def get_feature_importance(self, model_name: str, feature_names: List[str]) -> pd.DataFrame:
        """
        Lấy feature importance từ mô hình.
        
        Parameters:
        -----------
        model_name : str
            Tên mô hình
        feature_names : list
            Danh sách tên features
            
        Returns:
        --------
        pd.DataFrame
            Feature importance
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Lấy mô hình tốt nhất dựa trên F1 score.
        
        Returns:
        --------
        tuple
            (model_name, model)
        """
        if not self.results:
            return None, None
        
        results_df = pd.DataFrame(self.results).T
        best_name = results_df['f1'].idxmax()
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        return best_name, self.models[best_name]
    
    def cross_validate(self, model, X, y, cv_folds: int = 5, scoring: List[str] = None):
        """
        Thực hiện cross-validation.
        
        Parameters:
        -----------
        model : estimator
            Mô hình cần đánh giá
        X, y : data
            Dữ liệu
        cv_folds : int
            Số folds
        scoring : list
            Danh sách metrics cần tính
            
        Returns:
        --------
        dict
            Kết quả cross-validation
        """
        if scoring is None:
            scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        from sklearn.model_selection import cross_validate
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True)
        
        return scores
    
    def save_model(self, model_name: str, path: str):
        """
        Lưu mô hình ra file.
        
        Parameters:
        -----------
        model_name : str
            Tên mô hình
        path : str
            Đường dẫn lưu
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.models[model_name], path)
        
        self.training_log.append(f"Saved model {model_name} to {path}")
    
    def load_model(self, path: str, name: str = None):
        """
        Load mô hình từ file.
        
        Parameters:
        -----------
        path : str
            Đường dẫn file
        name : str
            Tên mô hình (nếu None thì lấy tên file)
        """
        model = joblib.load(path)
        
        if name is None:
            name = os.path.splitext(os.path.basename(path))[0]
        
        self.models[name] = model
        
        self.training_log.append(f"Loaded model {name} from {path}")
        
        return model
    
    def get_results(self) -> pd.DataFrame:
        """
        Lấy kết quả đánh giá các mô hình.
        
        Returns:
        --------
        pd.DataFrame
            Kết quả các mô hình
        """
        return pd.DataFrame(self.results).T
    
    def get_training_log(self) -> List[str]:
        """
        Lấy log của quá trình training.
        
        Returns:
        --------
        list
            Danh sách các bước đã thực hiện
        """
        return self.training_log