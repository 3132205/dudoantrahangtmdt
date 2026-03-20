"""
Semi-Supervised Learning Module
=================================
Module huấn luyện các mô hình bán giám sát.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.semi_supervised import SelfTrainingClassifier, LabelPropagation, LabelSpreading
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)
from typing import Optional, Dict, List, Tuple, Any
import logging
import time

logger = logging.getLogger(__name__)


class SemiSupervisedModel:
    """
    Class huấn luyện và đánh giá các mô hình bán giám sát.
    """
    
    def __init__(self, config: Optional[Dict] = None, random_state: int = 42):
        """
        Khởi tạo SemiSupervisedModel.
        
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
        self.training_log = []
    
    def create_missing_labels(self, X: pd.DataFrame, y: pd.Series,
                              labeled_percentage: float,
                              random_state: Optional[int] = None) -> Tuple:
        """
        Tạo dữ liệu thiếu nhãn.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target vector
        labeled_percentage : float
            % nhãn được giữ lại (0-100)
        random_state : int, optional
            Random seed
            
        Returns:
        --------
        tuple
            (X, y_missing, labeled_idx)
        """
        if random_state is None:
            random_state = self.random_state
        
        n_samples = len(y)
        n_labeled = int(n_samples * labeled_percentage / 100)
        
        # Stratified split để giữ tỷ lệ lớp
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_labeled, random_state=random_state)
        labeled_idx, _ = next(sss.split(X, y))
        
        # Tạo mảng nhãn với -1 cho unlabeled
        y_missing = np.full(n_samples, -1)
        y_missing[labeled_idx] = y.iloc[labeled_idx].values
        
        logger.info(f"Created missing labels: {n_labeled}/{n_samples} labeled ({labeled_percentage}%)")
        
        return X, y_missing, labeled_idx
    
    def self_training(self, X, y_missing,
                      base_estimator=None,
                      threshold: float = 0.8,
                      max_iter: int = 10,
                      criterion: str = 'threshold') -> SelfTrainingClassifier:
        """
        Huấn luyện với Self-Training.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y_missing : array-like
            Target với -1 cho unlabeled
        base_estimator : estimator
            Base estimator
        threshold : float
            Confidence threshold
        max_iter : int
            Maximum iterations
        criterion : str
            Selection criterion
            
        Returns:
        --------
        SelfTrainingClassifier
            Trained model
        """
        if base_estimator is None:
            base_estimator = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        
        start_time = time.time()
        
        self_training = SelfTrainingClassifier(
            base_estimator,
            threshold=threshold,
            criterion=criterion,
            k_best=100,
            max_iter=max_iter,
            verbose=False
        )
        
        self_training.fit(X, y_missing)
        
        train_time = time.time() - start_time
        
        self.models['self_training'] = self_training
        self.training_log.append(f"Self-training completed in {train_time:.2f}s")
        
        return self_training
    
    def label_propagation(self, X, y_missing,
                          kernel: str = 'rbf',
                          gamma: float = 20,
                          n_neighbors: int = 7,
                          max_iter: int = 30) -> LabelPropagation:
        """
        Huấn luyện với Label Propagation.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y_missing : array-like
            Target với -1 cho unlabeled
        kernel : str
            Kernel type ('rbf', 'knn')
        gamma : float
            Gamma parameter for rbf kernel
        n_neighbors : int
            Number of neighbors for knn kernel
        max_iter : int
            Maximum iterations
            
        Returns:
        --------
        LabelPropagation
            Trained model
        """
        start_time = time.time()
        
        model = LabelPropagation(
            kernel=kernel,
            gamma=gamma,
            n_neighbors=n_neighbors,
            max_iter=max_iter
        )
        
        model.fit(X, y_missing)
        
        train_time = time.time() - start_time
        
        self.models['label_propagation'] = model
        self.training_log.append(f"Label propagation completed in {train_time:.2f}s")
        
        return model
    
    def label_spreading(self, X, y_missing,
                        kernel: str = 'rbf',
                        gamma: float = 20,
                        alpha: float = 0.2,
                        max_iter: int = 30) -> LabelSpreading:
        """
        Huấn luyện với Label Spreading.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y_missing : array-like
            Target với -1 cho unlabeled
        kernel : str
            Kernel type ('rbf', 'knn')
        gamma : float
            Gamma parameter for rbf kernel
        alpha : float
            Clamping factor
        max_iter : int
            Maximum iterations
            
        Returns:
        --------
        LabelSpreading
            Trained model
        """
        start_time = time.time()
        
        model = LabelSpreading(
            kernel=kernel,
            gamma=gamma,
            alpha=alpha,
            max_iter=max_iter
        )
        
        model.fit(X, y_missing)
        
        train_time = time.time() - start_time
        
        self.models['label_spreading'] = model
        self.training_log.append(f"Label spreading completed in {train_time:.2f}s")
        
        return model
    
    def evaluate(self, model, X_test, y_test) -> Dict:
        """
        Đánh giá mô hình trên test set.
        
        Parameters:
        -----------
        model : estimator
            Mô hình cần đánh giá
        X_test : array-like
            Test features
        y_test : array-like
            Test target
            
        Returns:
        --------
        dict
            Các metrics đánh giá
        """
        y_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
            if y_pred_proba.shape[1] > 1:
                y_pred_proba = y_pred_proba[:, 1]
            else:
                y_pred_proba = None
        else:
            y_pred_proba = None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                metrics['pr_auc'] = average_precision_score(y_test, y_pred_proba)
            except:
                metrics['roc_auc'] = 0
                metrics['pr_auc'] = 0
        
        return metrics
    
    def analyze_pseudo_labels(self, model, X, y_true, y_missing):
        """
        Phân tích chất lượng pseudo labels.
        
        Parameters:
        -----------
        model : estimator
            Mô hình đã train
        X : array-like
            Feature matrix
        y_true : array-like
            True labels
        y_missing : array-like
            Labels with missing (-1)
            
        Returns:
        --------
        dict
            Phân tích pseudo labels
        """
        # Lấy pseudo labels
        if hasattr(model, 'transduction_'):
            pseudo_labels = model.transduction_
        elif hasattr(model, 'label_distributions_'):
            pseudo_labels = model.label_distributions_.argmax(axis=1)
        else:
            pseudo_labels = model.predict(X)
        
        # Xác định indices của pseudo-labeled data
        labeled_mask = y_missing != -1
        pseudo_mask = ~labeled_mask
        
        # So sánh với ground truth
        pseudo_correct = (pseudo_labels[pseudo_mask] == y_true[pseudo_mask])
        
        analysis = {
            'n_pseudo': np.sum(pseudo_mask),
            'n_correct': np.sum(pseudo_correct),
            'accuracy': np.mean(pseudo_correct) if np.sum(pseudo_mask) > 0 else 0,
            'pseudo_labels': pseudo_labels[pseudo_mask],
            'true_labels': y_true[pseudo_mask]
        }
        
        # Phân tích theo confidence nếu có
        if hasattr(model, 'label_distributions_'):
            confidences = model.label_distributions_.max(axis=1)
            analysis['confidences'] = confidences[pseudo_mask]
        
        return analysis
    
    def learning_curve(self, X, y, X_test, y_test,
                       labeled_percentages: List[float],
                       method: str = 'self_training',
                       n_repeats: int = 3) -> pd.DataFrame:
        """
        Vẽ learning curve cho semi-supervised.
        
        Parameters:
        -----------
        X, y : Training data
        X_test, y_test : Test data
        labeled_percentages : list
            Danh sách % nhãn cần thử
        method : str
            Phương pháp semi-supervised
        n_repeats : int
            Số lần lặp cho mỗi % nhãn
            
        Returns:
        --------
        pd.DataFrame
            Kết quả learning curve
        """
        results = []
        
        for pct in labeled_percentages:
            pct_results = []
            
            for repeat in range(n_repeats):
                seed = self.random_state + repeat * 10
                
                # Tạo dữ liệu thiếu nhãn
                X_missing, y_missing, _ = self.create_missing_labels(X, y, pct, random_state=seed)
                
                # Train model
                if method == 'self_training':
                    model = self.self_training(X_missing, y_missing)
                elif method == 'label_propagation':
                    model = self.label_propagation(X_missing, y_missing)
                elif method == 'label_spreading':
                    model = self.label_spreading(X_missing, y_missing)
                else:
                    raise ValueError(f"Method {method} not supported")
                
                # Evaluate
                metrics = self.evaluate(model, X_test, y_test)
                pct_results.append(metrics)
            
            # Tính trung bình
            avg_metrics = {
                'percentage': pct,
                'method': method
            }
            for metric in pct_results[0].keys():
                avg_metrics[f'{metric}_mean'] = np.mean([r[metric] for r in pct_results])
                avg_metrics[f'{metric}_std'] = np.std([r[metric] for r in pct_results])
            
            results.append(avg_metrics)
        
        return pd.DataFrame(results)
    
    def get_training_log(self) -> List[str]:
        """
        Lấy log của quá trình training.
        
        Returns:
        --------
        list
            Danh sách các bước đã thực hiện
        """
        return self.training_log