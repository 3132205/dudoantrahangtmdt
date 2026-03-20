"""
Metrics Calculator Module
==========================
Module tính toán các metrics đánh giá mô hình.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve, auc,
                             confusion_matrix, classification_report, average_precision_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from typing import Optional, Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Class tính toán các metrics đánh giá.
    """
    
    def __init__(self):
        """Khởi tạo MetricsCalculator"""
        self.metrics_history = []
    
    def classification_metrics(self, y_true, y_pred, y_pred_proba=None) -> Dict:
        """
        Tính các metrics cho bài toán classification.
        
        Parameters:
        -----------
        y_true : array-like
            Nhãn thực tế
        y_pred : array-like
            Nhãn dự đoán
        y_pred_proba : array-like, optional
            Xác suất dự đoán
            
        Returns:
        --------
        dict
            Các metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
            except:
                metrics['roc_auc'] = 0
                metrics['pr_auc'] = 0
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def regression_metrics(self, y_true, y_pred) -> Dict:
        """
        Tính các metrics cho bài toán regression.
        
        Parameters:
        -----------
        y_true : array-like
            Giá trị thực tế
        y_pred : array-like
            Giá trị dự đoán
            
        Returns:
        --------
        dict
            Các metrics
        """
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def get_roc_curve(self, y_true, y_pred_proba) -> Dict:
        """
        Tính ROC curve.
        
        Parameters:
        -----------
        y_true : array-like
            Nhãn thực tế
        y_pred_proba : array-like
            Xác suất dự đoán
            
        Returns:
        --------
        dict
            fpr, tpr, thresholds
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': auc(fpr, tpr)
        }
    
    def get_pr_curve(self, y_true, y_pred_proba) -> Dict:
        """
        Tính Precision-Recall curve.
        
        Parameters:
        -----------
        y_true : array-like
            Nhãn thực tế
        y_pred_proba : array-like
            Xác suất dự đoán
            
        Returns:
        --------
        dict
            precision, recall, thresholds
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist() if len(thresholds) > 0 else [],
            'auc': auc(recall, precision)
        }
    
    def find_optimal_threshold(self, y_true, y_pred_proba,
                               optimize_for: str = 'f1',
                               thresholds: List[float] = None) -> Dict:
        """
        Tìm ngưỡng tối ưu.
        
        Parameters:
        -----------
        y_true : array-like
            Nhãn thực tế
        y_pred_proba : array-like
            Xác suất dự đoán
        optimize_for : str
            Metric cần tối ưu ('f1', 'precision', 'recall')
        thresholds : list
            Danh sách thresholds cần thử
            
        Returns:
        --------
        dict
            Threshold tối ưu và các metrics
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.05)
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        results_df = pd.DataFrame(results)
        
        if optimize_for == 'f1':
            optimal_idx = results_df['f1'].idxmax()
        elif optimize_for == 'precision':
            optimal_idx = results_df['precision'].idxmax()
        elif optimize_for == 'recall':
            optimal_idx = results_df['recall'].idxmax()
        else:
            optimal_idx = results_df['f1'].idxmax()
        
        optimal = results_df.loc[optimal_idx].to_dict()
        
        return {
            'optimal_threshold': optimal['threshold'],
            'metrics': optimal,
            'all_thresholds': results
        }
    
    def calculate_business_cost(self, y_true, y_pred,
                                fp_cost: float = 10,
                                fn_cost: float = 100) -> Dict:
        """
        Tính chi phí kinh doanh dựa trên confusion matrix.
        
        Parameters:
        -----------
        y_true : array-like
            Nhãn thực tế
        y_pred : array-like
            Nhãn dự đoán
        fp_cost : float
            Chi phí cho false positive
        fn_cost : float
            Chi phí cho false negative
            
        Returns:
        --------
        dict
            Chi phí và thống kê
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        total_cost = fp * fp_cost + fn * fn_cost
        avg_cost_per_sample = total_cost / len(y_true)
        
        return {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
            'fp_cost': fp_cost,
            'fn_cost': fn_cost,
            'total_cost': total_cost,
            'avg_cost_per_sample': avg_cost_per_sample
        }
    
    def compare_models(self, models_results: Dict) -> pd.DataFrame:
        """
        So sánh kết quả của nhiều mô hình.
        
        Parameters:
        -----------
        models_results : dict
            Dictionary chứa kết quả của các mô hình
            
        Returns:
        --------
        pd.DataFrame
            Bảng so sánh
        """
        comparison = pd.DataFrame(models_results).T
        return comparison.round(4)
    
    def get_classification_report(self, y_true, y_pred, target_names=None) -> str:
        """
        Lấy classification report.
        
        Parameters:
        -----------
        y_true : array-like
            Nhãn thực tế
        y_pred : array-like
            Nhãn dự đoán
        target_names : list
            Tên các lớp
            
        Returns:
        --------
        str
            Classification report
        """
        return classification_report(y_true, y_pred, target_names=target_names)
    
    def calculate_confidence_intervals(self, scores: List[float],
                                       confidence: float = 0.95) -> Dict:
        """
        Tính khoảng tin cậy cho scores.
        
        Parameters:
        -----------
        scores : list
            Danh sách các scores
        confidence : float
            Mức độ tin cậy (0-1)
            
        Returns:
        --------
        dict
            Khoảng tin cậy
        """
        import scipy.stats as st
        
        n = len(scores)
        mean = np.mean(scores)
        std = np.std(scores)
        
        se = std / np.sqrt(n)
        ci = st.t.interval(confidence, n-1, loc=mean, scale=se)
        
        return {
            'mean': mean,
            'std': std,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'confidence_level': confidence
        }
    
    def get_metrics_history(self) -> List[Dict]:
        """
        Lấy lịch sử metrics đã tính.
        
        Returns:
        --------
        list
            Lịch sử metrics
        """
        return self.metrics_history