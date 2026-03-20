"""
Plotting Module
================
Module vẽ các biểu đồ phục vụ EDA và đánh giá kết quả.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from typing import Optional, Dict, List, Tuple, Any
import logging
import os

logger = logging.getLogger(__name__)


class Plotter:
    """
    Class vẽ các biểu đồ.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid',
                 figsize: Tuple[int, int] = (12, 8),
                 output_dir: str = '../outputs/figures/'):
        """
        Khởi tạo Plotter.
        
        Parameters:
        -----------
        style : str
            Style của matplotlib
        figsize : tuple
            Kích thước figure mặc định
        output_dir : str
            Thư mục lưu biểu đồ
        """
        plt.style.use(style)
        self.figsize = figsize
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_target_distribution(self, y, title: str = 'Phân phối biến mục tiêu',
                                  save: bool = False, filename: str = 'target_dist.png'):
        """
        Vẽ biểu đồ phân phối target.
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Count plot
        value_counts = y.value_counts()
        labels = ['No Return', 'Return'] if set(y.unique()) == {0, 1} else value_counts.index
        
        bars = axes[0].bar(range(len(value_counts)), value_counts.values, 
                          color=['#2ecc71', '#e74c3c'])
        axes[0].set_xticks(range(len(value_counts)))
        axes[0].set_xticklabels(labels)
        axes[0].set_title('Số lượng')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        
        for bar, count in zip(bars, value_counts.values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        str(count), ha='center', va='bottom')
        
        # Pie chart
        axes[1].pie(value_counts.values, labels=labels, autopct='%1.1f%%',
                   colors=['#2ecc71', '#e74c3c'], startangle=90, shadow=True)
        axes[1].set_title('Tỷ lệ %')
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def plot_missing_values(self, df, title: str = 'Missing Values Analysis',
                            save: bool = False, filename: str = 'missing_values.png'):
        """
        Vẽ biểu đồ phân tích missing values.
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Bar plot of missing percentages
        missing_percent = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        missing_percent = missing_percent[missing_percent > 0]
        
        if len(missing_percent) > 0:
            axes[0].barh(range(len(missing_percent.head(20))), 
                        missing_percent.head(20).values, color='coral')
            axes[0].set_yticks(range(len(missing_percent.head(20))))
            axes[0].set_yticklabels(missing_percent.head(20).index)
            axes[0].set_xlabel('Missing Percentage (%)')
            axes[0].set_title(f'Top 20 columns with missing values')
        
        # Matrix plot
        import missingno as msno
        msno.matrix(df, ax=axes[1], sparkline=False, figsize=(self.figsize[0]//2, self.figsize[1]))
        axes[1].set_title('Missing Values Matrix')
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_matrix(self, df, title: str = 'Correlation Matrix',
                                save: bool = False, filename: str = 'correlation.png'):
        """
        Vẽ ma trận tương quan.
        """
        plt.figure(figsize=self.figsize)
        
        # Chỉ lấy numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr()
            
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                       center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title(title, fontsize=14)
            
            if save:
                plt.savefig(os.path.join(self.output_dir, filename), dpi=100, bbox_inches='tight')
            
            plt.show()
    
    def plot_numerical_distributions(self, df, columns: List[str] = None,
                                      title: str = 'Numerical Distributions',
                                      save: bool = False, filename: str = 'numerical_dist.png'):
        """
        Vẽ phân phối của các biến số.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = min(4, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, col in enumerate(columns):
            if i < len(axes):
                sns.histplot(data=df, x=col, kde=True, ax=axes[i], color='skyblue', bins=30)
                axes[i].set_title(f'{col}\nSkew: {df[col].skew():.2f}')
                axes[i].axvline(df[col].mean(), color='red', linestyle='--', label='Mean')
                axes[i].axvline(df[col].median(), color='green', linestyle='--', label='Median')
                axes[i].legend()
        
        # Ẩn các subplot thừa
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def plot_categorical_distributions(self, df, columns: List[str] = None,
                                        title: str = 'Categorical Distributions',
                                        save: bool = False, filename: str = 'categorical_dist.png'):
        """
        Vẽ phân phối của các biến phân loại.
        """
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*5))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, col in enumerate(columns):
            if i < len(axes):
                value_counts = df[col].value_counts().head(10)
                axes[i].bar(range(len(value_counts)), value_counts.values, color='skyblue')
                axes[i].set_xticks(range(len(value_counts)))
                axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
                axes[i].set_title(f'{col} (Top 10)')
                axes[i].set_ylabel('Count')
        
        # Ẩn các subplot thừa
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def plot_time_series(self, df, date_col: str, value_col: str,
                         title: str = 'Time Series Analysis',
                         save: bool = False, filename: str = 'timeseries.png'):
        """
        Vẽ biểu đồ chuỗi thời gian.
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Đảm bảo date_col là datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        
        df_time = df.copy()
        df_time['year_month'] = df_time[date_col].dt.to_period('M')
        
        # 1. Daily trend
        daily = df_time.groupby(date_col)[value_col].sum()
        axes[0, 0].plot(daily.index, daily.values, linewidth=1, color='blue', alpha=0.7)
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel(value_col)
        axes[0, 0].set_title('Daily Trend')
        
        # 2. Monthly trend
        monthly = df_time.groupby('year_month')[value_col].sum()
        axes[0, 1].plot(range(len(monthly)), monthly.values, 'o-', color='green')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel(value_col)
        axes[0, 1].set_title('Monthly Trend')
        axes[0, 1].set_xticks(range(0, len(monthly), max(1, len(monthly)//6)))
        axes[0, 1].set_xticklabels([str(m)[:7] for m in monthly.index[::max(1, len(monthly)//6)]], rotation=45)
        
        # 3. Day of week
        df_time['dow'] = df_time[date_col].dt.day_name()
        dow_avg = df_time.groupby('dow')[value_col].mean()
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_avg = dow_avg.reindex(dow_order)
        
        axes[1, 0].bar(range(7), dow_avg.values, color='coral')
        axes[1, 0].set_xlabel('Day of Week')
        axes[1, 0].set_ylabel(f'Average {value_col}')
        axes[1, 0].set_title('Average by Day of Week')
        axes[1, 0].set_xticks(range(7))
        axes[1, 0].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
        # 4. Hour of day (nếu có)
        if 'hour' in df_time.columns or df_time[date_col].dt.hour.nunique() > 1:
            df_time['hour'] = df_time[date_col].dt.hour
            hourly = df_time.groupby('hour')[value_col].mean()
            
            axes[1, 1].plot(hourly.index, hourly.values, 'o-', color='purple')
            axes[1, 1].set_xlabel('Hour of Day')
            axes[1, 1].set_ylabel(f'Average {value_col}')
            axes[1, 1].set_title('Average by Hour')
            axes[1, 1].set_xticks(range(0, 24, 2))
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred,
                              title: str = 'Confusion Matrix',
                              save: bool = False, filename: str = 'confusion_matrix.png'):
        """
        Vẽ confusion matrix.
        """
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0], self.figsize[1]//2))
        
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        labels = ['No Return', 'Return'] if set(y_true.unique()) == {0, 1} else ['Negative', 'Positive']
        
        # Absolute
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=labels, yticklabels=labels)
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        axes[0].set_title('Absolute')
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                    xticklabels=labels, yticklabels=labels)
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        axes[1].set_title('Normalized')
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curve(self, y_true, y_pred_proba_dict: Dict[str, np.ndarray],
                       title: str = 'ROC Curves',
                       save: bool = False, filename: str = 'roc_curves.png'):
        """
        Vẽ ROC curves cho nhiều mô hình.
        """
        plt.figure(figsize=self.figsize)
        
        for name, y_pred_proba in y_pred_proba_dict.items():
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def plot_pr_curve(self, y_true, y_pred_proba_dict: Dict[str, np.ndarray],
                      title: str = 'Precision-Recall Curves',
                      save: bool = False, filename: str = 'pr_curves.png'):
        """
        Vẽ Precision-Recall curves cho nhiều mô hình.
        """
        plt.figure(figsize=self.figsize)
        
        for name, y_pred_proba in y_pred_proba_dict.items():
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, linewidth=2, label=f'{name} (PR-AUC={pr_auc:.3f})')
        
        # Baseline
        pos_ratio = y_true.mean()
        plt.axhline(y=pos_ratio, color='k', linestyle='--', label=f'Baseline (pos_ratio={pos_ratio:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame,
                                 title: str = 'Feature Importance',
                                 top_n: int = 20,
                                 save: bool = False, filename: str = 'feature_importance.png'):
        """
        Vẽ biểu đồ feature importance.
        """
        plt.figure(figsize=(self.figsize[0], self.figsize[1]))
        
        top_features = importance_df.head(top_n).sort_values('importance', ascending=True)
        
        plt.barh(range(len(top_features)), top_features['importance'].values, color='forestgreen')
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance')
        plt.title(f'{title} (Top {top_n})')
        plt.gca().invert_yaxis()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def plot_learning_curve(self, results_df: pd.DataFrame,
                            x_col: str = 'percentage',
                            y_col: str = 'f1_mean',
                            hue_col: str = 'method',
                            title: str = 'Learning Curve',
                            save: bool = False, filename: str = 'learning_curve.png'):
        """
        Vẽ learning curve cho semi-supervised.
        """
        plt.figure(figsize=self.figsize)
        
        methods = results_df[hue_col].unique()
        
        for method in methods:
            method_data = results_df[results_df[hue_col] == method]
            plt.plot(method_data[x_col], method_data[y_col], 'o-', linewidth=2, label=method)
        
        plt.xlabel('Percentage of Labeled Data (%)')
        plt.ylabel('F1 Score')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(self, results_df: pd.DataFrame,
                              title: str = 'Model Comparison',
                              save: bool = False, filename: str = 'model_comparison.png'):
        """
        Vẽ biểu đồ so sánh các mô hình.
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            
            if metric in results_df.columns:
                values = results_df[metric].sort_values()
                bars = ax.barh(range(len(values)), values.values, color='skyblue')
                ax.set_yticks(range(len(values)))
                ax.set_yticklabels(values.index)
                ax.set_xlabel(metric)
                ax.set_title(f'{metric.capitalize()} Comparison')
                ax.set_xlim(0, 1)
                
                for bar, val in zip(bars, values.values):
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{val:.3f}', va='center')
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def plot_threshold_analysis(self, threshold_df: pd.DataFrame,
                                optimal_threshold: float = None,
                                title: str = 'Threshold Analysis',
                                save: bool = False, filename: str = 'threshold_analysis.png'):
        """
        Vẽ biểu đồ phân tích threshold.
        """
        plt.figure(figsize=self.figsize)
        
        plt.plot(threshold_df['threshold'], threshold_df['precision'], 'o-', label='Precision', linewidth=2)
        plt.plot(threshold_df['threshold'], threshold_df['recall'], 's-', label='Recall', linewidth=2)
        plt.plot(threshold_df['threshold'], threshold_df['f1'], '^-', label='F1', linewidth=2)
        
        if optimal_threshold:
            plt.axvline(x=optimal_threshold, color='red', linestyle='--', 
                       label=f'Optimal ({optimal_threshold:.2f})')
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def plot_cluster_profiles(self, profiles: pd.DataFrame,
                              title: str = 'Cluster Profiles',
                              save: bool = False, filename: str = 'cluster_profiles.png'):
        """
        Vẽ biểu đồ profiles của các cụm.
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # 1. Cluster sizes
        sizes = profiles['count'].values if 'count' in profiles.columns else [1] * len(profiles)
        labels = [f'Cluster {i}' for i in profiles.index]
        
        axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, shadow=True)
        axes[0].set_title('Cluster Distribution')
        
        # 2. Feature comparison (heatmap)
        feature_cols = [col for col in profiles.columns if col not in ['count', 'percentage']]
        if feature_cols:
            # Normalize for heatmap
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            profiles_scaled = scaler.fit_transform(profiles[feature_cols])
            
            sns.heatmap(profiles_scaled.T, annot=profiles[feature_cols].T.round(2),
                       fmt='.2f', cmap='RdBu_r', center=0, ax=axes[1],
                       xticklabels=[f'Cluster {i}' for i in profiles.index],
                       yticklabels=feature_cols)
            axes[1].set_title('Feature Comparison')
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=100, bbox_inches='tight')
        
        plt.show()