"""
Evaluation Module - Đánh giá mô hình
=====================================
Module chứa các class để đánh giá mô hình và tổng hợp kết quả.
"""

from src.evaluation.metrics import MetricsCalculator
from src.evaluation.report import ReportGenerator

__all__ = ['MetricsCalculator', 'ReportGenerator']