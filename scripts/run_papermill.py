#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run Papermill Script
=====================
Script chạy lần lượt các Jupyter notebooks bằng papermill.
Tự động thực thi các notebook và lưu kết quả.
"""

import os
import sys
import yaml
import logging
import argparse
import time
import papermill as pm
from datetime import datetime
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'papermill_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NotebookRunner:
    """
    Class chạy các Jupyter notebooks theo thứ tự.
    """
    
    def __init__(self, config_path: str = 'configs/params.yaml'):
        """
        Khởi tạo NotebookRunner.
        
        Parameters:
        -----------
        config_path : str
            Đường dẫn đến file cấu hình
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.start_time = time.time()
        self.results = {}
        
        # Tạo thư mục outputs nếu chưa có
        os.makedirs('outputs/notebooks', exist_ok=True)
        os.makedirs('outputs/logs', exist_ok=True)
        
        logger.info("=" * 80)
        logger.info("KHỞI TẠO NOTEBOOK RUNNER")
        logger.info(f"Config: {config_path}")
        logger.info("=" * 80)
    
    def _load_config(self):
        """Đọc file cấu hình"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_notebooks(self) -> list:
        """
        Lấy danh sách notebooks cần chạy theo thứ tự.
        
        Returns:
        --------
        list
            Danh sách các tuples (tên, đường dẫn, có bắt buộc không)
        """
        notebooks = [
            {
                'name': '01_eda',
                'path': 'notebooks/01_eda.ipynb',
                'required': True,
                'description': 'Exploratory Data Analysis'
            },
            {
                'name': '02_preprocess_feature',
                'path': 'notebooks/02_preprocess_feature.ipynb',
                'required': True,
                'description': 'Preprocessing & Feature Engineering'
            },
            {
                'name': '03_mining_clustering',
                'path': 'notebooks/03_mining_clustering.ipynb',
                'required': True,
                'description': 'Association Mining & Clustering'
            },
            {
                'name': '04_modeling_supervised',
                'path': 'notebooks/04_modeling_supervised.ipynb',
                'required': True,
                'description': 'Supervised Modeling'
            },
            {
                'name': '04b_semi_supervised',
                'path': 'notebooks/04b_semi_supervised.ipynb',
                'required': False,
                'description': 'Semi-supervised Learning'
            },
            {
                'name': '05_evaluation_report',
                'path': 'notebooks/05_evaluation_report.ipynb',
                'required': True,
                'description': 'Evaluation & Report'
            }
        ]
        
        # Kiểm tra các file có tồn tại không
        valid_notebooks = []
        for nb in notebooks:
            if os.path.exists(nb['path']):
                valid_notebooks.append(nb)
            else:
                if nb['required']:
                    logger.error(f"Required notebook not found: {nb['path']}")
                    raise FileNotFoundError(f"Required notebook missing: {nb['path']}")
                else:
                    logger.warning(f"Optional notebook not found: {nb['path']}")
        
        return valid_notebooks
    
    def run_with_papermill(self, notebook_path: str, output_path: str = None,
                          parameters: dict = None, kernel_name: str = 'python3'):
        """
        Chạy notebook bằng papermill.
        
        Parameters:
        -----------
        notebook_path : str
            Đường dẫn đến notebook
        output_path : str
            Đường dẫn lưu notebook đầu ra
        parameters : dict
            Các tham số truyền vào notebook
        kernel_name : str
            Tên kernel
            
        Returns:
        --------
        dict
            Kết quả thực thi
        """
        if output_path is None:
            output_path = notebook_path.replace('.ipynb', '_output.ipynb')
        
        try:
            start_time = time.time()
            
            pm.execute_notebook(
                notebook_path,
                output_path,
                parameters=parameters or {},
                kernel_name=kernel_name,
                log_output=True,
                progress_bar=True
            )
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Executed {notebook_path} in {elapsed:.2f}s")
            
            return {
                'success': True,
                'time': elapsed,
                'output': output_path
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to execute {notebook_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'time': time.time() - start_time
            }
    
    def run_with_nbconvert(self, notebook_path: str, output_path: str = None):
        """
        Chạy notebook bằng nbconvert (fallback method).
        
        Parameters:
        -----------
        notebook_path : str
            Đường dẫn đến notebook
        output_path : str
            Đường dẫn lưu notebook đầu ra
            
        Returns:
        --------
        dict
            Kết quả thực thi
        """
        if output_path is None:
            output_path = notebook_path.replace('.ipynb', '_executed.ipynb')
        
        try:
            start_time = time.time()
            
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
            ep.preprocess(nb, {'metadata': {'path': 'notebooks/'}})
            
            with open(output_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Executed {notebook_path} with nbconvert in {elapsed:.2f}s")
            
            return {
                'success': True,
                'time': elapsed,
                'output': output_path
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to execute {notebook_path} with nbconvert: {e}")
            return {
                'success': False,
                'error': str(e),
                'time': time.time() - start_time
            }
    
    def run_sequential(self, use_papermill: bool = True, stop_on_error: bool = False):
        """
        Chạy các notebook theo thứ tự.
        
        Parameters:
        -----------
        use_papermill : bool
            Sử dụng papermill (True) hay nbconvert (False)
        stop_on_error : bool
            Dừng lại nếu có lỗi
            
        Returns:
        --------
        dict
            Kết quả tổng hợp
        """
        notebooks = self._get_notebooks()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"BẮT ĐẦU CHẠY {len(notebooks)} NOTEBOOKS")
        logger.info(f"{'='*80}\n")
        
        results = {
            'success': [],
            'failed': [],
            'skipped': []
        }
        
        for i, nb in enumerate(notebooks, 1):
            logger.info(f"\n{'─'*60}")
            logger.info(f"[{i}/{len(notebooks)}] ĐANG CHẠY: {nb['name']}")
            logger.info(f"Mô tả: {nb['description']}")
            logger.info(f"{'─'*60}")
            
            # Tạo đường dẫn output
            output_path = f"outputs/notebooks/{nb['name']}_output.ipynb"
            
            # Chạy notebook
            if use_papermill:
                result = self.run_with_papermill(nb['path'], output_path)
            else:
                result = self.run_with_nbconvert(nb['path'], output_path)
            
            # Lưu kết quả
            nb_result = {
                'name': nb['name'],
                'path': nb['path'],
                'output': result.get('output'),
                'time': result.get('time', 0),
                'description': nb['description']
            }
            
            if result['success']:
                results['success'].append(nb_result)
                logger.info(f"✅ HOÀN THÀNH: {nb['name']} - {result['time']:.2f}s")
            else:
                nb_result['error'] = result.get('error', 'Unknown error')
                results['failed'].append(nb_result)
                logger.error(f"❌ THẤT BẠI: {nb['name']} - {result.get('error')}")
                
                if stop_on_error and nb['required']:
                    logger.error("Dừng pipeline do lỗi ở notebook bắt buộc")
                    break
            
            # Cập nhật thời gian
            elapsed = time.time() - self.start_time
            logger.info(f"⏱️  Tổng thời gian: {elapsed:.2f}s ({elapsed/60:.2f} phút)")
        
        return results
    
    def run_selected(self, notebook_names: list, use_papermill: bool = True):
        """
        Chạy các notebook được chọn.
        
        Parameters:
        -----------
        notebook_names : list
            Danh sách tên notebook cần chạy
        use_papermill : bool
            Sử dụng papermill hay nbconvert
            
        Returns:
        --------
        dict
            Kết quả tổng hợp
        """
        all_notebooks = self._get_notebooks()
        selected = [nb for nb in all_notebooks if nb['name'] in notebook_names]
        
        if not selected:
            logger.error(f"Không tìm thấy notebook nào trong danh sách: {notebook_names}")
            return {'success': [], 'failed': [], 'skipped': []}
        
        logger.info(f"\n{'='*80}")
        logger.info(f"BẮT ĐẦU CHẠY {len(selected)} NOTEBOOKS ĐƯỢC CHỌN")
        logger.info(f"{'='*80}\n")
        
        results = {
            'success': [],
            'failed': [],
            'skipped': []
        }
        
        for i, nb in enumerate(selected, 1):
            logger.info(f"\n{'─'*60}")
            logger.info(f"[{i}/{len(selected)}] ĐANG CHẠY: {nb['name']}")
            logger.info(f"{'─'*60}")
            
            output_path = f"outputs/notebooks/{nb['name']}_output.ipynb"
            
            if use_papermill:
                result = self.run_with_papermill(nb['path'], output_path)
            else:
                result = self.run_with_nbconvert(nb['path'], output_path)
            
            nb_result = {
                'name': nb['name'],
                'path': nb['path'],
                'output': result.get('output'),
                'time': result.get('time', 0),
                'description': nb['description']
            }
            
            if result['success']:
                results['success'].append(nb_result)
                logger.info(f"✅ HOÀN THÀNH: {nb['name']} - {result['time']:.2f}s")
            else:
                nb_result['error'] = result.get('error', 'Unknown error')
                results['failed'].append(nb_result)
                logger.error(f"❌ THẤT BẠI: {nb['name']} - {result.get('error')}")
        
        return results
    
    def generate_report(self, results: dict):
        """
        Tạo báo cáo tổng kết.
        
        Parameters:
        -----------
        results : dict
            Kết quả từ run_sequential hoặc run_selected
        """
        total_time = time.time() - self.start_time
        
        report = f"""
{'='*80}
📊 BÁO CÁO TỔNG KẾT NOTEBOOK EXECUTION
{'='*80}

Thời gian bắt đầu: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}
Thời gian kết thúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Tổng thời gian: {total_time:.2f}s ({total_time/60:.2f} phút)

📈 THỐNG KÊ:
   • Thành công: {len(results['success'])} notebooks
   • Thất bại: {len(results['failed'])} notebooks
   • Bỏ qua: {len(results['skipped'])} notebooks

✅ NOTEBOOKS THÀNH CÔNG:
"""
        for nb in results['success']:
            report += f"   • {nb['name']}: {nb['time']:.2f}s\n"
        
        if results['failed']:
            report += "\n❌ NOTEBOOKS THẤT BẠI:\n"
            for nb in results['failed']:
                report += f"   • {nb['name']}: {nb.get('error', 'Unknown error')}\n"
        
        report += f"\n{'='*80}\n"
        
        logger.info(report)
        
        # Lưu báo cáo
        report_path = f"outputs/logs/execution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"📄 Đã lưu báo cáo tại: {report_path}")


def main():
    """Hàm chính"""
    parser = argparse.ArgumentParser(description='Run Jupyter notebooks with papermill')
    parser.add_argument('--config', type=str, default='configs/params.yaml',
                        help='Path to config file')
    parser.add_argument('--method', type=str, choices=['papermill', 'nbconvert'], 
                        default='papermill', help='Execution method')
    parser.add_argument('--notebooks', type=str, nargs='+',
                        help='Specific notebooks to run (e.g., --notebooks 01_eda 04_modeling)')
    parser.add_argument('--stop-on-error', action='store_true',
                        help='Stop execution on first error')
    parser.add_argument('--generate-report', action='store_true',
                        help='Generate execution report')
    
    args = parser.parse_args()
    
    # Kiểm tra file config
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    # Tạo runner
    runner = NotebookRunner(args.config)
    
    # Chạy notebooks
    use_papermill = (args.method == 'papermill')
    
    if args.notebooks:
        results = runner.run_selected(args.notebooks, use_papermill=use_papermill)
    else:
        results = runner.run_sequential(
            use_papermill=use_papermill,
            stop_on_error=args.stop_on_error
        )
    
    # Tạo báo cáo
    if args.generate_report:
        runner.generate_report(results)
    
    # Kết thúc
    total_success = len(results['success'])
    total_failed = len(results['failed'])
    
    if total_failed > 0:
        logger.error(f"❌ Hoàn thành với {total_failed} lỗi")
        sys.exit(1)
    else:
        logger.info(f"✅ Hoàn thành thành công {total_success} notebooks")
        sys.exit(0)


if __name__ == "__main__":
    main()