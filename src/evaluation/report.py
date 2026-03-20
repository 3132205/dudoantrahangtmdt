"""
Report Generator Module
========================
Module tổng hợp kết quả và tạo báo cáo.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Class tổng hợp kết quả và tạo báo cáo.
    """
    
    def __init__(self, output_dir: str = '../outputs/'):
        """
        Khởi tạo ReportGenerator.
        
        Parameters:
        -----------
        output_dir : str
            Thư mục đầu ra
        """
        self.output_dir = output_dir
        self.report_data = {}
        self.generation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def add_section(self, name: str, data: Any):
        """
        Thêm một section vào báo cáo.
        
        Parameters:
        -----------
        name : str
            Tên section
        data : any
            Dữ liệu của section
        """
        self.report_data[name] = data
    
    def generate_summary_table(self) -> pd.DataFrame:
        """
        Tạo bảng tổng hợp kết quả.
        
        Returns:
        --------
        pd.DataFrame
            Bảng tổng hợp
        """
        summary = []
        
        for section, data in self.report_data.items():
            if isinstance(data, pd.DataFrame):
                summary.append({
                    'Section': section,
                    'Type': 'DataFrame',
                    'Shape': f"{data.shape[0]}x{data.shape[1]}",
                    'Preview': str(data.columns.tolist()[:5])
                })
            elif isinstance(data, dict):
                summary.append({
                    'Section': section,
                    'Type': 'Dictionary',
                    'Shape': f"{len(data)} keys",
                    'Preview': str(list(data.keys())[:5])
                })
            elif isinstance(data, list):
                summary.append({
                    'Section': section,
                    'Type': 'List',
                    'Shape': f"{len(data)} items",
                    'Preview': str(data[:5])
                })
            else:
                summary.append({
                    'Section': section,
                    'Type': type(data).__name__,
                    'Shape': 'N/A',
                    'Preview': str(data)[:50]
                })
        
        return pd.DataFrame(summary)
    
    def save_to_json(self, filename: str = 'report.json'):
        """
        Lưu báo cáo ra file JSON.
        
        Parameters:
        -----------
        filename : str
            Tên file đầu ra
        """
        # Chuyển đổi các đối tượng không serializable
        serializable_data = {}
        
        for key, value in self.report_data.items():
            if isinstance(value, pd.DataFrame):
                serializable_data[key] = {
                    'type': 'DataFrame',
                    'shape': value.shape,
                    'columns': value.columns.tolist(),
                    'data': value.to_dict(orient='records')
                }
            elif isinstance(value, np.ndarray):
                serializable_data[key] = {
                    'type': 'ndarray',
                    'shape': value.shape,
                    'data': value.tolist()
                }
            elif isinstance(value, (np.integer, np.floating)):
                serializable_data[key] = value.item()
            else:
                serializable_data[key] = value
        
        # Thêm metadata
        serializable_data['_metadata'] = {
            'generation_time': self.generation_time,
            'version': '1.0'
        }
        
        filepath = os.path.join(self.output_dir, 'reports', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved report to {filepath}")
    
    def save_to_csv(self, filename: str = 'report.csv'):
        """
        Lưu báo cáo ra file CSV (chỉ các DataFrame).
        
        Parameters:
        -----------
        filename : str
            Tên file đầu ra
        """
        # Tạo một DataFrame tổng hợp từ tất cả các DataFrame trong report
        combined_data = {}
        
        for key, value in self.report_data.items():
            if isinstance(value, pd.DataFrame):
                combined_data[key] = value
        
        if combined_data:
            filepath = os.path.join(self.output_dir, 'tables', filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with pd.ExcelWriter(filepath.replace('.csv', '.xlsx'), engine='openpyxl') as writer:
                for sheet_name, df in combined_data.items():
                    df.to_excel(writer, sheet_name=sheet_name[:31])
            
            logger.info(f"Saved Excel report to {filepath.replace('.csv', '.xlsx')}")
    
    def generate_markdown_report(self, filename: str = 'report.md') -> str:
        """
        Tạo báo cáo dạng Markdown.
        
        Parameters:
        -----------
        filename : str
            Tên file đầu ra
            
        Returns:
        --------
        str
            Nội dung Markdown
        """
        md_lines = []
        
        # Tiêu đề
        md_lines.append("# BÁO CÁO TỔNG HỢP DỰ ÁN")
        md_lines.append(f"*Ngày tạo: {self.generation_time}*\n")
        
        # Mục lục
        md_lines.append("## MỤC LỤC")
        for i, section in enumerate(self.report_data.keys(), 1):
            md_lines.append(f"{i}. [{section}](#{section.lower().replace(' ', '-')})")
        md_lines.append("")
        
        # Nội dung từng section
        for section, data in self.report_data.items():
            md_lines.append(f"<a name='{section.lower().replace(' ', '-')}'></a>")
            md_lines.append(f"## {i}. {section.upper()}")
            
            if isinstance(data, pd.DataFrame):
                md_lines.append(f"*Shape: {data.shape[0]} rows x {data.shape[1]} columns*\n")
                md_lines.append(data.to_markdown())
            elif isinstance(data, dict):
                md_lines.append("```json")
                md_lines.append(json.dumps(data, indent=2, ensure_ascii=False))
                md_lines.append("```")
            elif isinstance(data, list):
                md_lines.append("```")
                for item in data[:10]:
                    md_lines.append(f"- {item}")
                if len(data) > 10:
                    md_lines.append(f"... and {len(data) - 10} more")
                md_lines.append("```")
            else:
                md_lines.append(str(data))
            
            md_lines.append("\n---\n")
        
        # Lưu file
        filepath = os.path.join(self.output_dir, 'reports', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))
        
        logger.info(f"Saved markdown report to {filepath}")
        
        return '\n'.join(md_lines)
    
    def generate_html_report(self, filename: str = 'report.html') -> str:
        """
        Tạo báo cáo dạng HTML.
        
        Parameters:
        -----------
        filename : str
            Tên file đầu ra
            
        Returns:
        --------
        str
            Nội dung HTML
        """
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Báo cáo dự án - Dự đoán trả hàng TMĐT</title>
            <meta charset="UTF-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
                h2 { color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 5px; margin-top: 30px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th { background-color: #3498db; color: white; padding: 12px; }
                td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .section { margin: 30px 0; padding: 20px; background-color: #f8f9fa; border-radius: 5px; }
                .timestamp { color: #7f8c8d; font-size: 12px; text-align: right; }
                pre { background-color: #f1f1f1; padding: 10px; border-radius: 5px; overflow-x: auto; }
            </style>
        </head>
        <body>
            <h1>📊 BÁO CÁO TỔNG HỢP DỰ ÁN</h1>
            <div class="timestamp">Tạo lúc: {generation_time}</div>
            
            <div class="toc">
                <h2>MỤC LỤC</h2>
                <ul>
                    {toc}
                </ul>
            </div>
            
            {sections}
            
            <div class="footer">
                <p>© 2024 - Nhóm ... - Học phần Khai phá dữ liệu</p>
            </div>
        </body>
        </html>
        """
        
        toc_items = []
        sections_html = []
        
        for i, (section, data) in enumerate(self.report_data.items(), 1):
            # TOC
            toc_items.append(f'<li><a href="#section-{i}">{section}</a></li>')
            
            # Section content
            section_html = f'<div class="section" id="section-{i}">'
            section_html += f'<h2>{i}. {section.upper()}</h2>'
            
            if isinstance(data, pd.DataFrame):
                section_html += f'<p><em>Shape: {data.shape[0]} rows x {data.shape[1]} columns</em></p>'
                section_html += data.to_html()
            elif isinstance(data, dict):
                section_html += '<pre>' + json.dumps(data, indent=2, ensure_ascii=False) + '</pre>'
            elif isinstance(data, list):
                section_html += '<ul>'
                for item in data[:10]:
                    section_html += f'<li>{item}</li>'
                if len(data) > 10:
                    section_html += f'<li>... and {len(data) - 10} more</li>'
                section_html += '</ul>'
            else:
                section_html += f'<p>{data}</p>'
            
            section_html += '</div>'
            sections_html.append(section_html)
        
        html_content = html_template.format(
            generation_time=self.generation_time,
            toc='\n'.join(toc_items),
            sections='\n'.join(sections_html)
        )
        
        # Lưu file
        filepath = os.path.join(self.output_dir, 'reports', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Saved HTML report to {filepath}")
        
        return html_content
    
    def generate_latex_report(self, filename: str = 'report.tex') -> str:
        """
        Tạo báo cáo dạng LaTeX.
        
        Parameters:
        -----------
        filename : str
            Tên file đầu ra
            
        Returns:
        --------
        str
            Nội dung LaTeX
        """
        latex_template = r"""
        \documentclass{article}
        \usepackage[utf8]{vietnam}
        \usepackage{geometry}
        \usepackage{graphicx}
        \usepackage{booktabs}
        \usepackage{longtable}
        \usepackage{hyperref}
        
        \geometry{a4paper, margin=1in}
        
        \title{BÁO CÁO DỰ ÁN\\ Dự đoán trả hàng TMĐT}
        \author{Nhóm ...}
        \date{\today}
        
        \begin{document}
        
        \maketitle
        \tableofcontents
        \newpage
        
        {content}
        
        \end{document}
        """
        
        content = []
        
        for section, data in self.report_data.items():
            content.append(f"\\section{{{section}}}")
            
            if isinstance(data, pd.DataFrame):
                content.append(f"\\begin{{longtable}}{{{'c' * len(data.columns)}}}")
                content.append("\\toprule")
                content.append(" & ".join(data.columns) + " \\\\")
                content.append("\\midrule")
                
                for _, row in data.head(20).iterrows():
                    content.append(" & ".join(str(v)[:50] for v in row.values) + " \\\\")
                
                if len(data) > 20:
                    content.append(f"... and {len(data) - 20} more rows \\\\")
                
                content.append("\\bottomrule")
                content.append("\\end{longtable}")
            else:
                content.append(f"\\begin{{verbatim}}")
                content.append(str(data))
                content.append(f"\\end{{verbatim}}")
        
        latex_content = latex_template.format(content='\n'.join(content))
        
        # Lưu file
        filepath = os.path.join(self.output_dir, 'reports', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        logger.info(f"Saved LaTeX report to {filepath}")
        
        return latex_content
    
    def get_report_data(self) -> Dict:
        """
        Lấy toàn bộ dữ liệu báo cáo.
        
        Returns:
        --------
        dict
            Dữ liệu báo cáo
        """
        return self.report_data