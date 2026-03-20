# DỰ ÁN DỰ ĐOÁN TRẢ HÀNG TMĐT

## Thông tin dự án
- **Tên dự án**: E-commerce Returns Prediction
- **Mã đề tài**: 13
- **Ngày hoàn thành**: 19/03/2026

## Cấu trúc thư mục
```
├── data/                 # Dữ liệu
│   ├── raw/             # Dữ liệu gốc
│   └── processed/       # Dữ liệu đã xử lý
├── notebooks/           # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_preprocess_feature.ipynb
│   ├── 03_mining_clustering.ipynb
│   ├── 04_modeling_supervised.ipynb
│   ├── 04b_semi_supervised.ipynb
│   └── 05_evaluation_report.ipynb
├── src/                 # Mã nguồn
├── outputs/             # Kết quả đầu ra
│   ├── figures/         # Biểu đồ
│   ├── tables/          # Bảng kết quả
│   ├── models/          # Models đã train
│   └── reports/         # Báo cáo
└── configs/             # Cấu hình
```

## Kết quả chính
- **Model tốt nhất**: Random Forest (Tuned)
- **Số cụm khách hàng**: 4
- **Số features quan trọng**: 76

## Hướng dẫn chạy lại
1. Cài đặt requirements: `pip install -r requirements.txt`
2. Cập nhật đường dẫn trong `configs/params.yaml`
3. Chạy các notebook theo thứ tự từ 01 đến 05
4. Xem kết quả trong `outputs/reports/final_report.html`

## Thành viên nhóm
- Thành viên 1 - MSSV
- Thành viên 2 - MSSV
- Thành viên 3 - MSSV
- Thành viên 4 - MSSV

## Giảng viên hướng dẫn
ThS. Lê Thị Thùy Trang