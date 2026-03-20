# 🛒 E-commerce Returns Prediction System (ERPS)
> **Dự án Khai phá Dữ liệu Lớn: Dự đoán hành vi hoàn trả hàng trong Thương mại điện tử**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/University-Dai_Nam-red.svg?style=for-the-badge">
  <img src="https://img.shields.io/badge/Model-XGBoost%20%26%20RandomForest-green.svg?style=for-the-badge">
  <img src="https://img.shields.io/badge/F1--Score-0.9946-orange.svg?style=for-the-badge">
</p>

---

## 📖 1. Đặt vấn đề & Mục tiêu (Context)
Trong kỷ nguyên TMĐT, tỉ lệ hoàn hàng (Product Returns) dao động từ 15-30%. Điều này gây tổn thất hàng tỷ đồng cho doanh nghiệp do chi phí logistics ngược và hỏng hóc hàng hóa.
**Dự án này giải quyết:**
- Dự đoán khả năng trả hàng ngay tại thời điểm đặt hàng.
- Phân khúc khách hàng để có chiến lược chăm sóc riêng.
- Tìm ra các "mẫu hành vi" dẫn đến việc hoàn hàng thông qua luật kết hợp.

---

## 🛠 2. Kiến trúc dữ liệu & Tiền xử lý (Data Engineering)

### 2.1. Cấu trúc tập dữ liệu
Dữ liệu bao gồm 1000+ bản ghi với các nhóm thuộc tính:
- **Nhóm Giao dịch:** Order_ID, Order_Date, Amount, Payment_Method.
- **Nhóm Logistics:** Delivery_Status, Shipping_Speed, Carrier_ID.
- **Nhóm Khách hàng:** Customer_ID, Location, Tenure.

### 2.2. Quy trình xử lý (Pipeline)
1. **Cleaning:** Loại bỏ nhiễu và xử lý giá trị khuyết thiếu (Null values).
2. **Outlier Detection:** Sử dụng phương pháp **IQR (Interquartile Range)** để loại bỏ các đơn hàng có giá trị bất thường.
3. **Feature Engineering:** Tạo các biến phái sinh quan trọng:
   - `Return_Probability`: Tỉ lệ trả hàng của khách trong 6 tháng gần nhất.
   - `High_Risk_Category`: Gán nhãn các mặt hàng dễ vỡ/giá trị cao.
4. **Data Balancing:** Áp dụng **SMOTE** (Synthetic Minority Over-sampling Technique) để xử lý tình trạng mất cân bằng dữ liệu (chỉ có 7.8% đơn hàng trả lại trong tập gốc).

---

## 🤖 3. Huấn luyện & Đánh giá mô hình (Model Performance)

Chúng tôi đã thử nghiệm 5 mô hình và kết quả cho thấy các thuật toán Boosting chiếm ưu thế tuyệt đối:

### 3.1. Bảng so sánh chỉ số (Trang 24)
| Thuật toán | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | **0.9946** | **0.9893** | **1.0000** | **0.9946** |
| Random Forest | 0.9920 | 0.9841 | 0.9980 | 0.9910 |
| LightGBM | 0.9850 | 0.9750 | 0.9900 | 0.9824 |
| Logistic Regression | 0.8540 | 0.8210 | 0.8650 | 0.8423 |

### 3.2. Ma trận nhầm lẫn (Confusion Matrix)
Phân tích ma trận nhầm lẫn cho thấy mô hình XGBoost không bỏ sót bất kỳ đơn hàng "Return" nào (False Negative = 0).

<p align="center">
  <img src="images/confusion_matrix.png" width="800">
  <br><i>Hình: Đối sánh Confusion Matrix giữa XGBoost và Random Forest (Trang 25)</i>
</p>
<img width="683" height="601" alt="image" src="https://github.com/user-attachments/assets/2b48aaf9-44dd-4b10-ada2-5419feeb1f2c" />

---

## 💡 4. Khai phá tri thức & Insights (Knowledge Discovery)

### 4.1. Luật kết hợp (Association Rules - Apriori)
Từ 50+ luật được tìm thấy, có 2 luật quan trọng nhất:
- **Luật 1:** `Nếu {Phương thức: COD, Ngành hàng: Điện tử} => {Khả năng trả hàng: 85%}` (Lift: 4.2).
- **Luật 2:** `Nếu {Giao hàng: Express, Cuối tuần} => {Khả năng hủy đơn: cao}`.

### 4.2. Phân cụm khách hàng (K-Means)
- **Cụm 1 (Vàng):** Khách hàng VIP, giá trị đơn hàng cao, tỉ lệ hoàn hàng < 1%.
- **Cụm 2 (Đỏ):** Khách hàng rủi ro, thường xuyên săn sale và hoàn hàng sau khi nhận.

---

## 🚀 5. Hướng dẫn sử dụng
1. **Môi trường:** Python 3.9+, Anaconda/Jupyter Notebook.
2. **Cài đặt thư viện:**
   ```bash
   pip install pandas numpy scikit-learn xgboost mlxtend matplotlib seaborn
