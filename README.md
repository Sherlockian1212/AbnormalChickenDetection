
# 🐔 Abnormal Chicken Detection

> **Khóa luận tốt nghiệp đại học ngành Khoa học Máy tính**  
> **Sinh viên thực hiện:** Thái Thị Kim Yến  
> **Trường:** Trường Đại học Sư phạm Thành phố Hồ Chí Minh  
> **Khoa:** Khoa Công nghệ Thông tin  
> **Niên khóa:** 2020–2024  
> **Giảng viên hướng dẫn:** PGS. TS. Tống Xuân Tám  

---

**Abnormal Chicken Detection** là một dự án ứng dụng thị giác máy tính (computer vision) và học sâu (deep learning) vào lĩnh vực chăn nuôi, nhằm phát hiện **gà có hành vi bất thường** trong chuồng thông qua phân tích video. Mục tiêu là hỗ trợ người chăn nuôi giám sát tự động tình trạng sức khỏe của gà, từ đó kịp thời can thiệp khi có bất thường xảy ra.

📌 **Dự án gồm 2 nhánh chính:**

- 🔴 **Dead Chicken Detection** – Phát hiện gà chết bằng YOLOv8  
- 🔥 **Heat-Stressed Chicken Detection** – Phát hiện gà bị stress nhiệt qua phân tích Optical Flow

---

## 📁 Cấu trúc thư mục

```bash
AbnormalChickenDetection/
├── DeadChicken/           # Nhánh phát hiện gà chết
├── HeatStressChicken/     # Nhánh phát hiện gà bị stress nhiệt
├── Weight/                # Trọng số mô hình
├── Frame/                 # Khung hình video đầu vào
├── Process/               # Dữ liệu trung gian
├── requirements.txt       # Thư viện cần thiết
└── README.md
```

---

## 🔥 Heat-Stressed Chicken Detection

Nhánh này sử dụng các kỹ thuật:
- Tách khung hình (frame extraction)
- Phát hiện vùng đầu và yếm của gà
- Tính toán **vector Optical Flow** để ghi nhận chuyển động
- Biến đổi Fourier để phân tích tần số thở  
→ Nếu tần số thở cao bất thường, hệ thống sẽ cảnh báo gà có dấu hiệu **stress nhiệt**.

🔧 **Cách chạy:**

```bash
cd HeatStressChicken
python main.py --video path/to/video.mp4
```

---

## 🔴 Dead Chicken Detection

Áp dụng mô hình **YOLOv8** để nhận diện hình ảnh gà chết trong chuồng. Mô hình đã được huấn luyện trên tập dữ liệu tuỳ chỉnh từ thực tế.

🔧 **Cách chạy:**

```bash
cd DeadChicken
python main.py --source path/to/image_or_video
```

> Mô hình `.pt` cần được đặt trong thư mục `Weight/`.

---

## ⚙️ Cài đặt

1. **Clone repo:**
```bash
git clone https://github.com/Sherlockian1212/AbnormalChickenDetection.git
cd AbnormalChickenDetection
```

2. **Tạo môi trường ảo (nếu cần):**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Cài đặt thư viện:**
```bash
pip install -r requirements.txt
```

---

## 🧪 Công nghệ sử dụng

- Python 3.10
- OpenCV
- NumPy
- Matplotlib
- Ultralytics YOLOv8
- SciPy (Fourier Transform)

---

## 📄 Giấy phép

Dự án phát hành dưới giấy phép MIT. Xem thêm tại [LICENSE](LICENSE).

---

🎓 *Dự án thực hiện với mục đích nghiên cứu học thuật. Dữ liệu và mô hình sử dụng đều mang tính thử nghiệm và có thể cần hiệu chỉnh thêm khi triển khai thực tế.*
