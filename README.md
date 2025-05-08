
# 🐔 Phát hiện trạng thái bất thường của gà dựa trên kết hợp mạng học sâu, luồng quang học và ứng dụng vào trang trại thực tế

> **Khóa luận tốt nghiệp đại học**  
> **Sinh viên thực hiện:** Thái Thị Kim Yến  
> **Trường:** Trường Đại học Sư phạm Thành phố Hồ Chí Minh  
> **Khoa:** Khoa Công nghệ Thông tin
> **Chuyên ngành:** Khoa học máy tính
> **Niên khóa:** 2021–2025  
> **Giảng viên hướng dẫn:** TS. Ngô Quốc Việt (Thầy Việt dễ thương)

---

**Phát hiện trạng thái bất thường của gà dựa trên kết hợp mạng học sâu, luồng quang học và ứng dụng vào trang trại thực tế** là một dự án ứng dụng thị giác máy tính (computer vision) và học sâu (deep learning) vào lĩnh vực chăn nuôi, nhằm phát hiện **gà có hành vi bất thường** trong chuồng thông qua phân tích video. Mục tiêu là hỗ trợ người chăn nuôi giám sát tự động tình trạng sức khỏe của gà, từ đó kịp thời can thiệp khi có bất thường xảy ra.

📌 **Dự án gồm 2 nhánh chính:**

- 🔴 **Dead Chicken Detection** – Phát hiện gà chết bằng YOLOv8  
- 🔥 **Heat-Stressed Chicken Detection** – Phát hiện gà bị stress nhiệt qua phân tích Optical Flow

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

Nghiên cứu sử dụng một mô hình học sâu để xác định vị trí của gà có nguy cơ chết thông qua hình ảnh nhiệt. Sau đó, áp dụng thuật toán luồng quang học để phát hiện các đối tượng đứng yên, nhằm đưa ra kết luận cuối cùng về việc có gà chết hay không.

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
