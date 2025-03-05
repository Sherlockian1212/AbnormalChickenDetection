from transform import *
import os
import re
import numpy as np

def pantingDetection(parent_dir):
    try:
        # Lấy danh sách các thư mục con trong thư mục cha
        dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
        dirs.sort()
    except FileNotFoundError:
        print(f"Parent directory not found: {parent_dir}")
        return

    peakFrequencyOfRoi = {}
    for folder_name in dirs:
        folder_path = os.path.join(parent_dir, folder_name)

        try:
            # Duyệt qua các file trong thư mục con
            for filename in os.listdir(folder_path):
                match = re.match(r'vector_optical_flow_roi_(\d+)', filename)
                if match:
                    index = int(match.group(1))
                    file_path = os.path.join(folder_path, filename)

                    # Khởi tạo danh sách cho index nếu chưa tồn tại
                    if index not in peakFrequencyOfRoi:
                        peakFrequencyOfRoi[index] = []

                    try:
                        # Tính peakFrequency cho file
                        peakFrequencyOfRoi[index].append(peakFrequency(file_path))
                    except FileNotFoundError:
                        print(f"File not found: {file_path}")
                        continue
        except FileNotFoundError:
            print(f"Subdirectory not found: {folder_path}")
            continue

    # Tính trung bình cho từng ROI
    group_size = 1
    for index, frequencies in peakFrequencyOfRoi.items():
        if frequencies:  # Kiểm tra nếu danh sách không rỗng
            averages = [np.mean(frequencies[i:i + group_size]) for i in range(0, len(frequencies), group_size)]
            print(f"ROI {index} has average Peak Frequencies: {averages}")
        else:
            print(f"ROI {index} has no data")




# # Duyệt qua các video từ 1 đến 30
# for i in range(30):
#     print(f"Video {i + 1}:")
#     pantingDetection(rf"D:\STUDY\DHSP\KLTN-2024-2025-With my idol\Source_Code\Practice\Dense optical flow\video {i + 1}")

pantingDetection(r"../Output/video 5")
# pantingDetection(rf"D:\STUDY\DHSP\KLTN-2024-2025-With my idol\Source_Code\Practice\Dense optical flow\video 5")

