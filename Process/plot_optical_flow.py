import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from transform import *
from filtering import *
from dense_optical_flow import *

# Hàm để tính trung bình của các khối 4x4 pixel
def averageBlocks(magnitude, block_size=16):
    height, width = magnitude.shape

    # Tính số khối có thể chứa trong kích thước hiện tại
    num_blocks_y = height // block_size
    num_blocks_x = width // block_size

    # Khởi tạo mảng trung bình khối
    averaged_magnitude = np.zeros((num_blocks_y, num_blocks_x))

    # Tính trung bình cho từng khối
    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            block = magnitude[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size]
            averaged_magnitude[y, x] = np.mean(block)

    return averaged_magnitude


# Hàm để vẽ biểu đồ cột cho các giá trị magnitude trung bình qua từng frame
def plotBarChart(magnitudes):
    frames = np.arange(len(magnitudes))

    plt.figure(figsize=(10, 6))
    plt.bar(frames, magnitudes, color='skyblue')
    plt.title('Magnitude Distribution of Optical Flow per Frame')
    plt.xlabel('Block Index')
    plt.ylabel('Average Magnitude')
    plt.grid(True)
    plt.show()


# Hàm để lưu biểu đồ cột
def saveBarChart(magnitudes, path):
    frames = np.arange(len(magnitudes))

    plt.figure(figsize=(10, 6))
    plt.bar(frames, magnitudes, color='skyblue')
    plt.title('Magnitude Distribution of Optical Flow per Frame')
    plt.xlabel('Block Index')
    plt.ylabel('Average Magnitude')
    plt.grid(True)

    # Lưu biểu đồ dưới dạng tệp hình ảnh
    plt.savefig(f"{path}.png")
    plt.close()


# Hàm để xử lý optical flow và lưu biểu đồ
def plotOpticalFlow(file_path):
    flows, height, width = loadHDF5(file_path)
    magnitudes = [calculateMagnitude(flow) for flow in flows]

    for i, magnitude in enumerate(magnitudes):
        print(np.max(magnitude))
        print(np.min(magnitude))

        # Trung bình magnitude theo các khối 4x4
        averaged_magnitude = averageBlocks(magnitude)

        # Làm phẳng mảng magnitude đã tính trung bình
        flat_magnitude = averaged_magnitude.flatten()

        # Lưu biểu đồ cột cho các giá trị magnitude
        saveBarChart(flat_magnitude, f"../Output/chart/video_27/f_4/roi_{i}")


# Gọi hàm với đường dẫn file HDF5
# file_path = r"D:\STUDY\DHSP\KLTN-2024-2025-With my idol\Source_Code\Practice\Dense optical flow\video 5\f_11\vector_optical_flow_roi_0.h5"
file_path = r"../Output/video 27\f_4\vector_optical_flow_roi_0.h5"
plotOpticalFlow(file_path)
