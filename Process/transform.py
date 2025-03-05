import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os
import cv2

# Hàm để đọc dữ liệu từ file HDF5
def loadHDF5(file_path):
    with h5py.File(file_path, 'r') as file:
        # Đọc dữ liệu về chiều cao và chiều rộng
        height = file['height'][()]
        width = file['width'][()]
        # Đọc số lượng khung hình
        num_flows = file['length'][()]

        # Đọc tất cả các dataset optical flow
        flows = []
        for i in range(num_flows):
            flow = file[f'optical_flow_{i}'][:]
            flows.append(flow)

        return flows, height, width


# Hàm để tính magnitude của optical flow
def calculateMagnitude(flow, threshold=1e-3):
    magnitude,_ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    magnitude[magnitude < threshold] = 0
    return magnitude


# Hàm để phân tích tần số với biến đổi Fourier
def analyzeFrequency(magnitudes, sampling_rate):
    N = len(magnitudes)
    yf = fft(magnitudes)
    xf = fftfreq(N, 1 / sampling_rate)

    # Chọn phần tần số tích cực
    xf = xf[:N // 2]
    yf = 2.0 / N * np.abs(yf[:N // 2])

    return xf, yf

def saveImage(xf, yf, output_image_path):
    plt.plot(xf, yf)
    plt.title(f'Biến đổi Fourier của tín hiệu magnitude')
    plt.xlabel('Tần số (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)

    # Lưu lại hình ảnh với tên file tương ứng
    plt.savefig(output_image_path)
    plt.close()

def peakFrequency(file_path):
    flows, height, width = loadHDF5(file_path)
    magnitudes = [calculateMagnitude(flow) for flow in flows]
    average_magnitudes = [np.mean(magnitude) for magnitude in magnitudes]
    centered_magnitudes = average_magnitudes - np.mean(average_magnitudes)
    sampling_rate = 15  # fps
    xf, yf = analyzeFrequency(centered_magnitudes, sampling_rate)
    peak_frequency = xf[np.argmax(yf)]
    return peak_frequency


# # Đọc dữ liệu từ file HDF5
# file_path = r"D:\STUDY\DHSP\KLTN-2024-2025-With my idol\Source_Code\HeatStressDetection\Output\video 5\f_0\vector_optical_flow_roi_0.h5"  # Đổi tên file theo đúng đường dẫn của bạn
# flows, height, width = loadHDF5(file_path)
#
# # Tính toán magnitude cho mỗi khung hình
# magnitudes = [calculateMagnitude(flow) for flow in flows]
#
# # Tổng hợp giá trị magnitude theo khung hình
# average_magnitudes = [np.mean(magnitude) for magnitude in magnitudes]
#
# centered_magnitudes = average_magnitudes - np.mean(average_magnitudes)
#
# # Phân tích tần số
# sampling_rate = 15  # fps
# xf, yf = analyzeFrequency(centered_magnitudes, sampling_rate)
#
# # Vẽ đồ thị FFT
# plt.plot(xf, yf)
# plt.title('Biến đổi Fourier của tín hiệu magnitude')
# plt.xlabel('Tần số (Hz)')
# plt.ylabel('Magnitude')
# plt.grid(True)
# plt.show()
#
# # Tìm tần số có magnitude lớn nhất
# peak_frequency = xf[np.argmax(yf)]
# print(f"Tần số dao động chính: {peak_frequency:.2f} Hz")
