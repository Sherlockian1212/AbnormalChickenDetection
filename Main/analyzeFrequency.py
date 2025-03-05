from videoAndImageProcess import *
from calAccuracy import *
from scipy.fft import fft, fftfreq
import numpy as np
import os
import cv2

def analyzeFrequency(magnitudes, sampling_rate):
    N = len(magnitudes)
    yf = fft(magnitudes)
    xf = fftfreq(N, 1 / sampling_rate)

    # Chọn phần tần số tích cực
    xf = xf[:N // 2]
    yf = 2.0 / N * np.abs(yf[:N // 2])

    return xf, yf

def calculateMagnitude(flow):
    magnitude,_ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return magnitude

def peakFrequency(flows, sampling_rate=15, min_prequency=3, save=False, path=''):
    # Lọc các flow không có giá trị vector (magnitude bằng 0)
    valid_magnitudes = []
    for flow in flows:
        magnitude = calculateMagnitude(flow)
        if np.any(magnitude > 0):  # Kiểm tra xem có vector nào khác 0 không
            valid_magnitudes.append(np.mean(magnitude))
    print(f"Flow hợp lệ: {len(valid_magnitudes)} / {len(flows)}")
    # Kiểm tra nếu không có flow hợp lệ
    if len(valid_magnitudes) == 0:
        print("Không có flow hợp lệ để tính toán.")
        return 0, 0

    # Chuẩn hóa tín hiệu
    centered_magnitudes = valid_magnitudes - np.mean(valid_magnitudes)

    # Phân tích tần số
    xf, yf = analyzeFrequency(centered_magnitudes, sampling_rate)

    # Lưu hình ảnh nếu cần
    if save:
        save_path = os.path.join(path, "FrequencyImage.png")
        saveFrequencyImage(xf, yf, save_path)

    # Tìm tần số có magnitude lớn nhất
    if yf.size == 0:
        peak_frequency = 0
    else:
        peak_frequency = xf[np.argmax(yf)]

    R2 = calAccuracyStep2(len(valid_magnitudes), len(flows), peak_frequency, min_prequency)
    return peak_frequency, R2


def peakFrequencyByRange(flows, sampling_rate=15, min_prequency=3, save=False, path=''):
    # Lọc các flow không có giá trị vector (magnitude bằng 0)
    valid_magnitudes = []
    for flow in flows:
        magnitude = calculateMagnitude(flow)
        if np.any(magnitude > 0):  # Kiểm tra xem có vector nào khác 0 không
            valid_magnitudes.append(np.mean(magnitude))
    print(f"Flow hợp lệ: {len(valid_magnitudes)} / {len(flows)}")

    # Kiểm tra nếu không có flow hợp lệ
    if len(valid_magnitudes) == 0:
        print("Không có flow hợp lệ để tính toán.")
        return 0, 0

    # Chuẩn hóa tín hiệu
    centered_magnitudes = valid_magnitudes - np.mean(valid_magnitudes)

    # Phân tích tần số
    xf, yf = analyzeFrequency(centered_magnitudes, sampling_rate)

    # Lưu hình ảnh nếu cần
    if save:
        save_path = os.path.join(path, "FrequencyImage.png")
        saveFrequencyImage(xf, yf, save_path)

    # Tính tổng magnitude trong từng khoảng
    sum_0_2 = np.sum(yf[(xf >= 0) & (xf < 2)])  # Tổng magnitude trong khoảng 0-2 Hz
    sum_2_5 = np.sum(yf[(xf >= 2) & (xf < 5)])  # Tổng magnitude trong khoảng 2-5 Hz
    sum_above_5 = np.sum(yf[xf >= 5])            # Tổng magnitude trong khoảng >5 Hz

    print(f"Tổng magnitude trong khoảng 0-2 Hz: {sum_0_2}")
    print(f"Tổng magnitude trong khoảng 2-5 Hz: {sum_2_5}")
    print(f"Tổng magnitude trong khoảng >5 Hz: {sum_above_5}")

    # Xác định khoảng nào có tổng magnitude cao nhất
    if sum_0_2 >= sum_2_5 and sum_0_2 >= sum_above_5:
        # Nếu khoảng 0-2 Hz có tổng magnitude cao nhất
        yf_0_2 = yf[(xf >= 0) & (xf < 2)]
        xf_0_2 = xf[(xf >= 0) & (xf < 2)]
        if len(yf_0_2) > 0:
            peak_frequency = np.mean(xf_0_2)  # Tính tần số trung bình
        else:
            peak_frequency = 0
    elif sum_2_5 >= sum_0_2 and sum_2_5 >= sum_above_5:
        # Nếu khoảng 2-5 Hz có tổng magnitude cao nhất
        yf_2_5 = yf[(xf >= 2) & (xf < 5)]
        xf_2_5 = xf[(xf >= 2) & (xf < 5)]
        if len(yf_2_5) > 0:
            peak_frequency = np.mean(xf_2_5)  # Tính tần số trung bình
        else:
            peak_frequency = 0
    else:
        # Nếu khoảng >5 Hz có tổng magnitude cao nhất
        yf_above_5 = yf[xf >= 5]
        xf_above_5 = xf[xf >= 5]
        if len(yf_above_5) > 0:
            peak_frequency = np.mean(xf_above_5)  # Tính tần số trung bình
        else:
            peak_frequency = 0

    R2 = calAccuracyStep2(len(valid_magnitudes), len(flows), peak_frequency, min_prequency)
    return peak_frequency, R2