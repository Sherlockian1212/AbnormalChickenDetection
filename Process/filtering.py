import cv2
import numpy as np

def calculateVectorStd(magnitude):
    return np.std(magnitude)

def filterMagnitudes(flow, min_threshold=5.0, max_threshold=20.0):
    magnitudes, angles = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask = (magnitudes > min_threshold) & (magnitudes < max_threshold)
    filtered_flow = np.zeros_like(flow)
    filtered_flow[mask] = flow[mask]
    return filtered_flow

def filterVectorsByStd(flow, std_multiplier=0.1):
    magnitudes, angles = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    std_dev = calculateVectorStd(magnitudes)
    mean_magnitude = np.mean(magnitudes)

    threshold = mean_magnitude + std_multiplier * std_dev

    # Lọc các vector có magnitude nhỏ hơn ngưỡng
    mask = magnitudes > threshold
    filtered_flow = flow.copy()

    filtered_flow[~mask] = 0
    return filtered_flow

def filterMean(flow):
    magnitudes, angles = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_magnitude = np.mean(magnitudes)
    mask = magnitudes > mean_magnitude  # Giữ lại những vector lớn hơn giá trị trung bình
    filtered_flow = np.zeros_like(flow)
    filtered_flow[mask] = flow[mask]
    return filtered_flow

def regionGrowing(flow, threshold=0.2, max_distance=10):
    height, width, _ = flow.shape

    magnitude, angles = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Tạo mask cho góc từ tím đến đỏ
    # Tím: 240° - 270° (4/3π đến 3/2π) => 120 đến 135 (trong HSV)
    # Đỏ: 0° - 30° => 0 đến 15 hoặc 345 đến 360 (trong HSV)
    # Tạo mask cho góc từ 30° đến 240°
    purple_red_mask = (
            (angles >= 0) & (angles <= np.pi / 6) |  # Từ 0° đến 30°
            (angles >= 300 * np.pi / 180) & (angles <= 2 * np.pi) |  # Từ 300° đến 360°
            (angles >= 170 * np.pi / 180) & (angles <= 210 * np.pi / 180)  # Từ 170° đến 210°
    )
    # Tìm seed point là vector có magnitude lớn nhất trong vùng màu tím đến đỏ
    seed_point = np.unravel_index(np.argmax(magnitude * purple_red_mask), magnitude.shape)

    # Khởi tạo vùng (region) với seed point
    region = np.zeros_like(magnitude, dtype=bool)
    region[seed_point] = True

    # Tạo mask kiểm tra xem điểm nào đã được thêm vào vùng
    visited = np.zeros_like(magnitude, dtype=bool)
    visited[seed_point] = True

    # Hàng đợi để lưu các điểm sẽ mở rộng tiếp theo
    queue = [seed_point]

    while queue:
        current_point = queue.pop(0)
        y, x = current_point

        # Lấy các điểm lân cận của current_point
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    # Kiểm tra xem điểm này đã được thăm chưa và có nằm trong khoảng cách tối đa không
                    if not visited[ny, nx] and np.sqrt(dy ** 2 + dx ** 2) <= max_distance:
                        # Kiểm tra điều kiện threshold để thêm vào vùng
                        if abs(magnitude[ny, nx] - magnitude[y, x]) <= threshold and purple_red_mask[ny, nx]:
                            region[ny, nx] = True
                            visited[ny, nx] = True
                            queue.append((ny, nx))

    # Tạo output flow chỉ giữ lại các vector trong vùng mở rộng
    output_flow = np.zeros_like(flow)
    output_flow[region] = flow[region]

    return output_flow, seed_point

def filterMandA(flow, top_percent=3):

    height, width, _ = flow.shape

    magnitude, angles = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Đếm số lượng vector trong từng khoảng góc
    angle_counts = np.zeros(360, dtype=int)

    for angle in range(360):
        angle_mask = (angles >= angle * np.pi / 180) & (angles < (angle + 1) * np.pi / 180)
        angle_counts[angle] = np.sum(angle_mask)

    # Tính trung bình số lượng
    average_count = np.mean(angle_counts)

    threshold = np.percentile(magnitude, 100 - top_percent)

    # Tạo mask để giữ lại vector thỏa mãn ít nhất một điều kiện
    mask = magnitude > threshold  # Vector có độ lớn lớn hơn trung bình

    # Thêm vào mask các vector có góc có số lượng nhỏ hơn trung bình
    for angle in range(360):
        if angle_counts[angle] < average_count:  # Góc có số lượng nhỏ hơn trung bình
            mask |= (angles >= angle * np.pi / 180) & (angles < (angle + 1) * np.pi / 180)

    # Lọc flow
    filtered_flow = np.zeros_like(flow)
    filtered_flow[mask] = flow[mask]

    return filtered_flow


def filterAngle(flow):
    height, width, _ = flow.shape

    magnitude, angles = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Đếm số lượng vector trong từng khoảng góc
    angle_counts = np.zeros(360, dtype=int)

    for angle in range(360):
        angle_mask = (angles >= angle * np.pi / 180) & (angles < (angle + 1) * np.pi / 180)
        angle_counts[angle] = np.sum(angle_mask)

    # Tính trung bình số lượng
    average_count = np.mean(angle_counts)

    # Tạo mask cho những góc lớn hơn trung bình
    mask = np.zeros_like(magnitude, dtype=bool)

    # Giữ lại góc có số lượng nhỏ hơn hoặc bằng trung bình
    for angle in range(360):
        if angle_counts[angle] <= average_count:
            mask |= (angles >= angle * np.pi / 180) & (angles < (angle + 1) * np.pi / 180)
    filtered_flow = np.zeros_like(flow)
    filtered_flow[mask] = flow[mask]

    return filtered_flow

from scipy.ndimage import convolve
def keepVectorsAround(flow, filtered_flow):
    height, width, _ = flow.shape

    # Tạo mask cho các pixel có vector trong filtered_flow
    mask = (filtered_flow[..., 0] != 0) | (filtered_flow[..., 1] != 0)

    # Tạo kernel cho convolution để đếm số vector xung quanh
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]])  # Chỉ xem xét 8 pixel xung quanh

    # Đếm số lượng vector xung quanh mỗi pixel
    vector_counts = convolve(mask.astype(int), kernel, mode='constant', cval=0)

    # Tạo output flow và ghi lại các vector gốc cho các pixel không trống
    updated_flow = np.zeros_like(flow)
    updated_flow[mask] = filtered_flow[mask]

    # Giữ lại các vector trong filtered_flow nếu có ít nhất 4 vector xung quanh
    lonely_pixels = (flow[..., 0] == 0) & (flow[..., 1] == 0)
    updated_flow[lonely_pixels & (vector_counts >= 4)] = filtered_flow[lonely_pixels & (vector_counts >= 4)]

    return updated_flow

def convertToRGB(flow):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Create mask
    height, width = flow.shape[:2]
    hsv_mask = np.zeros((height, width, 3), dtype=np.uint8)
    # Make image saturation to a maximum value
    hsv_mask[..., 1] = 255
    # Set image hue value according to the angle of optical flow
    hsv_mask[..., 0] = ang * 180 / np.pi / 2
    # Set value as per the normalized magnitude of optical flow
    hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
    return rgb_representation

# from transform import *
#
# file_path = r"D:\STUDY\DHSP\KLTN-2024-2025-With my idol\Source_Code\HeatStressDetection\Output\video 4\f_29\vector_optical_flow_roi_0.h5"
# flows, height, width = loadHDF5(file_path)
#
# for i, flow in enumerate(flows):
#     originImage = convertToRGB(flow)
#
#     filteringFlow = filterMandA(flow)
#     interFlow= keepVectorsAround(flow, filteringFlow)
#     fiteringImage = convertToRGB(interFlow)
#
#     cv2.imwrite(rf'origin_image_{i}.jpg', originImage)
#     cv2.imwrite(rf'filtering_image_roi_{i}.jpg', fiteringImage)
