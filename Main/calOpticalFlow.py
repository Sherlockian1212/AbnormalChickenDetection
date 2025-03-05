import cv2
import numpy as np

def calOpticalFlow(newFrame, oldFrame, rois, prev_flows):
    flows = []
    frames = []
    for (x, y, w, h), prev_flow in zip(rois, prev_flows):
        # Extract the ROI
        roiOld = oldFrame[y:y + h, x:x + w]
        prvs = cv2.cvtColor(roiOld, cv2.COLOR_BGR2GRAY)

        roiNew = newFrame[y:y + h, x:x + w]
        next = cv2.cvtColor(roiNew, cv2.COLOR_BGR2GRAY)

        del roiOld
        flow = cv2.calcOpticalFlowFarneback(prvs, next, prev_flow,
                                            pyr_scale=0.5,
                                            levels=4,
                                            winsize=10,
                                            iterations=10,
                                            poly_n=5,
                                            poly_sigma=1.0,
                                            flags=cv2.OPTFLOW_USE_INITIAL_FLOW if prev_flow is not None else 0)
        filtered_flow = filterMandA(flow)
        # filtered_flow = filterSize(filtered_flow)
        flows.append(filtered_flow)
        frames.append(roiNew)
    return flows, frames


def filterMandA(flow, top_percent=10):
    height, width, _ = flow.shape
    magnitude, angles = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Đếm số lượng vector trong từng khoảng góc (36 khoảng, mỗi khoảng 10 độ)
    angle_bins = 36
    angle_step = 10  # Mỗi khoảng 10 độ
    angle_counts = np.zeros(angle_bins, dtype=int)

    for i in range(angle_bins):
        start_angle = i * angle_step * np.pi / 180
        end_angle = (i + 1) * angle_step * np.pi / 180
        angle_mask = (angles >= start_angle) & (angles < end_angle)
        angle_counts[i] = np.sum(angle_mask)

    # Tính trung bình số lượng
    average_count = np.mean(angle_counts)

    threshold = np.percentile(magnitude, 100 - top_percent)

    # Tạo mask để giữ lại vector thỏa mãn ít nhất một điều kiện
    mask = magnitude > threshold

    # Thêm vào mask các vector có góc có số lượng nhỏ hơn trung bình
    for i in range(angle_bins):
        if angle_counts[i] < average_count:  # Góc có số lượng nhỏ hơn trung bình
            start_angle = i * angle_step * np.pi / 180
            end_angle = (i + 1) * angle_step * np.pi / 180
            mask |= (angles >= start_angle) & (angles < end_angle)

    # Lọc flow
    filtered_flow = np.zeros_like(flow)
    filtered_flow[mask] = flow[mask]

    return filtered_flow


# def filterMandA(flow, top_percent=10):
#     height, width, _ = flow.shape
#     magnitude, angles = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#
#     # Đếm số lượng vector trong từng khoảng góc
#     angle_counts = np.zeros(360, dtype=int)
#     for angle in range(360):
#         angle_mask = (angles >= angle * np.pi / 180) & (angles < (angle + 1) * np.pi / 180)
#         angle_counts[angle] = np.sum(angle_mask)
#     # Tính trung bình số lượng
#     average_count = np.mean(angle_counts)
#
#     threshold = np.percentile(magnitude, 100 - top_percent)
#     # Tạo mask để giữ lại vector thỏa mãn ít nhất một điều kiện
#     mask = magnitude > threshold
#     # Thêm vào mask các vector có góc có số lượng nhỏ hơn trung bình
#     for angle in range(360):
#         if angle_counts[angle] < average_count:  # Góc có số lượng nhỏ hơn trung bình
#             mask |= (angles >= angle * np.pi / 180) & (angles < (angle + 1) * np.pi / 180)
#     # Lọc flow
#     filtered_flow = np.zeros_like(flow)
#     filtered_flow[mask] = flow[mask]
#
#     return filtered_flow

def filterSize(flow):
    magnitude, angles = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    total_pixels = flow.shape[0] * flow.shape[1]

    # Tính toán min_size dựa trên tỷ lệ phần trăm
    min_size = int(total_pixels * 0.01)

    # Ngưỡng magnitude để phân đoạn
    _, binary_mask = cv2.threshold(magnitude, 1, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)

    # Tìm các vùng kết nối (connected components)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # Tạo mặt nạ mới chỉ giữ lại các vùng có kích thước lớn hơn min_size
    filtered_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):  # Bỏ qua label 0 (background)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered_mask[labels == i] = 255

    # Áp dụng mặt nạ vào cả magnitude và angle để giữ lại các vùng mong muốn
    filtered_flow = np.zeros_like(flow)
    filtered_flow[..., 0] = cv2.bitwise_and(flow[..., 0], flow[..., 0], mask=filtered_mask)
    filtered_flow[..., 1] = cv2.bitwise_and(flow[..., 1], flow[..., 1], mask=filtered_mask)

    return filtered_flow




