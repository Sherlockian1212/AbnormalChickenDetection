import cv2
import numpy as np
import h5py
import os
from filtering import *

def detectMoving(frames, rois):
    prev = frames[0]
    prev_flows = [None] * len(rois)
    # Danh sách lưu trữ vector optical flow của tất cả ROIs
    vector_optical_flows = [[] for _ in range(len(rois))]

    i = 0
    for frame in frames[1:]:
        flows, frame_rois = calOpticalFlow(frame, prev, rois, prev_flows)
        # Xử lý từng ROI
        for j, (flow, frame_roi) in enumerate(zip(flows, frame_rois)):
            # Thêm flow vào danh sách tương ứng với ROI j
            vector_optical_flows[j].append(np.copy(flow))
            image = convertToRGB(flow, frame_roi)
            cv2.imwrite(rf'flow_image_roi_{j}_{i}.jpg', image)  # Lưu ảnh flow
            cv2.imwrite(rf'normal_image_roi_{j}_{i}.jpg', frame_roi)  # Lưu ảnh gốc của ROI

        prev_flows = flows
        prev = frame
        i += 1

    # Lưu tất cả các optical flow vào file HDF5, mỗi ROI một file
    for k, f in enumerate(vector_optical_flows):
        saveHDF5(f, f"vector_optical_flow_roi_{k}")


def calOpticalFlow(newFrame, oldFrame, rois, prev_flows):
    flows = []
    frames = []
    for (x, y, w, h), prev_flow in zip(rois, prev_flows):
        # Extract the ROI
        roiOld = oldFrame[y:y + h, x:x + w]
        prvs = cv2.cvtColor(roiOld, cv2.COLOR_BGR2GRAY)

        roiNew = newFrame[y:y + h, x:x + w]
        next = cv2.cvtColor(roiNew, cv2.COLOR_BGR2GRAY)


        flow = cv2.calcOpticalFlowFarneback(prvs, next, prev_flow,
                                            pyr_scale=0.5,
                                            levels=4,
                                            winsize=10,
                                            iterations=5,
                                            poly_n=5,
                                            poly_sigma=1.0,
                                            flags=cv2.OPTFLOW_USE_INITIAL_FLOW if prev_flow is not None else 0)
        # filtered_flow = filterMean(flow)
        # flows.append(filtered_flow)
        flows.append(flow)
        frames.append(roiNew)
    return flows, frames

def saveHDF5(flows, name):
    path = rf"{name}.h5"
    if len(flows) > 0:
        height, width = flows[0].shape[:2]
    with h5py.File(path, 'w') as file:
        for i, flow in enumerate(flows):
            file.create_dataset(f'optical_flow_{i}', data=flow)
        file.create_dataset(f'length', data=len(flows))
        file.create_dataset(f'height', data=height)
        file.create_dataset(f'width', data=width)

def convertToRGB(flow, frame = 1):
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

def readVideoFrames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (count % 10 == 0):
            frames.append(frame)
        count += 1
    cap.release()
    return frames

video_path = '../Input/video (5).mp4'  # Thay đổi đường dẫn đến video của bạn
frames = readVideoFrames(video_path)
directory = "../Output/Dense optical flow/"
# Chuyển vào thư mục đó
os.chdir(directory)

# Kiểm tra thư mục hiện tại
print("Current directory:", os.getcwd())
flow_frame = detectMoving(frames, [(641, 154, 403, 359)])