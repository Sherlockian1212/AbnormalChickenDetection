from calOpticalFlow import *
from processHDF5 import *
from analyzeFrequency import *
import os


def pantingDetection(frames, rois, FPS=15, min_prequency=3, save_image = False, save_HDF5 = False, save_frequency = False, path = ''):
    prev = frames[0]

    have_rois, miss_location = [], []
    for l,roi in enumerate(rois):
        if roi == ():
            miss_location.append(l)
        else:
            have_rois.append(roi)

    prev_flows = [None] * len(have_rois)
    vector_optical_flows = [[] for _ in range(len(have_rois))]

    i = 0
    for frame in frames[1:]:
        flows, frame_rois = calOpticalFlow(frame, prev, have_rois, prev_flows)
        # Xử lý từng ROI
        for j, (flow, frame_roi) in enumerate(zip(flows, frame_rois)):
            # Thêm flow vào danh sách tương ứng với ROI j
            vector_optical_flows[j].append(np.copy(flow))
            if save_image:
                image = convertToRGB(flow)
                flow_image_path = os.path.join(path, rf'flow_image_roi_{j}_{i}.jpg')
                normal_image_path = os.path.join(path, rf'normal_image_roi_{j}_{i}.jpg')
                cv2.imwrite(flow_image_path, image)  # Lưu ảnh flow
                cv2.imwrite(normal_image_path, frame_roi)  # Lưu ảnh gốc của ROI

        prev_flows = flows
        prev = frame
        i += 1

    list_R2, list_peak_frequency, list_filtered_count, list_total_count = [], [], [], []

    for index, vector_flow in enumerate(vector_optical_flows):
        peak_frequency, R2 = peakFrequencyByRange(vector_flow, FPS, min_prequency, save_frequency, path)
        list_R2.append(R2)
        print(f"ROI {index}: Peak Frequency: {peak_frequency} Hz; Accuracy: {R2}")
        # if peak_frequency>=min_prequency:
        #     print("Panting")
        # else:
        #     print("Not Panting")

    for loc in sorted(miss_location):
        list_R2.insert(loc, 0)

    if save_HDF5:
        # Lưu tất cả các optical flow vào file HDF5, mỗi ROI một file
        for k, f in enumerate(vector_optical_flows):
            HDF5_path = os.path.join(path, f"vector_optical_flow_roi_{k}.h5")
            saveHDF5(f, HDF5_path)

    return list_R2