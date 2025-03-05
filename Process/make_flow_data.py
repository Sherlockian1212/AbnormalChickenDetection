from Process.open_mouth import open_mouthDetection
from dense_optical_flow import *
from get_object import *

import cv2

def readVideoFrames(video_path, target_fps=15):
    cap = cv2.VideoCapture(video_path)
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    if original_fps == 0:
        print("Error: Could not read FPS from video.")
        return []
    sampling_interval = round(original_fps / target_fps)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (count % sampling_interval == 0):
            frames.append(frame)
        count += 1
    cap.release()
    return frames

def get_list_frame(video_path):
    frames = readVideoFrames(video_path)
    print(len(frames))
    num_frame = 15
    detected_frame = []
    temp = []
    for i, frame in enumerate(frames):
        if i % num_frame == 0 and i != 0:
            temp.append(frame)
            detected_frame.append(temp)
            temp = []
        else:
            temp.append(frame)
    return detected_frame

def detection(video_path, index):
    detected_frame = get_list_frame(video_path)
    print(len(detected_frame))

    for i,frames in enumerate(detected_frame):
        os.chdir("D:\STUDY\DHSP\KLTN-2024-2025-With my idol\Source_Code\HeatStressDetection\Process")
        print("Thư mục hiện tại:", os.getcwd())
        rois, cls_name, predict_image = open_mouthDetection().detect_image(frames[0])
        os.makedirs(f"../Output/video {index}", exist_ok=True)
        os.chdir(f"../Output/video {index}")
        print("Thư mục hiện tại:", os.getcwd())
        if rois:
            new_folder = fr"f_{i}"
            os.makedirs(new_folder, exist_ok=True)
            os.chdir(new_folder)
            # In đường dẫn hiện tại để kiểm tra
            print("Thư mục hiện tại:", os.getcwd())
            head_rois = get_head(rois, cls_name)
            detectMoving(frames, head_rois)

# import time
#
# start_time = time.time()
#
# for i in range(17):
#     detection(rf"D:\STUDY\DHSP\KLTN-2024-2025-With my idol\Dataset\heat stress\video\video ({i+1}).mp4", i+1)
#
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Thời gian thực thi: {execution_time} giây")

detection(rf"D:\STUDY\DHSP\KLTN-2024-2025-With my idol\Dataset\heat stress\video\video (4).mp4", 4)

# for i in range(30):
#     video_path = rf"D:\STUDY\DHSP\KLTN-2024-2025-With my idol\Dataset\heat stress\video\video ({i+1}).mp4"
#     cap = cv2.VideoCapture(video_path)
#     print(f"Video {i+1} has fps: {int(cap.get(cv2.CAP_PROP_FPS))}")
'''
Video 1 has fps: 60
Video 2 has fps: 29
Video 3 has fps: 30
Video 4 has fps: 30
Video 5 has fps: 30
Video 6 has fps: 30
Video 7 has fps: 25
Video 8 has fps: 30
Video 9 has fps: 30
Video 10 has fps: 30
Video 11 has fps: 29
Video 12 has fps: 30
Video 13 has fps: 30
Video 14 has fps: 29
Video 15 has fps: 29
Video 16 has fps: 30
Video 17 has fps: 30
Video 18 has fps: 59
Video 19 has fps: 30
Video 20 has fps: 30
Video 21 has fps: 30
Video 22 has fps: 30
Video 23 has fps: 19
Video 24 has fps: 29
Video 25 has fps: 60
Video 26 has fps: 30
Video 27 has fps: 15
Video 28 has fps: 29
Video 29 has fps: 29
Video 30 has fps: 29
'''
