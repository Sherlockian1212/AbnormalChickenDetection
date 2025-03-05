from openMouthDetection import *
from getObject import *
from pantingDetection import *
import pandas as pd

def heatStressDetection(video_path, FPS=15, period=3, num_video=0):
    video_frames, total_frames = readVideoFrames(video_path, target_fps=FPS)
    index = 0

    # Danh sách để lưu thông tin cần thiết
    results = []

    while index < total_frames:
        if index % FPS == 0:
            period_id = index // (FPS * period)
            print(f"\nPeriod: {period_id}")
            rois, cls_name, scores = openMouthDetection(video_frames[index], save=False)

        if rois:
            if index + period * FPS + 1 <= total_frames:
                frames = video_frames[index:index + period * FPS + 1]
            else:
                frames = video_frames[index:]

            if frames:
                head_rois, head_scores = getHead(rois, cls_name, scores)
                mouth_rois, list_R1 = getMouth(rois, cls_name, scores)
                head_rois_of_mouth = []

                for i, mouth in enumerate(mouth_rois):
                    temp = getObject(mouth, head_rois)
                    if temp:
                        head_rois_of_mouth.append(temp)
                    # if temp == ():
                    #     print(f"ROI {i}: Open-mouth: {list_R1[i]}. Couldn't find its head.")
                    # else:
                    #     print(f"ROI {i}: Open-mouth: {list_R1[i]}. Found its head.")

                # print(f"Found {len(head_rois_of_mouth)} heads with open-mouth")
                # print("Panting Detection:")
                dir_path=""
                # dir_path = f'../Output/Period {period_id}'
                # os.makedirs(dir_path, exist_ok=True)
                if head_rois_of_mouth:
                    list_R2 = pantingDetection(frames=frames,
                                               rois=head_rois_of_mouth,
                                               FPS=FPS,
                                               min_prequency=2,
                                               save_image=False,
                                               save_HDF5=False,
                                               save_frequency=False,
                                               path=dir_path)

                    print(list_R1)
                    print(list_R2)

                    list_R = [(r1 + r2) / 2 for r1, r2 in zip(list_R1, list_R2)]
                    print(list_R)
                    list_R_New = [calR(r1,r2) for r1, r2 in zip(list_R1, list_R2)]
                    print(list_R_New)

                # Lưu thông tin vào danh sách results
                for i in range(len(head_rois_of_mouth)):
                    results.append({
                        'Video ID': video_path,  # hoặc tên video tương ứng
                        'Period ID': period_id,
                        'Open-Mouth ID': i + 1,
                        'Head ID': head_rois_of_mouth[i] if head_rois_of_mouth[i] else None,
                        'R1': list_R1[i],
                        'R2': list_R2[i] if i < len(list_R2) else None,
                        'R': list_R[i],
                        'R New': list_R_New[i]
                    })

                index += period * FPS
                rois, cls_names, scores, head_rois_of_mouth, list_R2 = [], [], [], [], []
        else:
            index += 1

    # Chuyển đổi results thành DataFrame và lưu vào file Excel
    df_results = pd.DataFrame(results)
    df_results.to_excel(rf'D:\STUDY\DHSP\KLTN-2024-2025-With my idol\Source_Code\HeatStressDetection\Output\report/heat_stress_detection_results_video_{num_video}.xlsx', index=False)

    return results  # Trả về danh sách kết quả nếu cần

# for i in range(28):
#     video_path = rf"D:\STUDY\DHSP\KLTN-2024-2025-With my idol\Dataset\heat stress\video_processed\video ({i+1}).mp4"
#     heatStressDetection(video_path=video_path, FPS=15, period=3, num_video=i+1)

# for i in range(28,33):
#     print(i)
#     video_path = rf"D:\STUDY\DHSP\KLTN-2024-2025-With my idol\Dataset\heat stress\video_processed\video ({i}).webm"
#     heatStressDetection(video_path=video_path, FPS=15, period=3, num_video=i)

video_path = rf"D:\STUDY\DHSP\KLTN-2024-2025-With my idol\Dataset\heat stress\video_processed\video (5).mp4"
heatStressDetection(video_path=video_path, FPS=15, period=3, num_video=5)

# import cv2
# import pandas as pd
# from openMouthDetection import *
# from getObject import *
# from pantingDetection import *
#
# def heatStressDetection(video_path, FPS=15, period=3, num_video=0):
#     # Mở video
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     index = 0
#
#     # Danh sách để lưu thông tin cần thiết
#     results = []
#     period_frames = FPS * period  # Tổng số frame trong một khoảng thời gian
#
#     while index < total_frames:
#         # Đọc frame tiếp theo
#         ret, frame = cap.read()
#         if not ret:
#             break  # Nếu không còn frame để đọc, thoát
#
#         if index % FPS == 0:
#             period_id = index // period_frames
#             print(f"\nPeriod: {period_id}")
#             rois, cls_name, scores = openMouthDetection(frame, save=False)
#
#         if rois:
#             # Lấy các frame trong khoảng thời gian
#             frames = []
#             for i in range(period_frames):
#                 if index + i < total_frames:
#                     ret, temp_frame = cap.read()
#                     if ret:
#                         frames.append(temp_frame)
#
#             if frames:
#                 head_rois, head_scores = getHead(rois, cls_name, scores)
#                 mouth_rois, list_R1 = getMouth(rois, cls_name, scores)
#                 head_rois_of_mouth = []
#
#                 for i, mouth in enumerate(mouth_rois):
#                     head_rois_of_mouth.append(getObject(mouth, head_rois))
#
#                 dir_path = ""
#                 list_R2, list_peak_frequency, list_filtered_count, list_total_count = pantingDetection(frames=frames,
#                                                                                                        rois=head_rois_of_mouth,
#                                                                                                        FPS=FPS,
#                                                                                                        min_prequency=2,
#                                                                                                        save_image=False,
#                                                                                                        save_HDF5=False,
#                                                                                                        save_frequency=False,
#                                                                                                        path=dir_path)
#
#                 list_R = [(r1 + r2) / 2 for r1, r2 in zip(list_R1, list_R2)]
#                 print(list_R)
#
#                 # Lưu thông tin vào danh sách results
#                 for i in range(len(mouth_rois)):
#                     results.append({
#                         'Video ID': video_path,
#                         'Period ID': period_id,
#                         'Open-Mouth ID': i + 1,
#                         'Head ID': head_rois_of_mouth[i] if head_rois_of_mouth[i] else None,
#                         'Peak_Frequency': list_peak_frequency[i],
#                         'Filtered': list_filtered_count[i],
#                         'Total': list_total_count[i],
#                         'R1': list_R1[i],
#                         'R2': list_R2[i] if i < len(list_R2) else None,
#                         'R': list_R[i]
#                     })
#
#                 index += period_frames
#                 # Reset thông tin đã lưu
#                 rois, cls_names, scores, list_R2, head_rois_of_mouth = [], [], [], [], []
#         else:
#             index += 1
#
#     # Giải phóng video
#     cap.release()
#
#     # Chuyển đổi results thành DataFrame và lưu vào file Excel
#     df_results = pd.DataFrame(results)
#     df_results.to_excel(rf'D:\STUDY\DHSP\KLTN-2024-2025-With my idol\Source_Code\HeatStressDetection\Output\report/heat_stress_detection_results_video_{num_video}.xlsx', index=False)
#
#     return results  # Trả về danh sách kết quả nếu cần
#
# # Ví dụ gọi hàm
# video_path = rf"D:\STUDY\DHSP\KLTN-2024-2025-With my idol\Dataset\heat stress\video\video (5).mp4"
# heatStressDetection(video_path=video_path, FPS=15, period=3, num_video=5)
