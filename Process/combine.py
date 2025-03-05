import cv2
import os
from collections import Counter
from Process.open_mouth import open_mouthDetection
from Process.sprase_optical_flow import detectMoving
from get_object import *

# Draw rois on frames and write it into video
def draw_rectangles_and_write(frames, filtered_rois, out):
    for frame in frames:
        for (x, y, w, h) in filtered_rois:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        out.write(frame)

# process step
def process_frames(frames, rois, count_frame, fps, out, frames_to_draw):
    moving,_ = detectMoving(frames=frames, rois=rois)
    flattened_data = [roi for sublist in moving for roi in sublist]
    roi_counts = Counter(flattened_data)
    roi_percentages = {roi: (count / count_frame) * 100 for roi, count in roi_counts.items()}
    filtered_rois = [roi for roi, count in roi_counts.items() if count >= 0.7 * count_frame]
    for roi, percentage in roi_percentages.items():
        print(f"ROI: {roi}, Count: {roi_counts[roi]}/{count_frame}, Percentage: {percentage:.2f}%")
    if filtered_rois:
        print("NOTIFICATION: HEAT STRESS!")
    draw_rectangles_and_write(frames_to_draw, filtered_rois, out)

def detection(video_path, second=5):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    fps = 25  #int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    flag = True
    count = 0
    count_frame = 0
    rois = []
    frames_to_draw = []

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        f"../Output/{video_name}_result.mp4",
        fourcc, fps, (frame_width, frame_height)
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            count += 1
            if flag:
                if count % fps == 0:
                    rois, cls_name, predict_image = open_mouthDetection().detect_image(frame)
                out.write(frame)
            if rois:
                if flag:
                    print(f"Open-mouth at {(count // fps) // 60}:{(count // fps) % 60}")
                flag = False
                if count % (fps) == 0:
                    frames.append(frame)
                    count_frame += 1
                frames_to_draw.append(frame)
                head_rois = get_head(rois, cls_name)
                if count_frame >= second*2:
                    process_frames(frames, head_rois, count_frame, fps, out, frames_to_draw)
                    count_frame = 0
                    head_rois = []
                    flag = True
                    frames = []
                    frames_to_draw = []
        elif count_frame > 0:
            process_frames(frames, head_rois, count_frame, fps, out, frames_to_draw)
            count_frame = 0
            head_rois = []
            flag = True
            frames = []
            frames_to_draw = []
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Call the function to play the video
video_path = r"../Input/video (24).mp4"
detection(video_path, 5)
