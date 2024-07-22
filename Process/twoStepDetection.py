import cv2
import os
from collections import Counter
from Process.yoloDetection import yoloDetection
from Process.stationaryDetection import detectStationary

# Convert bounding box of YOLO to normal (x,y,w,h)
def convert_rois(rois):
    rois_convert = []
    for roi in rois[0]:
        x_center, y_center, width, height = roi
        x = int(x_center - width / 2)
        y = int(y_center - height / 2)
        w = int(width)
        h = int(height)
        rois_convert.append((x, y, w, h))
    return rois_convert

# Draw rois on frames and write it into video
def draw_rectangles_and_write(frames, filtered_rois, out):
    for frame in frames:
        for (x, y, w, h) in filtered_rois:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        out.write(frame)

# process step
def process_frames(frames, rois, count_frame, fps, out, frames_to_draw):
    rois_convert = convert_rois(rois)
    dead = detectStationary(frames=frames, rois=rois_convert)
    flattened_data = [roi for sublist in dead for roi in sublist]
    roi_counts = Counter(flattened_data)
    roi_percentages = {roi: (count / count_frame) * 100 for roi, count in roi_counts.items()}
    filtered_rois = [roi for roi, count in roi_counts.items() if count >= 0.7 * count_frame]
    for roi, percentage in roi_percentages.items():
        print(f"ROI: {roi}, Count: {roi_counts[roi]}/{count_frame}, Percentage: {percentage:.2f}%")
    if filtered_rois:
        print("NOTIFICATION: DEAD!")
    draw_rectangles_and_write(frames_to_draw, filtered_rois, out)

def detection(video_path, second=30):
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
        f"Results/{video_name}_result.mp4",
        fourcc, fps, (frame_width, frame_height)
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            count += 1
            if flag:
                if count % fps == 0:
                    rois, predict_image = yoloDetection().detect_image(frame)
                out.write(frame)
            if rois:
                if flag:
                    print(f"Have risk at {(count // fps) // 60}:{(count // fps) % 60}")
                flag = False
                if count % (fps//2) == 0:
                    frames.append(frame)
                    count_frame += 1
                frames_to_draw.append(frame)
                if count_frame >= second*2:
                    process_frames(frames, rois, count_frame, fps, out, frames_to_draw)
                    count_frame = 0
                    rois = []
                    flag = True
                    frames = []
                    frames_to_draw = []
        elif count_frame > 0:
            process_frames(frames, rois, count_frame, fps, out, frames_to_draw)
            count_frame = 0
            rois = []
            flag = True
            frames = []
            frames_to_draw = []
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
#
# # Call the function to play the video
# video_path = r"D:\Project\Dead Laying Hens Detection\Data\Video\2\Thermal\192.168.0.103_02_20240522140244290.mp4"
# detection(video_path, 30)
