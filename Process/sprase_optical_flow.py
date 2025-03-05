import cv2
import numpy as np
import time
# Detect the ROIs are moving or not
# frames: list of frames extracted from cap. Maybe list of frames in 10 seconds
#           depends on fps, we have fps*(num of seconds) in frames
# rois: list of roi (x, y, w, h)
# Output: moving ROIs

def detectMoving(frames, rois,
                     lkParams = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))):
    moveROIs = []
    # Initialize background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2()
    prev = frames[0]
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    flow_images = []
    for frame in frames[1:]:
        start_time = time.time()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Check for moving ROIs
        movingRois, frameWithRois, prev, flow_image = checkMovingRois(frame, prev, rois)

        # Apply background subtraction and maintain moving ROIs
        fgMask = backgroundSubtraction(frame, backSub, movingRois)
        # print("rois", movingRois)
        # Display the resulting frame
        # for (x, y, w, h) in movingRois:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        end_time = time.time()
        execution_time = (end_time - start_time)*1000
        print(f"Execution time for checkMovingRois: {execution_time:.2f} ms")

        moveROIs.append(movingRois)
        flow_images.append(flow_image)

    return moveROIs,flow_images

# Function to apply background subtraction and maintain moving ROIs
def backgroundSubtraction(frame, backSub, rois):
    fgMask = backSub.apply(frame, learningRate=0.01)

    # Apply mask to frame
    mask = np.zeros_like(fgMask)
    for (x, y, w, h) in rois:
        mask[y:y+h, x:x+w] = 255
    fgMask = cv2.bitwise_and(fgMask, fgMask, mask=mask)

    return fgMask

# Check if rois are moving
# frame: the current frame
# oldFrame: the previous frame
# thresh: (u, v) < = thresh --> roi is moving
# vector optical flow between (u, v) the roi in two adjacent frames
def checkMovingRois(frame, oldFrame, rois, thresh = 2):
    lk_params = dict(winSize=(30, 30), maxLevel=5,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Convert to gray
    newFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    movingRois = []
    flow_image = frame.copy()

    for (x, y, w, h) in rois:
        # Extract the ROI
        roiOld = oldFrame[y:y + h, x:x + w]
        roiNew = newFrame[y:y + h, x:x + w]
        # Detect corners in the ROI
        p0 = cv2.goodFeaturesToTrack(roiOld, mask=None, maxCorners=200, qualityLevel=0.1, minDistance=3,
                                     blockSize=9)
        if p0 is not None:
            # Calculate optical flow within the ROI
            p1, st, err = cv2.calcOpticalFlowPyrLK(roiOld, roiNew, p0, None, **lk_params)
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                cv2.line(flow_image, (a + x, b + y), (c + x, d + y), color=(0, 255, 0), thickness=1,
                         lineType=cv2.LINE_AA)
                cv2.circle(flow_image, (a + x, b + y), 2, color=(0, 0, 255))

            # Check if the points have moved significantly
            if np.all(np.linalg.norm(good_new - good_old, axis=1) > thresh):
                movingRois.append((x, y, w, h))
    return movingRois, frame, newFrame, flow_image

def read_video_frames(video_path):
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



# # Đọc các khung hình từ video
# video_path = '../Input/video (24).mp4'  # Thay đổi đường dẫn đến video của bạn
# frames = read_video_frames(video_path)
#
# result, flow_images = detectMoving(frames,[(22, 177, 158, 227), (165, 140, 323, 452)])
# print(result)
#
# i=0
# for flow_image in flow_images:
#     cv2.imwrite(rf"../Output/flow_image_video_24_frame_at_{i}.png", flow_image)
#     i += 1