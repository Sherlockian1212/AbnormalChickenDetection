import cv2
import numpy as np
# Detect the ROIs are stationary or not
# frames: list of frames extracted from cap. Maybe list of frames in 10 seconds
#           depends on fps, we have fps*(num of seconds) in frames
# rois: list of roi (x, y, w, h)
# Output: stationary ROIs

def detectStationary(frames, rois,
                     lkParams = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))):
    staticROIs = []
    # Initialize background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2()
    prev = frames[0]
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for frame in frames[1:]:
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Check for stationary ROIs
        stationaryRois, frameWithRois, prev = checkStationaryRois(frame, prev, rois)

        # Apply background subtraction and maintain stationary ROIs
        fgMask = backgroundSubtraction(frame, backSub, stationaryRois)
        # print("rois", stationaryRois)
        # Display the resulting frame
        # for (x, y, w, h) in stationaryRois:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        staticROIs.append(stationaryRois)
    return staticROIs
# Detect the objects in ROIs are stationary or not.
# Using combination of background and Lucas-Kanade
# rois: list of roi (x, y, w, h)
# cap:  input video
def stationaryDetect(cap, rois):
    # Initialize background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2()
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Take first frame
    ret, old_frame = cap.read()
    if not ret:     #some thing wrong in video
        cap.release()
        return None
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        '''
        # Apply background subtraction
        fgMask = backSub.apply(frame)
        # Threshold the mask to remove shadows and noise
        _, fgMask = cv2.threshold(fgMask, 250, 255, cv2.THRESH_BINARY)
        '''

        # Check for stationary ROIs
        stationary_rois, frame_with_rois, old_gray = checkStationaryRois(frame, old_gray, rois)

        # Apply background subtraction and maintain stationary ROIs
        fgMask = backgroundSubtraction(frame, backSub, stationary_rois)

        # Display the resulting frame
        for (x, y, w, h) in stationary_rois:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        cv2.imshow('Stationary Objects', frame)
        cv2.imshow('Foreground Mask', fgMask)
        if cv2.waitKey(30) & 0xFF == 27:  # Press ESC to exit
            break
    cap.release()
    cv2.destroyAllWindows()
# Function to apply background subtraction and maintain stationary ROIs
def backgroundSubtraction(frame, backSub, rois):
    fgMask = backSub.apply(frame, learningRate=0.01)

    # Apply mask to frame
    mask = np.zeros_like(fgMask)
    for (x, y, w, h) in rois:
        mask[y:y+h, x:x+w] = 255
    fgMask = cv2.bitwise_and(fgMask, fgMask, mask=mask)

    return fgMask

# Check if rois are stationary
# frame: the current frame
# oldFrame: the previous frame
# thresh: (u, v) < = thresh --> roi is stationary
# vector optical flow between (u, v) the roi in two adjacent frames
def checkStationaryRois(frame, oldFrame, rois, thresh = 2):
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Convert to gray
    newFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    stationaryRois = []
    for (x, y, w, h) in rois:
        # Extract the ROI
        roiOld = oldFrame[y:y + h, x:x + w]
        roiNew = newFrame[y:y + h, x:x + w]
        # Detect corners in the ROI
        p0 = cv2.goodFeaturesToTrack(roiOld, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7,
                                     blockSize=7)
        if p0 is not None:
            # Calculate optical flow within the ROI
            p1, st, err = cv2.calcOpticalFlowPyrLK(roiOld, roiNew, p0, None, **lk_params)
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            # Check if the points have moved significantly
            if np.all(np.linalg.norm(good_new - good_old, axis=1) < thresh):
                stationaryRois.append((x, y, w, h))
    return stationaryRois, frame, newFrame

# # Read video file
# video = '../datasets/192.168.0.103_01_20240522140244288.mp4'
# cap = cv2.VideoCapture(video)
#
# # Read the first frame
# ret, prev = cap.read()
#
# # Select a region of interest (ROI) for object tracking
# x, y, w, h = cv2.selectROI(prev, False)
#
#
# rois = [(x, y, w, h )]
# stationaryDetect(cap, rois)
