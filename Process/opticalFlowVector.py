import cv2
import numpy as np

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

def getVectorTranslation(img1, img2, fx = 2887.24, fy = 2095.35, cx = 1344, cy = 760):
    # Phát hiện các điểm đặc trưng (keypoints) trong ảnh đầu tiên
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Tạo đối tượng BFMatcher và tìm các điểm match giữa hai ảnh
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Lọc ra các điểm match tốt nhất
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:50]

    # Lấy tọa độ của các điểm match
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Ma trận camera (giả sử camera đã được hiệu chỉnh)
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    # Tính toán ma trận thiết yếu
    E, mask = cv2.findEssentialMat(pts1, pts2, K)

    # Khôi phục pose (rotation và translation) từ ma trận thiết yếu
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

    print("Rotation matrix:\n", R)
    print("Translation vector:\n", t)
    return R, t

# fx: 2887.24, fy: 2095.35
# cx: 1344
# cy: 760


img1 = cv2.imread('../Frame/normal_image_1.png')
img2 = cv2.imread('../Frame/normal_image_2.png')
R, t = getVectorTranslation(img1, img2)

def checkRotationMatrix(R):
    # Tính định thức của ma trận xoay
    det = np.linalg.det(R)
    print(f"Determinant: {det}")

    # Kiểm tra trực giao của ma trận xoay
    RtR = np.dot(R.T, R)
    I = np.identity(3)
    ortho_check = np.allclose(RtR, I)
    print(f"Orthogonal Check: {ortho_check}")

    return ortho_check and np.isclose(det, 1.0)

K = np.array([[2887.24, 0, 1344],
              [0, 2095.35, 760],
              [0, 0, 1]])

# Kiểm tra tính chất toán học của ma trận xoay
valid_rotation = checkRotationMatrix(R)
print(f"Is the rotation matrix valid? {valid_rotation}")

# [[ 0.77738142]
#  [ 0.21611635]
#  [-0.5907384 ]]