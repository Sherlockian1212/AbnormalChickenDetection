from Process.yoloDetection import yoloDetection
from Process.twoStepDetection import convert_rois
import cv2
import numpy as np


def drawROIs(img, rois, color=(0, 255, 0), thickness=2):
    # Sao chép hình ảnh để không làm thay đổi hình ảnh gốc
    img_copy = img.copy()

    for roi in rois:
        x, y, w, h = roi
        # Vẽ hình chữ nhật lên hình ảnh
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, thickness)

    return img_copy

def getMainPoint(roi):
    x, y, width, height = roi
    x_center = x + width / 2
    y_center = y + height / 2
    return [x_center, y_center]

def getLocationVector(ROI_1, ROI_2):
    point1 = getMainPoint(ROI_1[0])
    point2 = getMainPoint(ROI_2[0])

    # Tính toán vector dịch chuyển
    vector = np.array(point2) - np.array(point1)

    return vector

def getTranslationVector(img1, img2, roi1, roi2):
    # Cắt ROI từ hình ảnh
    img1_roi = img1[roi1[1]:roi1[1] + roi1[3], roi1[0]:roi1[0] + roi1[2]]
    img2_roi = img2[roi2[1]:roi2[1] + roi2[3], roi2[0]:roi2[0] + roi2[2]]

    # Phát hiện các điểm đặc trưng trong ROI
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1_roi, None)
    kp2, des2 = orb.detectAndCompute(img2_roi, None)

    # Kiểm tra số lượng điểm đặc trưng
    if len(kp1) == 0 or len(kp2) == 0 or len(des1) == 0 or len(des2) == 0:
        print("No keypoints or descriptors found in one of the ROIs.")
        return None

    # Tạo đối tượng BFMatcher và tìm các điểm match giữa hai ROI
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Kiểm tra số lượng match
    if len(matches) < 5:
        print("Not enough matches found.")
        return None

    # Lọc ra các điểm match tốt nhất
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:5]

    # Lấy tọa độ của các điểm match
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Chuyển đổi tọa độ các điểm từ ROI sang tọa độ toàn bộ hình ảnh
    pts1 += np.float32([roi1[0], roi1[1]])
    pts2 += np.float32([roi2[0], roi2[1]])

    # Tính toán vector dịch chuyển trung bình
    translation_vector = np.mean(pts2 - pts1, axis=0)

    return translation_vector



# Tính toán vector dịch chuyển
img1 = cv2.imread('../Frame/thermal_image_1.png')
img2 = cv2.imread('../Frame/thermal_image_2.png')

rois1, predict_image1 = yoloDetection().detect_image(img1, conf=0.7)
rois2, predict_image2 = yoloDetection().detect_image(img2, conf=0.7)

roi1 = convert_rois(rois1)
roi2 = convert_rois(rois2)

print(roi1)
print(roi2)

translation_vector = getTranslationVector(img1, img2, roi1[0], roi2[0])
print(f"Translation vector: {translation_vector}")

print(getLocationVector(roi1, roi2))

# Vẽ các ROI lên hình ảnh
img1_with_rois = drawROIs(img1, roi1)
img2_with_rois = drawROIs(img2, roi2)

# Hiển thị hình ảnh với các ROI
cv2.imshow('Image 1 with ROIs', img1_with_rois)
cv2.imshow('Image 2 with ROIs', img2_with_rois)

cv2.waitKey(0)
cv2.destroyAllWindows()

# [15.5, 3.5]

# [[ 0.77738142]
#  [ 0.21611635]
#  [-0.5907384 ]]