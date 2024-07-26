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
    print(point1, point2)
    # Tính toán vector dịch chuyển
    vector = np.array(point2) - np.array(point1)
    delta_x, delta_y = vector
    length = np.linalg.norm(vector)
    angle_rad = np.arctan2(delta_y, delta_x)
    angle_deg = np.degrees(angle_rad)
    print(f"Vector: ({delta_x}, {delta_y}); Length: {length}; Angle (degrees): {angle_deg}")

def checkStationaryRois(frame, oldFrame, rois, thresh=2):
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Convert to gray
    newFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    oldFrameGray = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)
    stationaryRois = []
    copy_frame = frame.copy()
    total_length = 0
    total_angle = 0
    num_vectors = 0
    for (x, y, w, h) in rois:
        # Extract the ROI
        roiOld = oldFrameGray[y:y + h, x:x + w]
        roiNew = newFrame[y:y + h, x:x + w]
        # Detect corners in the ROI
        p0 = cv2.goodFeaturesToTrack(roiOld, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        if p0 is not None:
            # Calculate optical flow within the ROI
            p1, st, err = cv2.calcOpticalFlowPyrLK(roiOld, roiNew, p0, None, **lk_params)
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Draw the optical flow vectors
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                delta_x, delta_y = a - c, b - d
                length = np.linalg.norm([delta_x, delta_y])
                angle_rad = np.arctan2(delta_y, delta_x)
                angle_deg = np.degrees(angle_rad)
                cv2.arrowedLine(copy_frame, (x + int(c), y + int(d)), (x + int(a), y + int(b)), (0, 255, 0), 2,
                                tipLength=0.5)

                # Print the vector
                print(f"Vector: ({delta_x}, {delta_y}); Length: {length}; Angle (degrees): {angle_deg}")

                total_length += length
                total_angle += angle_deg
                num_vectors += 1
            # Check if the points have moved significantly
            if np.all(np.linalg.norm(good_new - good_old, axis=1) < thresh):
                stationaryRois.append((x, y, w, h))
    if num_vectors > 0:
        avg_length = total_length / num_vectors
        avg_angle = total_angle / num_vectors
        print(f"Average Length: {avg_length}")
        print(f"Average Angle (degrees): {avg_angle}")
    else:
        print("No vectors to calculate averages.")
    return stationaryRois, copy_frame


# Tính toán vector dịch chuyển
img1 = cv2.imread('../Frame/thermal_image_1.png')
img2 = cv2.imread('../Frame/thermal_image_2.png')
t_height, t_width = img1.shape[:2]

nor_img1 = cv2.imread('../Frame/normal_image_1.png')
nor_img2 = cv2.imread('../Frame/normal_image_2.png')
n_height, n_width = nor_img1.shape[:2]

rois1, predict_image1 = yoloDetection().detect_image(img1, conf=0.7)
rois2, predict_image2 = yoloDetection().detect_image(img2, conf=0.7)

roi1 = convert_rois(rois1)
roi2 = convert_rois(rois2)
point1 = getMainPoint(roi1[0])
point2 = getMainPoint(roi2[0])

# Tạo hình nền trắng
combined_image = np.ones((t_height, t_width, 3), dtype=np.uint8) * 255

# Vẽ các ROI từ img1 lên nền trắng
for (x, y, w, h) in roi1:
    cv2.rectangle(combined_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Vẽ các ROI từ img2 lên nền trắng
for (x, y, w, h) in roi2:
    cv2.rectangle(combined_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

point1 = tuple(map(int, point1))
point2 = tuple(map(int, point2))
cv2.circle(combined_image, point1, 5, (0, 255, 0), -1)  # Điểm từ roi1, màu xanh lá
cv2.circle(combined_image, point2, 5, (255, 0, 255), -1)  # Điểm từ roi2, màu tím

# Hiển thị hình ảnh kết hợp
cv2.imshow("Combined ROIs", combined_image)

# Tính tỷ lệ thay đổi kích thước
scale_x = n_width / t_width
scale_y = n_height / t_height
x, y, w, h = roi1[0]  # Thay đổi tên biến cho phù hợp với dữ liệu của bạn

# Tính toán ROI tương ứng trong nor_img1
new_x = int(x * scale_x)
new_y = int(y * scale_y)
new_w = int(w * scale_x)
new_h = int(h * scale_y)

# ROI mới
n_roi1 = [(new_x, new_y, new_w, new_h)]

point1 = getMainPoint(roi1[0])
point2 = getMainPoint(roi2[0])


print("Location vector:")
getLocationVector(roi1, roi2)

print("\nOptical Flow Vector in normal image:")
stationaryRois, n_vector = checkStationaryRois(nor_img2, nor_img1, n_roi1, thresh=2)
cv2.imshow('Optical flow vector in normal image', n_vector)

print("\nOptical Flow Vector in thermal image:")
stationaryRois, t_vector = checkStationaryRois(img2, img1, roi1, thresh=2)
cv2.imshow('Optical flow vector in thermal image', t_vector)

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


# Location vector:
# Vector: (15.5, 3.5); Length: 15.890248582070704; Angle (degrees): 12.724355685422369
#
# Optical Flow Vector in normal image:
# Vector: (27.4683837890625, 4.4430389404296875); Length: 27.825395584106445; Angle (degrees): 9.188070297241211
# Vector: (26.275375366210938, 4.329160690307617); Length: 26.62962532043457; Angle (degrees): 9.356060981750488
# Vector: (28.4075927734375, 4.241443634033203); Length: 28.72248649597168; Angle (degrees): 8.491910934448242
# Vector: (28.420913696289062, 4.463722229003906); Length: 28.769309997558594; Angle (degrees): 8.925826072692871
# Vector: (28.309707641601562, 4.498958587646484); Length: 28.66496467590332; Angle (degrees): 9.029891014099121
# Vector: (27.715805053710938, 4.264118194580078); Length: 28.041906356811523; Angle (degrees): 8.746461868286133
# Vector: (21.71990394592285, 4.7375030517578125); Length: 22.230567932128906; Angle (degrees): 12.304527282714844
# Vector: (29.333202362060547, -2.7956953048706055); Length: 29.466127395629883; Angle (degrees): -5.4443135261535645
# Vector: (-14.665557861328125, 4.504037857055664); Length: 15.341608047485352; Angle (degrees): 162.92739868164062
# Vector: (28.70441436767578, 4.782329559326172); Length: 29.10007095336914; Angle (degrees): 9.458943367004395
# Vector: (26.703826904296875, 3.8314285278320312); Length: 26.977291107177734; Angle (degrees): 8.164995193481445
# Average Length: 26.52448671514338
# Average Angle (degrees): 21.922706560655072
#
# Optical Flow Vector in thermal image:
# Vector: (13.203344345092773, 2.9123220443725586); Length: 13.520722389221191; Angle (degrees): 12.438814163208008
# Vector: (15.880542755126953, 2.729848861694336); Length: 16.11346435546875; Angle (degrees): 9.753758430480957
# Vector: (13.853248596191406, 1.8623912334442139); Length: 13.977875709533691; Angle (degrees): 7.656773090362549
# Vector: (15.034744262695312, 3.106403350830078); Length: 15.35230541229248; Angle (degrees): 11.673896789550781
# Vector: (14.010917663574219, 1.5060303211212158); Length: 14.09162712097168; Angle (degrees): 6.135153293609619
# Average Length: 14.611198997497558
# Average Angle (degrees): 9.531679153442383