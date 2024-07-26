import cv2
import numpy as np

def getVectorTranslation(img1, img2, fx, fy, cx, cy):
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
# normal camera
# fx: 3052.51
# fy: 2367.92
# cx: 1344.00
# cy: 760.00

# thermal camera
# fx: 176.78
# fy: 140.57
# cx: 128.00
# cy: 96.00


n_img1 = cv2.imread('../Frame/normal_image_1.png')
n_img2 = cv2.imread('../Frame/normal_image_2.png')
print("Normal image")
R, t = getVectorTranslation(n_img1, n_img2,
                            fx = 3052.51,
                            fy = 2367.92,
                            cx = 1344.00,
                            cy = 760.00)
K = np.array([[3052.51, 0, 1344],
              [0, 2367.92, 760],
              [0, 0, 1]])
# Kiểm tra tính chất toán học của ma trận xoay
valid_rotation = checkRotationMatrix(R)
print(f"Is the rotation matrix valid? {valid_rotation}")

t_img1 = cv2.imread('../Frame/normal_image_1.png')
t_img2 = cv2.imread('../Frame/normal_image_2.png')
print("\nThermal image")
R, t = getVectorTranslation(t_img1, t_img2,
                            fx = 176.78,
                            fy = 140.57,
                            cx = 128.00,
                            cy = 96.00,
                            )
K = np.array([[176.78, 0, 128.00],
              [0, 140.57, 96.00],
              [0, 0, 1]])
# Kiểm tra tính chất toán học của ma trận xoay
valid_rotation = checkRotationMatrix(R)
print(f"Is the rotation matrix valid? {valid_rotation}")

# Normal image
# Rotation matrix:
#  [[ 1.00000000e+00  1.38222767e-14 -1.13742349e-13]
#  [-1.35169653e-14  1.00000000e+00 -2.14550600e-14]
#  [ 1.13797860e-13  2.16909823e-14  1.00000000e+00]]
# Translation vector:
#  [[ 0.76402542]
#  [ 0.19871243]
#  [-0.61382287]]
# Determinant: 0.9999999999999998
# Orthogonal Check: True
# Is the rotation matrix valid? True
#
# Thermal image
# Rotation matrix:
#  [[ 1.00000000e+00 -9.93741062e-11 -1.39574602e-11]
#  [ 9.93751262e-11  1.00000000e+00 -3.14687304e-11]
#  [ 1.39558781e-11  3.14686155e-11  1.00000000e+00]]
# Translation vector:
#  [[ 0.99643165]
#  [ 0.04974914]
#  [-0.06818345]]
# Determinant: 0.9999999999999994
# Orthogonal Check: True
# Is the rotation matrix valid? True