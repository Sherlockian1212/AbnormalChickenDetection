import cv2
import numpy as np
import glob

# Số hàng và cột của mẫu lưới
checkerboard_size = (9, 6)
# Kích thước thực của mỗi ô trên mẫu lưới (ví dụ: 25mm)
square_size = 25

# Tiêu chuẩn kết thúc của thuật toán tìm kiếm góc
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Các điểm 3D thực tế của mẫu lưới (world coordinates)
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Các danh sách để lưu trữ các điểm 3D và 2D
objpoints = []  # Điểm 3D trong thế giới thực
imgpoints = []  # Điểm 2D trong hình ảnh

# Đọc tất cả các ảnh của mẫu lưới
images = glob.glob('../calibration_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Tìm các góc của mẫu lưới
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Vẽ và hiển thị các góc
        cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# Hiệu chỉnh camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Ma trận nội tại (intrinsic matrix)
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

# Các tham số nội tại
fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]

print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
