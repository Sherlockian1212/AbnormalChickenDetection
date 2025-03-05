import cv2
import numpy as np
import matplotlib.pyplot as plt

def convertToRGB(flow):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Create mask
    height, width = flow.shape[:2]
    hsv_mask = np.zeros((height, width, 3), dtype=np.uint8)
    # Make image saturation to a maximum value
    hsv_mask[..., 1] = 255
    # Set image hue value according to the angle of optical flow
    hsv_mask[..., 0] = ang * 180 / np.pi / 2
    # Set value as per the normalized magnitude of optical flow
    hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
    return rgb_representation

def saveFrequencyImage(xf, yf, output_image_path):
    plt.plot(xf, yf)
    plt.title(f'Fourier transform of the magnitude signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.savefig(output_image_path)
    plt.close()

def readVideoFrames(video_path, target_fps=15):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return [],0
    # Get the original FPS of the video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / target_fps)  # Calculate interval for target FPS
    frames = []
    count = 0
    total_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frames.append(frame)
            total_frames += 1
        count += 1
    cap.release()
    return frames, total_frames
