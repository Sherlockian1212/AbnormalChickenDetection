from ultralytics import YOLO
import os
from moviepy.editor import *
import cv2

class yoloDetection:
    def __init__(self, video_path=""):
        self.model = YOLO(r'../Weight/best_v5.pt')
        self.video_path = video_path
    def loadVideo(self):
        try:
            print(f"Attempting to load video: {self.video_path}")
            clip = cv2.VideoCapture(self.video_path)
            print("Video loaded successfully")
            return clip
        except Exception as e:
            print(f"Error loading video: {e}")
            return None

    def getFrames(self, cap):
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % int(fps) == 0:
                frame_filename = f"Frame/{frame_count / int(fps)}.png"
                cv2.imwrite(frame_filename, frame)
            frame_count += 1
        cap.release()
        cv2.destroyAllWindows()

        print("Frames have been saved.")
    def detect_image(self, image, conf=0.7):
        source = image
        results = self.model(source, conf=conf)
        predict_image = results[0].plot()
        ROIs = []
        for result in results:
            if result:
                ROIs.append(result.boxes.xywh)
        # print(ROIs)
        return ROIs, predict_image

    def detect_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("Results/Duoi_chan.mp4", fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            if success:
                # Run YOLOv8 inference on the frame
                results = self.model(frame)
                # Visualize the results on the frame
                annotated_frame = results[0].plot()
                out.write(annotated_frame)
                # Display the annotated frame
                cv2.imshow("YOLOv8 Inference", annotated_frame)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        cap.release()
        out.release()
        cv2.destroyAllWindows()
# # # Open the video file
# video_path = r"C:\Users\admin\Downloads\Quay duoi chan_ga chet 3.mp4"
# predetect = PreDetection().detect_video(video_path)