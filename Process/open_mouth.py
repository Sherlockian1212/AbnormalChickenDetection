from ultralytics import YOLO
import cv2

class open_mouthDetection:
    def __init__(self, video_path=""):
        self.model = YOLO(r'../Weight/best.pt')
        self.video_path = video_path
    def detect_image(self, image, conf=0.7):
        source = image
        ROIs, cls_names, predict_image = [], [], None
        results = self.model(source, conf=conf)
        predict_image = results[0].plot()
        for result in results:
            if result:
                ROIs = result.boxes.xywh.tolist()
                class_ids = result.boxes.cls
                cls_names = [self.model.names[int(cls_id)] for cls_id in class_ids]  # Chuyển đổi ID thành tên

        # print(ROIs)
        return ROIs, cls_names, predict_image

# img = cv2.imread(r"../Input/panting_video_2_frame_at_0.png")
#
# ROIs, cls_names, predict_image = open_mouthDetection().detect_image(image = img)
# print(ROIs)
# print(cls_names)
# cv2.imwrite(r"../Output/result_video_24.png", predict_image)

"""
video2
0: 384x640 2 heads, 3 open-mouths, 85.7ms
Speed: 4.5ms preprocess, 85.7ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
        [[597.9583, 169.9123, 398.5019, 304.5443],
        [979.4059, 535.5466, 535.1905, 650.4451],
        [814.4603, 410.0126, 207.0403, 186.0936],
        [471.3812, 150.1105, 161.2735, 117.9204],
        [ 92.4316, 672.6202, 184.8632, 167.4857]]
['head', 'head', 'open-mouth', 'open-mouth', 'open-mouth']

video5
0: 384x640 1 head, 1 open-mouth, 144.2ms
Speed: 6.6ms preprocess, 144.2ms inference, 15.1ms postprocess per image at shape (1, 3, 384, 640)
[[842.7332763671875, 334.2538757324219, 403.325439453125, 359.86614990234375], [696.191162109375, 239.81924438476562, 113.18096923828125, 75.60089111328125]]
['head', 'open-mouth']

video24
0: 640x448 2 heads, 2 open-mouths, 151.9ms
Speed: 9.9ms preprocess, 151.9ms inference, 14.9ms postprocess per image at shape (1, 3, 640, 448)
[[102.06138610839844, 291.1856384277344, 158.81759643554688, 227.8517303466797], [327.3940124511719, 366.11773681640625, 323.52783203125, 452.2311096191406], [144.61953735351562, 247.03912353515625, 76.63640594482422, 67.62336730957031], [243.97747802734375, 208.90011596679688, 157.67201232910156, 134.72950744628906]]
['head', 'head', 'open-mouth', 'open-mouth']
"""

