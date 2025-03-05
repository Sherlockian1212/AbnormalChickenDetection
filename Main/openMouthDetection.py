from ultralytics import YOLO
import cv2

def openMouthDetection(source, modelPath = r'../Weight/YOLO11.pt', conf=0.7, save = False):
    model = YOLO(modelPath)
    ROIs, cls_names, scores, predict_image = [], [], [], None
    results = model(source, conf=conf)
    predict_image = results[0].plot()
    if save:
        output_path = '../Output/openMouthDetection.jpg'  # Specify your desired output path
        cv2.imwrite(output_path, predict_image)
    for result in results:
        if result:
            ROIs = result.boxes.xywh.tolist()
            class_ids = result.boxes.cls
            scores = result.boxes.conf.tolist()
            cls_names = [model.names[int(cls_id)] for cls_id in class_ids]
    return ROIs, cls_names, scores