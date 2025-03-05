def getHead(ROIs, cls_names, scores):
    head_rois = [ROIs[i] for i in range(len(cls_names)) if cls_names[i] == 'head']
    head_scores = [scores[i] for i in range(len(cls_names)) if cls_names[i] == 'head']
    head_rois = convertRois(head_rois)
    return head_rois, head_scores

def getMouth(ROIs, cls_names, scores):
    open_mouth_rois = [ROIs[i] for i in range(len(cls_names)) if cls_names[i] == 'open-mouth']
    open_mouth_scores = [scores[i] for i in range(len(cls_names)) if cls_names[i] == 'open-mouth']
    open_mouth_rois = convertRois(open_mouth_rois)
    return open_mouth_rois, open_mouth_scores

# Convert bounding box of YOLO to normal (x,y,w,h)
def convertRois(rois):
    rois_convert = []
    for roi in rois:
        x_center, y_center, width, height = roi
        x = int(x_center - width / 2)
        y = int(y_center - height / 2)
        w = int(width)
        h = int(height)
        rois_convert.append((x, y, w, h))
    return rois_convert

def getObject(open_mouth, heads):
    max = -1
    result = ()
    for head in heads:
        if overlap(open_mouth, head) > max and overlap(open_mouth, head) > 0:
            max = overlap(open_mouth, head)
            result = head
    return result

def overlap(roi1, roi2):
    """
    Tính diện tích chồng lấp giữa hai hộp giới hạn (ROIs).

    :param roi1: Tuple hoặc list chứa (x, y, w, h) của hộp giới hạn thứ nhất.
    :param roi2: Tuple hoặc list chứa (x, y, w, h) của hộp giới hạn thứ hai.
    :return: Diện tích chồng lấp giữa roi1 và roi2.
    """
    # Tọa độ góc trên bên trái và góc dưới bên phải của roi1
    x1_min = roi1[0]
    y1_min = roi1[1]
    x1_max = roi1[0] + roi1[2]
    y1_max = roi1[1] + roi1[3]

    # Tọa độ góc trên bên trái và góc dưới bên phải của roi2
    x2_min = roi2[0]
    y2_min = roi2[1]
    x2_max = roi2[0] + roi2[2]
    y2_max = roi2[1] + roi2[3]

    # Tính tọa độ của vùng chồng lấp
    x_overlap_min = max(x1_min, x2_min)
    y_overlap_min = max(y1_min, y2_min)
    x_overlap_max = min(x1_max, x2_max)
    y_overlap_max = min(y1_max, y2_max)

    # Tính diện tích chồng lấp
    overlap_width = max(0, x_overlap_max - x_overlap_min)
    overlap_height = max(0, y_overlap_max - y_overlap_min)
    overlap_area = overlap_width * overlap_height

    return overlap_area
