def get_head(ROIs, cls_names):
    head_rois = [ROIs[i] for i in range(len(cls_names)) if cls_names[i] == 'head']
    head_rois = convert_rois(head_rois)
    return head_rois

def get_open_mouth(ROIs, cls_names):
    open_mouth_rois = [ROIs[i] for i in range(len(cls_names)) if cls_names[i] == 'open-mouth']
    open_mouth_rois = convert_rois(open_mouth_rois)
    return open_mouth_rois

# Convert bounding box of YOLO to normal (x,y,w,h)
def convert_rois(rois):
    rois_convert = []
    for roi in rois:
        x_center, y_center, width, height = roi
        x = int(x_center - width / 2)
        y = int(y_center - height / 2)
        w = int(width)
        h = int(height)
        rois_convert.append((x, y, w, h))
    return rois_convert

def get_object(open_mouth, heads):
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

# ROIs = [[102.06138610839844, 291.1856384277344, 158.81759643554688, 227.8517303466797], [327.3940124511719, 366.11773681640625, 323.52783203125, 452.2311096191406], [144.61953735351562, 247.03912353515625, 76.63640594482422, 67.62336730957031], [243.97747802734375, 208.90011596679688, 157.67201232910156, 134.72950744628906]]
ROIs = [[842.7332763671875, 334.2538757324219, 403.325439453125, 359.86614990234375], [696.191162109375, 239.81924438476562, 113.18096923828125, 75.60089111328125]]
cls_names = ['head', 'open-mouth']
# cls_names = ['head', 'head', 'open-mouth', 'open-mouth']

mouths = get_open_mouth(ROIs, cls_names)
heads = get_head(ROIs,cls_names)

for m in mouths:
    h = get_object(m, heads)
    print(f"{m} --- {h}")

"""
(710, 316, 207, 186) --- (711, 210, 535, 650)
(390, 91, 161, 117) --- (398, 17, 398, 304)
(0, 588, 184, 167) --- ()

(639, 202, 113, 75) --- (641, 154, 403, 359)

(106, 213, 76, 67) --- (22, 177, 158, 227)
(165, 141, 157, 134) --- (165, 140, 323, 452)
"""