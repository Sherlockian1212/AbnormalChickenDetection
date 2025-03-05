def calAccuracy(mouth_score, head_score, filtered_frame, total_frame, f_max, f_threshold, wf=0.5, wt=0.5):
    frame_ratio = filtered_frame / total_frame
    frequency_ratio = min(1, f_max / f_threshold)
    R2 = (frame_ratio * wf) + (frequency_ratio * wt)
    average_detection_score = (mouth_score + head_score) / 2
    accuracy = (average_detection_score + R2) / 2
    return accuracy

def calAccuracyStep2(filtered_frame, total_frame, f_max, f_threshold, wf=0.5, wt=0.5):
    frame_ratio = filtered_frame / total_frame
    frequency_ratio = min(1, f_max / f_threshold)
    R2 = (frame_ratio * wf) + (frequency_ratio * wt)
    return R2

def calR(R1,R2):
    u1 = 1-R1
    u2 = 1-R2
    w1 = u2/(u1+u2)
    w2 = u1/(u1+u2)
    R = R1*w1 + R2*w2
    return R