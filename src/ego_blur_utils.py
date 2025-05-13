import torch
import torchvision
import cv2
import numpy as np

def blur_image_array(
    image: np.ndarray,
    face_detector,
    lp_detector,
    face_model_score_threshold=0.9,
    lp_model_score_threshold=0.9,
    nms_iou_threshold=0.3,
    scale_factor_detections=1.1,
    device=torch.device("cpu")
) -> np.ndarray:
    def get_image_tensor(bgr_image):
        image_tensor = torch.from_numpy(bgr_image).permute(2, 0, 1).float()  # [3, H, W]
        return image_tensor.to(device)

    def get_detections(detector, image_tensor, score_thresh, iou_thresh):
        with torch.no_grad():
            detections = detector(image_tensor[0])  # [boxes, classes, scores, ??]

        boxes, _, scores, _ = detections  # Assume boxes shape [N,4], scores shape [N]

        # Filter out low-confidence detections first (optional optimization)
        keep_conf = scores > score_thresh
        boxes = boxes[keep_conf]
        scores = scores[keep_conf]

        # Apply Non-Maximum Suppression
        keep_nms = torchvision.ops.nms(boxes, scores, iou_thresh)
        boxes = boxes[keep_nms]
        scores = scores[keep_nms]

        return boxes.cpu().numpy().tolist()


    def scale_box(box, max_w, max_h, scale):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        xc, yc = x1 + w / 2, y1 + h / 2
        w, h = scale * w, scale * h
        return [
            max(xc - w / 2, 0), max(yc - h / 2, 0),
            min(xc + w / 2, max_w), min(yc + h / 2, max_h)
        ]

    def visualize_blur(image, boxes):
        image_fg = image.copy()
        for box in boxes:
            if scale_factor_detections != 1.0:
                box = scale_box(box, image.shape[1], image.shape[0], scale_factor_detections)
            x1, y1, x2, y2 = map(int, box)
            if x2 > x1 and y2 > y1:
                roi = image_fg[y1:y2, x1:x2]
                kx = max(3, (x2 - x1) // 3 | 1)  # ensure odd number
                ky = max(3, (y2 - y1) // 3 | 1)
                blurred_roi = cv2.GaussianBlur(roi, (kx, ky), 0)
                image_fg[y1:y2, x1:x2] = blurred_roi
        return image_fg

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = get_image_tensor(image_rgb.copy())

    detections = []
    if face_detector:
        detections += get_detections(face_detector, image_tensor, face_model_score_threshold, nms_iou_threshold)
    if lp_detector:
        detections += get_detections(lp_detector, image_tensor, lp_model_score_threshold, nms_iou_threshold)

    return visualize_blur(image.copy(), detections)
