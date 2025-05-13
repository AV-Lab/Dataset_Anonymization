import torch
import torchvision
import cv2
import numpy as np

def eliptical_blur_image_array(
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
        image_tensor = torch.from_numpy(bgr_image).permute(2, 0, 1).float().unsqueeze(0)  # [1, 3, H, W]
        return image_tensor.to("cpu")

    def get_detections(detector, image_tensor, score_thresh, iou_thresh):
        with torch.no_grad():
            detections = detector(image_tensor[0])
        boxes, _, scores, _ = detections
        keep = torchvision.ops.nms(boxes, scores, iou_thresh)
        boxes, scores = boxes[keep], scores[keep]
        boxes = boxes[scores > score_thresh]
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

    def apply_elliptical_blur_region(image, box):
        x1, y1, x2, y2 = map(int, box)
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            return image

        # Create elliptical mask
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        center = ((x2 - x1) // 2, (y2 - y1) // 2)
        axes = (max((x2 - x1) // 2, 1), max((y2 - y1) // 2, 1))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        # Heavily blur the ROI
        kx = max(15, (x2 - x1) | 1)
        ky = max(15, (y2 - y1) | 1)
        blurred = cv2.GaussianBlur(roi, (kx, ky), 0)

        # Blend using mask
        mask_3ch = cv2.merge([mask] * 3)
        inv_mask = cv2.bitwise_not(mask_3ch)
        blended = cv2.add(cv2.bitwise_and(roi, inv_mask), cv2.bitwise_and(blurred, mask_3ch))

        # Place blended ROI back
        image[y1:y2, x1:x2] = blended
        return image

    # Convert input and prepare tensors
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = get_image_tensor(image_rgb.copy())
    image_tensor_copy = image_tensor.clone()

    detections = []
    if face_detector:
        detections += get_detections(face_detector, image_tensor, face_model_score_threshold, nms_iou_threshold)
    if lp_detector:
        detections += get_detections(lp_detector, image_tensor_copy, lp_model_score_threshold, nms_iou_threshold)

    # Apply elliptical blur for each detection
    for box in detections:
        if scale_factor_detections != 1.0:
            box = scale_box(box, image.shape[1], image.shape[0], scale_factor_detections)
        image = apply_elliptical_blur_region(image, box)

    return image
