import torch
import torchvision
import cv2
import numpy as np

def blur_image_array_faces(
    image: np.ndarray,
    face_detector,
    lp_detector,
    face_model_score_threshold=0.9,
    lp_model_score_threshold=0.9,
    nms_iou_threshold=0.3,
    scale_factor_detections=1.1,
    device=torch.device("cpu")
) -> np.ndarray:
    def get_image_tensor(bgr_image, device):
        image_tensor = torch.from_numpy(bgr_image).permute(2, 0, 1).float().unsqueeze(0)  # [1, 3, H, W]
        return image_tensor.to(device)

    def get_detections(detector, image_tensor, score_thresh, iou_thresh):
        with torch.no_grad():
            detections = detector(image_tensor[0])  # for torchscript EgoBlur
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

    def visualize_blur(image, boxes):
        image_fg = image.copy()
        mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
        for box in boxes:
            if scale_factor_detections != 1.0:
                box = scale_box(box, image.shape[1], image.shape[0], scale_factor_detections)
            x1, y1, x2, y2 = map(int, box)
            if x2 > x1 and y2 > y1:
                def make_odd(x):
                    return x if x % 2 == 1 else x + 1

                k = max((y2 - y1) // 4, 3)  # Use object size to choose blur strength
                ksize = (make_odd(k), make_odd(k))  # Ensure ksize is odd and > 0

                # ksize = (image.shape[0] // 20, image.shape[1] // 20)  # moderate Gaussian blur size
                image_fg[y1:y2, x1:x2] = cv2.GaussianBlur(image_fg[y1:y2, x1:x2], ksize, 0)
                cv2.ellipse(mask, ((x1 + x2)//2, (y1 + y2)//2), ((x2 - x1)//2, (y2 - y1)//2), 0, 0, 360, 255, -1)
        inverse_mask = cv2.bitwise_not(mask)
        img_bg = cv2.bitwise_and(image, image, mask=inverse_mask)
        img_fg = cv2.bitwise_and(image_fg, image_fg, mask=mask)
        return cv2.add(img_bg, img_fg)

    # Prepare input
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = get_image_tensor(image_rgb.copy(), device)
    image_tensor_copy = image_tensor.clone()

    detections = []
    if face_detector:
        detections += get_detections(face_detector, image_tensor, face_model_score_threshold, nms_iou_threshold)
    if lp_detector:
        detections += get_detections(lp_detector, image_tensor_copy, lp_model_score_threshold, nms_iou_threshold)

    return visualize_blur(image.copy(), detections)
