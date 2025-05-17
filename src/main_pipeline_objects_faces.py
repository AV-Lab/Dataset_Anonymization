# main_pipeline_with_cropping.py (Object ID Tracking + Blurring for Plates and Faces)

import os
import cv2
import torch
from tqdm import tqdm

from ego_blur_utils_faces import blur_image_array_faces
# Define relative paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FRAMES_DIR = os.path.join(ROOT_DIR, 'frames')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'blurred_video_054907')

video_index_map = {
    'video_054604': 0,
    'video_054907': 1,
    'video_115533': 2,
    'video_115833': 3,
    'video_122233': 4,
    'video_125233': 5,
    'video_131333': 6,
    'video_141432-141733': 7,
    'video_142911': 8,
    'video_143739': 9,
    'video_151901': 10,
    'video_155425': 11,
    'video_160325': 12,
    'video_160902-161205': 13,
    'video_174340': 14,
    'video_204347': 15,
    'video_204647': 16,
    'video_210906': 17,
    'video_214547': 18,
    'video_220047': 19
}
selected_video = 'video_141432-141733'

ANNOTATION_FILENAME = str(selected_video+'.txt')   # v2
# ANNOTATION_FILENAME = 'video_115533.txt'   # v2


ANNOTATION_PATH = os.path.join(ROOT_DIR, 'annotations', 'annotations', 'tracking_annotations', 'gmot', ANNOTATION_FILENAME)

# Always use CPU for this version
device = torch.device("cpu")

# Load both face and license plate models
LP_MODEL_PATH = os.path.join(MODELS_DIR, 'ego_blur_lp.jit')
FACE_MODEL_PATH = os.path.join(MODELS_DIR, 'ego_blur_face.jit')

lp_detector = torch.jit.load(LP_MODEL_PATH, map_location=device).eval()
face_detector = torch.jit.load(FACE_MODEL_PATH, map_location=device).eval()

# Parse annotation file into a dictionary
frame_objects = {}  # {frame_id: [(obj_id, bbox)]}
object_blur_log = {}  # {obj_id: {"frames": [...], "blurred": [...]}}

with open(ANNOTATION_PATH, 'r') as file:
    for line in file:
        parts = line.strip().split()
        frame_id = int(parts[0])
        obj_id = int(parts[1])
        x = int(float(parts[2]))
        y = int(float(parts[3]))
        w = int(float(parts[4]))
        h = int(float(parts[5]))
        bbox = (x, y, x + w, y + h)
        frame_objects.setdefault(frame_id, []).append((obj_id, bbox))

video_folders = sorted([f for f in os.listdir(FRAMES_DIR) if os.path.isdir(os.path.join(FRAMES_DIR, f))])
if not video_folders:
    print("❌ No video folders found in /frames")
else:
    video_name = video_folders[video_index_map[selected_video]]  # Select desired video index
    input_folder = os.path.join(FRAMES_DIR, video_name)
    output_folder = os.path.join(OUTPUT_DIR, video_name)
    os.makedirs(output_folder, exist_ok=True)

    print(f"📂 Processing folder: {video_name}")

    frame_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png'))])[0:]

    for frame in tqdm(frame_files, desc=f"🚀 Tracking-Aware EgoBlur: {video_name}", unit="frame"):
        frame_index = int(os.path.splitext(frame)[0])
        input_path = os.path.join(input_folder, frame)
        output_path = os.path.join(output_folder, frame)

        image = cv2.imread(input_path)
        if image is None:
            continue

        min_crop_area_ratio = 0.0002
        min_crop_area = int(image.shape[0] * image.shape[1] * min_crop_area_ratio)

        objects = frame_objects.get(frame_index, [])
        detections_made = False

        for obj_id, (x1, y1, x2, y2) in objects:
            if y2 > y1 and x2 > x1:
                crop = image[y1:y2, x1:x2]
                if crop.size == 0 or (crop.shape[0] * crop.shape[1] < min_crop_area):
                    continue

                object_blur_log.setdefault(obj_id, {"frames": [], "blurred": []})
                object_blur_log[obj_id]["frames"].append(frame_index)

                blurred = False
                for attempt in range(5):
                    threshold_score = 0.9 - (attempt * 0.2)
                    blurred_crop = blur_image_array_faces(
                        crop,
                        face_detector,
                        lp_detector,
                        face_model_score_threshold=threshold_score,
                        lp_model_score_threshold=threshold_score,
                        nms_iou_threshold=0.3,
                        scale_factor_detections=1.1
                    )
                    if not (blurred_crop == crop).all():
                        image[y1:y2, x1:x2] = blurred_crop
                        blurred = True
                        detections_made = True
                        break

                object_blur_log[obj_id]["blurred"].append(1 if blurred else 0)

        cv2.imwrite(output_path, image)

    print(f"✅ Done processing: {video_name}")

    import json
    log_path = os.path.join(OUTPUT_DIR, f"{video_name}_blur_log.json")
    with open(log_path, "w") as f:
        json.dump(object_blur_log, f, indent=2)
    print(f"📝 Blur log saved to {log_path}")
