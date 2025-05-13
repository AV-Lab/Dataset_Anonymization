import cv2
import os

# Define paths
# frames_dir = r'D:\Dataset anonymizing\output_annotation_crop_fast5\video_220047'.strip()
og_frames_dir = r'D:\Dataset anonymizing\blurred_video_220047\video_220047'.strip()
# og_frames_dir = r'D:\Dataset anonymizing\frames\video_220047'.strip()
output_video = 'output_video_220047.mp4'

# Get list of frames
frame_files = sorted([
    f for f in os.listdir(og_frames_dir) 
    if f.lower().endswith(('.jpg', '.png'))
])

# Read first frame to get dimensions
first_frame = cv2.imread(os.path.join(og_frames_dir, frame_files[0]))
height, width, _ = first_frame.shape

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'avc1' or 'XVID' if 'mp4v' fails
fps = 25  # Adjust FPS as needed
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Write frames to video
for frame_file in frame_files:
    frame_path = os.path.join(og_frames_dir, frame_file)
    frame = cv2.imread(frame_path)
    if frame is not None:
        video_writer.write(frame)

video_writer.release()
print(f"🎥 Video saved to: {output_video}")
