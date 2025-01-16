import cv2
import os

# Parameters
FPS = 1

# Define paths
video_folder = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "raw_videos"
)
save_folder = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "extracted_frames"
)
os.makedirs(save_folder, exist_ok=True)

# Load video
for video in os.listdir(video_folder):
    video_path = os.path.join(video_folder, video)
    video_name = os.path.splitext(video)[0]

    # Extract frames
    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save frames
        if frame_id % int(cap.get(cv2.CAP_PROP_FPS) / FPS) == 0:
            frame_filename = os.path.join(save_folder, f"frame_{frame_id:04d}.jpg")
            cv2.imwrite(frame_filename, frame)

        frame_id += 1

    cap.release()
    print(f"{video} saved")

print("All videos processed")
