import cv2
import time
import os

# Parameters
MOTION_DURATION_THRESHOLD = 5
CONTOUR_AREA_THRESHOLD = 200
THRESHOLD_SENSITIVITY = 25
BACKGROUND_SUBTRACTOR_HISTORY = 500
BACKGROUND_SUBTRACTOR_THRESHOLD = 50
FPS = 30
POST_MOTION_FRAMES = 600


# Initialize video capture with 1 for back camera 0 for front camera
cap = cv2.VideoCapture(1)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_file = None
recording = False

# Initialize background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=BACKGROUND_SUBTRACTOR_HISTORY, varThreshold=BACKGROUND_SUBTRACTOR_THRESHOLD
)

# Define save path
save_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw_videos")
os.makedirs(save_folder, exist_ok=True)

motion_detected_frames = 0
post_motion_counter = 0

while True:
    ret, frame = cap.read()

    # Break if no frame
    if not ret:
        break

    # Convert to grayscale and apply background subtractor
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = background_subtractor.apply(gray_frame)

    # Apply threshold to foreground mask
    _, fgmask_thresh = cv2.threshold(
        fgmask, THRESHOLD_SENSITIVITY, 255, cv2.THRESH_BINARY
    )

    # Detect contours to determine motion
    contours, _ = cv2.findContours(
        fgmask_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    motion_detected = any(
        cv2.contourArea(contour) > CONTOUR_AREA_THRESHOLD for contour in contours
    )

    # Start recording if motion is detected
    if motion_detected:
        motion_detected_frames += 1
        # Reset post-motion counter
        post_motion_counter = POST_MOTION_FRAMES
    else:
        motion_detected_frames = 0

    # Start recording when threshold reached
    if motion_detected_frames > MOTION_DURATION_THRESHOLD and not recording:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(save_folder, f"cat_{timestamp}.avi")
        output_file = cv2.VideoWriter(
            output_path, fourcc, FPS, (frame.shape[1], frame.shape[0])
        )
        recording = True
        print(f"Recording started at {timestamp}")

    # Continue if motion, otherwise countdown or release if counter is at 0
    if recording:
        if motion_detected:
            output_file.write(frame)
        elif post_motion_counter > 0:
            output_file.write(frame)
            post_motion_counter -= 1
        else:
            recording = False
            output_file.release()
            print("Recording stopped")

    # Display frame
    cv2.imshow("Motion Detection", fgmask_thresh)

    # Break loop with 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release
cap.release()
if output_file is not None:
    output_file.release()
cv2.destroyAllWindows()
