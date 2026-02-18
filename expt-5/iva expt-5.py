import cv2
import numpy as np

# Function to perform motion analysis using moving edges
def motion_analysis(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video")
        return

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Cannot read video")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform Canny edge detection on both frames
        edges_prev = cv2.Canny(prev_gray, 50, 150)
        edges_curr = cv2.Canny(gray, 50, 150)

        # Compute frame difference to detect moving edges
        frame_diff = cv2.absdiff(edges_prev, edges_curr)

        # Display the moving edges
        cv2.imshow("Moving Edges", frame_diff)

        # Press q to exit
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        # Update previous frame
        prev_gray = gray.copy()

    cap.release()
    cv2.destroyAllWindows()


# Replace with your actual video file name/path
video_path = "Human Analytics video.mp4"
motion_analysis(video_path)
