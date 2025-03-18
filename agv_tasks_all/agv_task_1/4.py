import cv2
import numpy as np

def compute_optical_flow(prev_frame, next_frame, window_size=15):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Compute gradients
    Ix = cv2.Sobel(prev_gray, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(prev_gray, cv2.CV_64F, 0, 1, ksize=5)
    It = next_gray.astype(np.float64) - prev_gray.astype(np.float64)

    # Initialize flow vectors
    flow = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float64)

    # Half window size
    half_window = window_size // 2

    for y in range(half_window, prev_gray.shape[0] - half_window):
        for x in range(half_window, prev_gray.shape[1] - half_window):
            # Extract the window
            Ix_window = Ix[y-half_window:y+half_window+1, x-half_window:x+half_window+1].flatten()
            Iy_window = Iy[y-half_window:y+half_window+1, x-half_window:x+half_window+1].flatten()
            It_window = It[y-half_window:y+half_window+1, x-half_window:x+half_window+1].flatten()

            # Form the matrices
            A = np.vstack((Ix_window, Iy_window)).T
            b = -It_window

            # Solve the system
            if np.linalg.matrix_rank(A) == 2:
                flow[y, x] = np.linalg.lstsq(A, b, rcond=None)[0]

    return flow

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=2, blockSize=7)

# Read the video
cap = cv2.VideoCapture("opticalflow.mp4")
ret, first_frame = cap.read()
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
prev = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
mask = np.zeros_like(first_frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = compute_optical_flow(first_frame, frame)

    # Draw the flow vectors
    for y in range(0, flow.shape[0], 10):
        for x in range(0, flow.shape[1], 10):
            if np.linalg.norm(flow[y, x]) > 0.5:  # Threshold to draw significant flows
                cv2.line(mask, (x, y), (int(x + flow[y, x, 0]), int(y + flow[y, x, 1])), (0, 255, 0), 2)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    output = cv2.add(frame, mask)
    cv2.imshow('Optical Flow', output)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    prev_gray = gray.copy()

cap.release()
cv2.destroyAllWindows()