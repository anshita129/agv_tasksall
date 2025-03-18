import cv2 as cv
import numpy as np

def compute_optical_flow(prev_gray, next_gray, feature_points, window_size=15):
    """
    Computes sparse optical flow using the Lucas-Kanade method manually.
    
    Parameters:
        - prev_gray: Previous grayscale frame
        - next_gray: Current grayscale frame
        - feature_points: Array of feature points to track (Nx2)
        - window_size: Size of the neighborhood for computing flow

    Returns:
        - new_points: Updated feature point positions (Nx2)
        - status: Status of tracking for each point (1 if successful, 0 if not)
    """
    # Compute gradients
    Ix = cv.Sobel(prev_gray, cv.CV_64F, 1, 0, ksize=3)  # Gradient in X direction
    Iy = cv.Sobel(prev_gray, cv.CV_64F, 0, 1, ksize=3)  # Gradient in Y direction
    It = next_gray.astype(np.float32) - prev_gray.astype(np.float32)  # Temporal gradient

    half_w = window_size // 2
    new_points = []
    status = []

    for point in feature_points:
        x, y = int(point[0]), int(point[1])  # Get integer coordinates

        # Ensure point is within valid range
        if x < half_w or y < half_w or x >= prev_gray.shape[1] - half_w or y >= prev_gray.shape[0] - half_w:
            new_points.append((x, y))
            status.append(0)
            continue

        # Extract local gradients within the window
        Ix_window = Ix[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()
        Iy_window = Iy[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()
        It_window = It[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()

        # Construct the A matrix and b vector
        A = np.vstack((Ix_window, Iy_window)).T  # (N x 2) matrix
        b = -It_window  # (N x 1) vector

        # Solve for velocity vector using least squares
        if A.shape[0] >= 2:  # Ensure enough pixels for solving equations
            V = np.linalg.pinv(A) @ b  # Solve (Vx, Vy)
            dx, dy = V[:2]  # Extract motion vector
            new_points.append((x + dx, y + dy))
            status.append(1)
        else:
            new_points.append((x, y))
            status.append(0)

    return np.array(new_points), np.array(status)

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=2, blockSize=7)
# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("opticalflow.mp4")
# Variable for color to draw optical flow track
color = (0, 255, 0)
# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
prev = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
# Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
mask = np.zeros_like(first_frame)

while cap.isOpened():
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    if not ret:
        break
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Calculates sparse optical flow using the custom Lucas-Kanade function
    next, status = compute_optical_flow(prev_gray, gray, prev.squeeze())
    
    # Select good feature points for previous position
    good_old = prev[status == 1].astype(int)
    # Select good feature points for next position
    good_new = next[status == 1].astype(int)
    
    # Draws the optical flow tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        # Returns a contiguous flattened array as (x, y) coordinates for new point
        a, b = new.ravel()
        # Returns a contiguous flattened array as (x, y) coordinates for old point
        c, d = old.ravel()
        # Draws line between new and old position with green color and 2 thickness
        mask = cv.line(mask, (a, b), (c, d), color, 2)
        # Draws filled circle (thickness of -1) at new position with green color and radius of 3
        frame = cv.circle(frame, (a, b), 3, color, -1)
    
    # Overlays the optical flow tracks on the original frame
    output = cv.add(frame, mask)
    # Updates previous frame
    prev_gray = gray.copy()
    # Updates previous good feature points
    prev = good_new.reshape(-1, 1, 2)
    # Opens a new window and displays the output frame
    cv.imshow("Sparse Optical Flow", output)
    # Frames are read by intervals of 10 milliseconds. The program breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()