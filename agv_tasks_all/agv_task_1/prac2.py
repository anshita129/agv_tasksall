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

def build_pyramid(image, levels):
    """
    Builds a Gaussian pyramid for the input image.
    
    Parameters:
        - image: Input image
        - levels: Number of pyramid levels

    Returns:
        - pyramid: List of images at different scales
    """
    pyramid = [image]
    for i in range(1, levels):
        pyramid.append(cv.pyrDown(pyramid[i-1]))
    return pyramid

def lucas_kanade_pyramid(prev_gray, next_gray, feature_points, levels=3, window_size=15):
    """
    Computes optical flow using the Lucas-Kanade method with a pyramid approach.
    
    Parameters:
        - prev_gray: Previous grayscale frame
        - next_gray: Current grayscale frame
        - feature_points: Array of feature points to track (Nx2)
        - levels: Number of pyramid levels
        - window_size: Size of the neighborhood for computing flow

    Returns:
        - new_points: Updated feature point positions (Nx2)
        - status: Status of tracking for each point (1 if successful, 0 if not)
    """
    # Build pyramids for previous and next frames
    prev_pyramid = build_pyramid(prev_gray, levels)
    next_pyramid = build_pyramid(next_gray, levels)

    # Initialize points at the highest level (coarsest scale)
    points = feature_points / (2 ** (levels - 1))

    # Iterate from the coarsest to the finest level
    for level in range(levels - 1, -1, -1):
        # Scale points to the current level
        points *= 2 if level != levels - 1 else 1

        # Compute optical flow at the current level
        points, status = compute_optical_flow(prev_pyramid[level], next_pyramid[level], points, window_size)

    return points, status

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=2, blockSize=7)

# Read video
cap = cv.VideoCapture("opticalflow2.mp4")
ret, first_frame = cap.read()
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

# Detect initial feature points
p0 = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
p0 = np.array([pt.ravel() for pt in p0], dtype=np.float32)

# Create a mask image for drawing tracks
mask = np.zeros_like(first_frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Compute optical flow using pyramid approach
    p1, status = lucas_kanade_pyramid(prev_gray, gray, p0, levels=3, window_size=15)

    # Select good points
    good_new = p1[status == 1]
    good_old = p0[status == 1]

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)

    output = cv.add(frame, mask)
    cv.imshow("Lucas-Kanade Optical Flow", output)

    # Update previous frame and points
    prev_gray = gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()