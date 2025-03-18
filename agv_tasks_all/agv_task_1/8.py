import cv2 as cv
import numpy as np

# Build image pyramids for multi-resolution analysis
def build_pyramid(image, numdepths):
    pyramid = [image]
    for _ in range(1, numdepths):
        image = cv.pyrDown(image)
        pyramid.append(image)
    return pyramid

# Compute image gradients (Sobel operator for x and y directions)
def compute_gradients(image):
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0, ksize=3) / 255.0
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1, ksize=3) / 255.0
    return grad_x, grad_y

# Lucas-Kanade Optical Flow algorithm
def lucas_kanade(prev_img, curr_img, points, numdepths=4, num_iter=5):
    pyramid1 = build_pyramid(prev_img, numdepths)
    pyramid2 = build_pyramid(curr_img, numdepths)

    flow_points = points / (2 ** (numdepths - 1))  # Correct scaling logic

    for k in range(numdepths - 1, -1, -1):
        img1, img2 = pyramid1[k], pyramid2[k]
        grad_x, grad_y = compute_gradients(img1)

        for _ in range(num_iter):
            movement = np.zeros_like(flow_points)
            for idx, point in enumerate(flow_points):
                x, y = point.ravel()
                nx, ny = int(x), int(y)

                if 0 <= nx < img1.shape[1] and 0 <= ny < img1.shape[0]:
                    intensity_diff = img2[ny, nx] - img1[ny, nx]
                    movement[idx] = 0.5 * np.array([
                        grad_x[ny, nx] * intensity_diff,
                        grad_y[ny, nx] * intensity_diff
                    ])
            flow_points += movement

    return flow_points * (2 ** (numdepths - 1))  # Scale points back to original resolution

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=300, qualityLevel=0.3, minDistance=5, blockSize=7)

# Video capture setup
cap = cv.VideoCapture("opticalflow2.mp4")
color = (0, 255, 0)

# Read the first frame
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read the video file.")
    exit()

prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

# Detect feature points in the first frame
mask = np.zeros_like(prev_gray)
mask[100:400, 50:300] = 255  # Refined mask to focus on the person
p0 = cv.goodFeaturesToTrack(prev_gray, mask=mask, **feature_params)

if p0 is None:
    print("Error: No features detected in the first frame.")
    exit()

p0 = np.array([pt.ravel() for pt in p0], dtype=np.float32)

# Create a mask for drawing
mask = np.zeros_like(first_frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    p1 = lucas_kanade(prev_gray, gray, p0)

    # Draw tracking points with arrowed lines for better visualization
    MAX_MOVEMENT = 50
    valid_points = []
    for new, old in zip(p1, p0):
        if np.linalg.norm(new - old) < MAX_MOVEMENT:
            a, b = new.ravel()
            c, d = old.ravel()

            if np.isfinite(a) and np.isfinite(b):
                a, b, c, d = int(a), int(b), int(c), int(d)
                if 0 <= a < frame.shape[1] and 0 <= b < frame.shape[0]:
                    cv.arrowedLine(mask, (c, d), (a, b), color, 2, tipLength=0.3)
                    valid_points.append([a, b])

    output = cv.add(frame, mask)
    cv.imshow("Lucas-Kanade Optical Flow", output)

    prev_gray = gray.copy()
    if valid_points:
        p0 = np.array(valid_points, dtype=np.float32).reshape(-1, 1, 2)
    else:
        print("Warning: No valid points found in this frame.")
        break

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv.destroyAllWindows()
