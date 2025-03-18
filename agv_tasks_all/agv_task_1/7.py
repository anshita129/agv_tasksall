import cv2 as cv
import numpy as np

# Compute image gradients using Sobel operator
def compute_gradients(image):
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0, ksize=3)
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1, ksize=3)
    return grad_x, grad_y

# Build image pyramids for multi-resolution analysis
def build_pyramid(image, levels):
    pyramid = [image]
    for _ in range(1, levels):
        image = cv.pyrDown(image)
        pyramid.append(image)
    return pyramid

# Lucas-Kanade Optical Flow Algorithm
def lucas_kanade(prev_img, curr_img, points, pyramid_levels=3, iterations=5):
    pyramid1 = build_pyramid(prev_img, pyramid_levels)
    pyramid2 = build_pyramid(curr_img, pyramid_levels)

    flow_points = np.copy(points).astype(np.float32) / (2 ** (pyramid_levels - 1))

    for k in range(pyramid_levels - 1, -1, -1):  
        img1, img2 = pyramid1[k], pyramid2[k]
        grad_x, grad_y = compute_gradients(img1)

        for _ in range(iterations):
            movement = np.zeros_like(flow_points)

            for idx, point in enumerate(flow_points):
                x, y = point.ravel()

                if 0 <= x < img1.shape[1] and 0 <= y < img1.shape[0]:
                    nx, ny = int(x), int(y)

                    # Intensity difference between frames
                    intensity_diff = img2[ny, nx] - img1[ny, nx]

                    # Motion estimation based on gradients
                    movement[idx] += 0.1 * np.array([
                        grad_x[ny, nx] * intensity_diff,
                        grad_y[ny, nx] * intensity_diff
                    ])

            # Refine movement and update points
            flow_points += movement / (np.maximum(1e-3, np.linalg.norm(movement, axis=1, keepdims=True)))
            flow_points = np.clip(flow_points, 0, np.array(img1.shape[::-1]) - 1)

    return flow_points

# Main video capture and visualization
cap = cv.VideoCapture("opticalflow2.mp4")
ret, first_frame = cap.read()
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

# Detect strong corners for tracking
p0 = cv.goodFeaturesToTrack(prev_gray, maxCorners=300, qualityLevel=0.3, minDistance=5, blockSize=7)
p0 = np.array([pt.ravel() for pt in p0], dtype=np.float32)

# Drawing mask for visualization
mask = np.zeros_like(first_frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calculate optical flow using Lucas-Kanade
    p1 = lucas_kanade(prev_gray, gray, p0)

    # Draw tracking points
    for new, old in zip(p1, p0):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        cv.circle(frame, (a, b), 5, (0, 255, 0), -1)
        cv.line(mask, (c, d), (a, b), (0, 255, 0), 2)

    # Overlay tracks
    output = cv.add(frame, mask)
    cv.imshow("Lucas-Kanade Optical Flow", output)

    prev_gray = gray.copy()
    p0 = np.array(p1).reshape(-1, 1, 2)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
