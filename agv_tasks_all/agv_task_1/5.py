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
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0, ksize=3)
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1, ksize=3)
    return grad_x, grad_y

# Lucas-Kanade Optical Flow algorithm
def lucas_kanade(prev_img, curr_img, points, numdepths=4, num_iter=5):
    pyramid1 = build_pyramid(prev_img, numdepths)
    pyramid2 = build_pyramid(curr_img, numdepths)

    flow_points = np.copy(points).astype(np.float32)

    for k in range(numdepths - 1, -1, -1):
        flow_points *= 2
        img1, img2 = pyramid1[k], pyramid2[k]

        grad_x, grad_y = compute_gradients(img1)

        for _ in range(num_iter):
            movement = np.zeros_like(flow_points)

            for idx, point in enumerate(flow_points):
                x, y = point.ravel()
                nx, ny = int(x), int(y)
                if 0 <= nx < img1.shape[1] and 0 <= ny < img1.shape[0]:
                    intensity_diff = img2[ny, nx] - img1[ny, nx]

                    movement[idx] += [
                        grad_x[ny, nx] * intensity_diff,
                        grad_y[ny, nx] * intensity_diff
                    ]

            flow_points += movement / np.maximum(1e-6, movement.sum(axis=0))

    return flow_points

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=2, blockSize=7)

# Video capture setup
cap = cv.VideoCapture("opticalflow2.mp4")
color = (0, 255, 0)

# Read the first frame
ret, first_frame = cap.read()
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

# Detect feature points in the first frame
p0 = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
p0 = np.array([pt.ravel() for pt in p0], dtype=np.float32)

# Create a mask for drawing
mask = np.zeros_like(first_frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End video if no frame is read

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calculate optical flow using the improved Lucas-Kanade function
    p1 = lucas_kanade(prev_gray, gray, p0)

    # Draw tracking points
    for i, (new, old) in enumerate(zip(p1, p0)):
        a, b = new.ravel()
        c, d = old.ravel()
        cv.circle(frame, (int(a), int(b)), 5, color, -1)
        cv.line(mask, (int(c), int(d)), (int(a), int(b)), color, 2)

    # Overlay tracks on the frame
    output = cv.add(frame, mask)
    cv.imshow("Lucas-Kanade Optical Flow", output)

    # Update previous frame and points
    prev_gray = gray.copy()
    p0 = p1.reshape(-1, 1, 2)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv.destroyAllWindows()