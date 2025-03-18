import cv2
import numpy as np
import os

def inRange(cordinates, limits):
    x, y = cordinates
    X_Limit, Y_Limit = limits
    return 0 <= x < X_Limit and 0 <= y < Y_Limit

def optical_flow(old_frame, new_frame, window_size, min_quality=0.01):
    max_corners = 10000
    min_distance = 0.1
    feature_list = cv2.goodFeaturesToTrack(old_frame, max_corners, min_quality, min_distance)

    w = int(window_size / 2)

    old_frame = old_frame / 255
    new_frame = new_frame / 255

    kernel_x = np.array([[-1, 1], [-1, 1]])
    kernel_y = np.array([[-1, -1], [1, 1]])
    kernel_t = np.array([[1, 1], [1, 1]])

    fx = cv2.filter2D(old_frame, -1, kernel_x)
    fy = cv2.filter2D(old_frame, -1, kernel_y)
    ft = cv2.filter2D(new_frame, -1, kernel_t) - cv2.filter2D(old_frame, -1, kernel_t)

    u = np.zeros(old_frame.shape)
    v = np.zeros(old_frame.shape)

    for feature in feature_list:
        j, i = feature.ravel()
        i, j = int(i), int(j)

        I_x = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
        I_y = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
        I_t = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()

        b = np.reshape(I_t, (I_t.shape[0], 1))
        A = np.vstack((I_x, I_y)).T

        U = np.matmul(np.linalg.pinv(A), b)

        u[i, j] = U[0][0]
        v[i, j] = U[1][0]

    return u, v

def drawOnFrame(frame, U, V):
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            u, v = U[i][j], V[i][j]

            if u and v:
                frame = cv2.arrowedLine(frame, (j, i), (int(j + u), int(i + v)), (0, 255, 0), thickness=1)
    return frame

# Ensure Results folder exists
os.makedirs("./Results", exist_ok=True)

# Video Processing
cap = cv2.VideoCapture("opticalflow.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter("./Results/output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame_width, frame_height))

ret, old_frame = cap.read()
old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    ret, new_frame = cap.read()
    if not ret:
        break

    new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    U, V = optical_flow(old_frame_gray, new_frame_gray, 3, 0.05)

    output_frame = drawOnFrame(new_frame, U, V)
    out.write(output_frame)

    old_frame_gray = new_frame_gray

cap.release()
out.release()
cv2.destroyAllWindows()

# Auto-play the output video after processing
output_path = "./Results/output_video.mp4"
cap_out = cv2.VideoCapture(output_path)

while cap_out.isOpened():
    ret, frame = cap_out.read()
    if not ret:
        break

    cv2.imshow('Optical Flow Output', frame)

    # Press 'Q' to exit early
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap_out.release()
cv2.destroyAllWindows()
