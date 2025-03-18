import cv2
import numpy as np

def build_pyramid(image, numdepths):
    pyramid = [image]
    for _ in range(1, numdepths):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def compute_gradients(image):
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return grad_x, grad_y

def lucas_kanade(image1, image2, numdepths=7, num_iter=5):
    pyramid1 = build_pyramid(image1, numdepths)
    pyramid2 = build_pyramid(image2, numdepths)

    total_move = np.array([0.0, 0.0])

    for k in range(numdepths - 1, -1, -1):
        move = total_move * 2
        img1 = pyramid1[k]
        img2 = pyramid2[k]

        grad_x, grad_y = compute_gradients(img1)
        
        for _ in range(num_iter):
            sum_vec = np.array([0.0, 0.0])
            der_sum = 0.0

            for i in range(img1.shape[0]):
                for j in range(img1.shape[1]):
                    nx, ny = int(j + move[0]), int(i + move[1])
                    if 0 <= nx < img1.shape[1] and 0 <= ny < img1.shape[0]:
                        intensity_diff = img2[i, j] - img1[ny, nx]
                        sum_vec += np.array([
                            grad_x[ny, nx] * intensity_diff,
                            grad_y[ny, nx] * intensity_diff
                        ])
                        der_sum += grad_x[ny, nx]**2 + grad_y[ny, nx]**2

            move += sum_vec / der_sum if der_sum != 0 else 0

        total_move += move

    return total_move

# Example usage
if __name__ == "__main__":
    cap = cv2.VideoCapture('opticalflow2.mp4')  # Webcam capture

    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, next_frame = cap.read()
        if not ret:
            break

        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        move = lucas_kanade(prev_gray, next_gray)

        cv2.arrowedLine(next_frame, (320, 240), (320 + int(move[0]), 240 + int(move[1])), (0, 0, 255), 2)
        cv2.imshow("Lucas-Kanade Optical Flow", next_frame)

        prev_gray = next_gray.copy()

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
