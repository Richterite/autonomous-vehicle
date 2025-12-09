import cv2
import numpy as np


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_lines(image, lines):
    display_line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(display_line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return display_line_image


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0) if left_fit else None
    right_fit_average = np.average(right_fit, axis=0) if right_fit else None

    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)

    if left_line is not None and right_line is not None:
        return np.array([left_line, right_line])
    elif left_line is not None:
        return np.array([left_line])
    elif right_line is not None:
        return np.array([right_line])
    else:
        return None


def make_coordinates(image, line_params):
    if line_params is not None:
        slope, interecept = line_params
        y1 = image.shape[0]
        y2 = int(y1 * (3/5))
        x1 = int((y1 - interecept) / slope)
        x2 = int((y2 - interecept) / slope)
        return np.array([x1, y1, x2, y2])
    else:
        return None


WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_NAME = 'Result'


if __name__ == "__main__":
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
    # img = cv2.imread('./asset/image/test_image.jpg')
    # lane_image = np.copy(img)

    # canny_image = canny(lane_image)

    # cropped_image = region_of_interest(canny_image)
    # Hlines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180,
    #                          100, np.array([]), minLineLength=40, maxLineGap=5)
    # averaged_lines = average_slope_intercept(lane_image, Hlines)

    # line_image = display_lines(lane_image, averaged_lines)
    # combined_imaged = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    # cv2.imshow("Result", combined_imaged)
    # cv2.waitKey(0)

    cap = cv2.VideoCapture("./asset/video/test2.mp4")
    while (cap.isOpened()):
        _, frame = cap.read()
        if frame is None:
            break

        canny_image = canny(frame)

        cropped_image = region_of_interest(canny_image)
        Hlines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180,
                                 100, np.array([]), minLineLength=40, maxLineGap=5)

        if Hlines is not None and len(Hlines) > 0:
            averaged_lines = average_slope_intercept(frame, Hlines)
            line_image = display_lines(frame, averaged_lines)
        else:
            line_image = frame
        combined_imaged = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow(WINDOW_NAME, combined_imaged)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
