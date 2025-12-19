import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt


class Detector:
    """
   A utility class used for detecting lane lines in an image or video frame.

    This class encapsulates the computer vision pipeline including:
    - Canny edge detection
    - Region of Interest (ROI) masking
    - Hough Transform for line detection
    - Slope and intercept averaging for smooth lane drawing
    """

    def mask_image(self, image: np.ndarray, region_to_mask: np.array):
        """
        Applies a polygonal mask to the image, keeping only the region of interest.

        Args:
            image (np.ndarray): The input image (typically an edge-detected/grayscale image).
            region_to_mask (np.ndarray): An array of vertices defining the polygon to keep.

        Returns:
            np.ndarray: The masked image where pixels outside the polygon are set to black (0).
        """
        # REMEMBER: Use grayscaled image, not the original because of the difference in the channel (single channel is used in this case)
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, region_to_mask, 255)
        masked = cv2.bitwise_and(image, mask)
        return masked

    def show_image(self, image: np.ndarray) -> None:
        """
        Displays the image using Matplotlib with axis coordinates.

        This is useful for debugging and determining the coordinates for the
        Region of Interest (ROI) vertices.

        Args:
            image (np.ndarray): The image to display.
        """
        plt.imshow(image)
        plt.show()

    def copy_image(self, image: np.ndarray) -> np.ndarray:
        """
        Creates a deep copy of the image.

        This prevents mutation of the original image array when operations 
        are performed on the copy.

        Args:
            image (np.ndarray): The original image.

        Returns:
            np.ndarray: A copy of the image.
        """
        return np.copy(image)

    def to_canny_image(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the Canny Edge Detection algorithm pipeline.

        The pipeline consists of:
        1. Converting RGB to Grayscale.
        2. Applying Gaussian Blur (5x5 kernel) to reduce noise.
        3. Applying Canny Edge Detection (thresholds 50, 150).

        Args:
            image (np.ndarray): The original input image (RGB).

        Returns:
            np.ndarray: A single-channel image representing detected edges.
        """
        # Converting image to grayscale can improve edge detection by algortihm due to single color channel that can maximize the each pixel gradient difference
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 5 x 5 kernel is good choice, eventhough you can still choose other kernel size
        # The standard deviation (param 3) is set to 0
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold1: lower bound threshold
        # threshold2: upper bound threshold
        # if the value is greater than upper bound threshold, than it's accepted as edge pixel.
        # if the value is below than lower bound threshold, than it's rejected as edge pixel.
        # if the value in between both threshold, it's accepted only if it's connected with high gradient edge
        # Note: the best ratio between lower and upper are 1:2 or 1:3
        canny = cv2.Canny(blur, 50, 150)

        return canny

    def average_slope_intercept(self, image: np.ndarray, lines: np.ndarray):
        """
        Calculates the average slope and intercept for the left and right lanes.

        It separates the lines detected by Hough Transform into left and right groups
        based on their slope, averages them, and converts them back to coordinates.

        Args:
            image (np.ndarray): The reference image (used for height calculations).
            lines (np.ndarray): The output from cv2.HoughLinesP.

        Returns:
            np.ndarray: An array containing the coordinates [x1, y1, x2, y2] for 
                        the averaged left and right lines. Returns None if no lines found.
        """

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
        right_fit_average = np.average(
            right_fit, axis=0) if right_fit else None

        left_line = self.make_coordinates(image, left_fit_average)
        right_line = self.make_coordinates(image, right_fit_average)

        if left_line is not None and right_line is not None:
            return np.array([left_line, right_line])
        elif left_line is not None:
            return np.array([left_line])
        elif right_line is not None:
            return np.array([right_line])
        else:
            return None

    def make_coordinates(self, image: np.ndarray, line_params):
        """
        Converts line parameters (slope, intercept) into pixel coordinates.

        Args:
            image (np.ndarray): The image (used to determine the Y-axis limits).
            line_params (list): A tuple or list containing (slope, intercept).

        Returns:
            np.ndarray: An array [x1, y1, x2, y2] representing the line segment.
                        Returns None if line_params is missing.
        """
        if line_params is not None:
            slope, interecept = line_params
            y1 = image.shape[0]
            y2 = int(y1 * (3/5))
            x1 = int((y1 - interecept) / slope)
            x2 = int((y2 - interecept) / slope)
            return np.array([x1, y1, x2, y2])
        else:
            return None

    def display_lines(self, image: np.ndarray, lines):
        """
        Draws the detected lines onto a blank (black) image.

        Args:
            image (np.ndarray): The reference image to determine the canvas size.
            lines (np.ndarray): Array of line coordinates [[x1, y1, x2, y2], ...].

        Returns:
            np.ndarray: An image with lines drawn on a black background.
        """
        display_line_image = np.zeros_like(image)
        if lines is not None:
            for x1, y1, x2, y2 in lines:
                cv2.line(display_line_image, (x1, y1),
                         (x2, y2), (255, 0, 0), 10)
        return display_line_image

    def hough_line(self, image):
        """
        Performs the Probabilistic Hough Transform to detect lines and overlays 
        the averaged lines onto the input image.

        Args:
            image (np.ndarray): The masked edge-detected image.

        Returns:
            np.ndarray: The input image blended with the detected lines.
        """
        # rho=2, theta=pi/180, threshold=100
        h_lines = cv2.HoughLinesP(
            image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

        if h_lines is not None and len(h_lines) > 0:
            averaged_lines = self.average_slope_intercept(image, h_lines)
            line_image = self.display_lines(image, averaged_lines)
        else:
            line_image = image
        combined_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
        return combined_image

    def detect_lines(self, image: np.ndarray, region_of_interest: np.array) -> np.ndarray:
        """
        The main public method to execute the lane detection process.

        It runs the sequence: 
        Canny Edge Detection -> ROI Masking -> Hough Transform & Line Drawing.

        Args:
            image (np.ndarray): The original RGB input image/frame.
            region_of_interest (np.ndarray): Vertices for the polygon mask.

        Returns:
            np.ndarray: The processed image with detected lanes.
        """
        canny_image = self.to_canny_image(image)
        masked_image = self.mask_image(canny_image, region_of_interest)
        return self.hough_line(masked_image)
