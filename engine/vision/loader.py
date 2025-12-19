import os
import datetime
from typing import Optional

from cv2 import cv2
import numpy as np
from detector import Detector


class FrameLoader:
    """
    A class used for loading and processing video frames or images.

    It manages the display window and coordinates with the Detector class 
    to process each frame retrieved from a video source.
    """

    def __init__(self, path: str = None, window_name: str = 'result',
                 window_width: int = 800, window_height: int = 600) -> None:
        """
        Initializes the FrameLoader with window settings and a detector instance.

        Args:
            path (str, optional): Path to a default image (if applicable). Defaults to None.
            window_name (str): The title of the display window. Defaults to 'Result'.
            window_width (int): The width of the window in pixels. Defaults to 800.
            window_height (int): The height of the window in pixels. Defaults to 600.
        """
        # Load an initial image if path is provided (optional usage)
        self.ori_image = cv2.imread(path) if path else None

        self.window_name = window_name
        self.window_width = window_width
        self.window_height = window_height

        # Instantiate the detector logic
        self.detector = Detector()

        # Initialize the window immediately so resizing works
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.set_window_size(window_width, window_height)

    def load_and_detect(self, video_path: str, region_of_interest, save_to_path: Optional[str] = None) -> None:
        """
        Loads a video file, processes each frame to detect lines, and displays the result.

        Args:
            video_path (str): The file path to the video source.
            region_of_interest (np.ndarray): The polygon vertices used for masking the road area.
            save_to_path (str | None, optional): Directory path to save processed frames. 
                                                 If None, frames are not saved. Defaults to None.
        """
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            _, frame = cap.read()
            if frame is None:
                break
            detected_line_image = self.detector.detect_lines(
                frame, region_of_interest)
            if save_to_path:
                self._save_frame(cap, detected_line_image, save_to_path)

            cv2.imshow(self.window_name, detected_line_image)
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            print(frame_number)
            if cv2.waitkey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def set_window_size(self, window_width: int, window_height: int) -> bool:
        """
        Resizes the display window to the specified dimensions.

        Args:
            window_width (int): Desired width.
            window_height (int): Desired height.

        Returns:
            bool: True if resizing was successful, False otherwise.
        """
        try:
            cv2.resizeWindow(self.window_name, window_width, window_height)
            self.window_width = window_width
            self.window_height = window_height
            return True
        except cv2.error:
            return False

    def _save_frame(self, cap: cv2.VideoCapture, image: np.ndarray, save_path: str) -> None:
        """
        Internal helper method to save the current frame to the disk.

        Args:
            cap (cv2.VideoCapture): The video capture object (to get frame count).
            image (np.ndarray): The image data to save.
            save_path (str): The directory to save the file in.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{current_date}_frame-{frame_number}.jpg"
        full_path = os.path.join(save_path, filename)

        cv2.imwrite(full_path, image)
