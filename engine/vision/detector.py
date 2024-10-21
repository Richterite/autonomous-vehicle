from loader import FrameLoader
import numpy as np
from cv2 import cv2


class Detector(FrameLoader):
    """
    A class use for detecting edges
    """
    def __init__(self, path_source: str) -> None:
        super().__init__(path=path_source)

        # which region in the image we want to focus. In this case, the road are gonna be the main focus. 
        # Create imaginary triangle shape, imitating the road want to focus.
        # you can pinpoint first which three point that can make up the criteria, by using show_image method in FrameLoader class
        self.region_of_interest = np.array([
            [(200, self.height), (1100, self.height), (550, 250)]
            ])

    def mask_image(self, region: np.array):
        # REMEMBER: Use grayscaled image, not the original because of the difference in the channel (single channel is used in this case)
        mask = np.zeros_like(self.gradient_image)
        cv2.fillPoly(mask, region, 255)
        masked = cv2.bitwise_and(self.gradient_image, mask)
        return masked





        