from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
class FrameLoader:
    """
    A Class use for loading certain image/frame from different source, namely video or file image.

    """

    def __init__(self, path: str) -> None:
        self.ori_image = cv2.imread(path)
        self.gradient_image = self.to_gradient_image()
        self.height, self.width = self.ori_image.shape[:2]

    def show_image(self) -> None:
        """
        Method to show the image, including it's axis value.
        This can be use to pinpoint the shape of the mask.
        """
        plt.imshow(self.ori_image)
        plt.show()

    def copy_image(self):
        """
        Method to create copy of the original image, to prevent the image mutation if some operation is applied to the image.
        """
        return np.copy(self.ori_image)

    
    def to_gradient_image(self):
        """
        Convert image to grayscale then applying gaussian blur into the image can smoothen out the image, although canny function still use blurring and 5 x 5 kernel while in action but using blur first can improve the result.

        """
        # Converting image to grayscale can improve edge detection by algortihm due to single color channel that can maximize the each pixel gradient difference
        gray = cv2.cvtColor(self.copy_image(), cv2.COLOR_RGB2GRAY)

        # 5 x 5 kernel is good choice, eventhough you can still choose other kernel size
        # The standard deviation (param 3) is set to 0
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        # threshold1: lower bound threshold
        # threshold2: upper bound threshold
        ## if the value is greater than upper bound threshold, than it's accepted as edge pixel. 
        ## if the value is below than lower bound threshold, than it's rejected as edge pixel. 
        ## if the value in between both threshold, it's accepted only if it's connected with high gradient edge
        ### Note: the best ratio between lower and upper are 1:2 or 1:3
        canny = cv2.Canny(blur, 50, 150)

        return canny


        


