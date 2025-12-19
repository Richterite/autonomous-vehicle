import os
import cv2
import numpy as np
from engine.vision.loader import FrameLoader


def get_video_dimensions(video_path: str):
    """
    Helper function to get video width and height before processing.
    This ensures our ROI matches the video resolution.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def main():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(base_dir, 'asset', 'video', 'test2.mp4')
    # Optional, if want to save the result (Uncomment if needed)
    # output_path = os.path.join(base_dir, 'output_frames')

    width, height = get_video_dimensions(video_path)

    if width is None:
        print(f"Error: Video not found at {video_path}")
        return

    print(f"Processing video: {video_path}")
    print(f"Resolution: {width}x{height}")

    # Region of Interest
    roi_vertices = np.array([[
        (200, height),       # Bottom Left
        (1100, height),      # Bottom Right
        (550, 250)           # Top Apex
    ]], dtype=np.int32)

    # FrameLoader initialize
    loader = FrameLoader(window_name='Lane Detection Result',
                         window_width=800,
                         window_height=600)

    try:
        loader.load_and_detect(
            video_path=video_path,
            region_of_interest=roi_vertices,
            save_to_path=None  # Change to 'output_file' if need to save the result
        )
        print("Processing finished.")
    except KeyboardInterrupt:
        print("\nProcess stopped by user.")

    except Exception as e:  # pylint: disable=broad-except
        print(f"An error occurred during execution: {e}")


if __name__ == "__main__":
    main()
