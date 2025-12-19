# Lane Detection & Autonomous Driving System

A comprehensive computer vision and deep learning project designed to simulate self-driving car capabilities. This system consists of two main modules: a classic computer vision pipeline for detecting lane lines and a deep learning behavioral cloning engine for autonomous steering control using the NVIDIA CNN architecture.

## 🎥 Demos

### 1. Lane Detection (Computer Vision)
![Lane Detection Demo](asset/demo/hough_line_detection.gif)
<br>
*Real-time lane detection using Canny Edge Detection and Hough Transform on video footage.*

### 2. Autonomous Driving (Deep Learning)
![Autonomous Driving Demo](asset/demo/autonomous_drive.gif)
<br>
*The vehicle driving autonomously in the Udacity Simulator using a trained Keras model.*

---

## 🧠 Neural Network Architecture (NVIDIA Model)

For the autonomous driving module (`drive.py`), this project utilizes the Convolutional Neural Network (CNN) architecture proposed by the **NVIDIA Autonomous Driving Team** in their paper *"End to End Learning for Self-Driving Cars"*.

This model is designed to map raw pixels from a single front-facing camera directly to steering commands.

### Architecture Breakdown:
The network consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers:

1.  **Input Layer**: Image shape `(66, 200, 3)` in **YUV color space**.
2.  **Normalization**: Hard-coded within the model to center the data.
3.  **Convolutional Layers**:
    * 24 filters, 5x5 kernel, 2x2 stride (ReLU)
    * 36 filters, 5x5 kernel, 2x2 stride (ReLU)
    * 48 filters, 5x5 kernel, 2x2 stride (ReLU)
    * 64 filters, 3x3 kernel (no stride) (ReLU)
    * 64 filters, 3x3 kernel (no stride) (ReLU)
4.  **Flatten Layer**
5.  **Fully Connected Layers**:
    * Dense (100 neurons)
    * Dense (50 neurons)
    * Dense (10 neurons)
6.  **Output Layer**: Single neuron (Steering Angle).

---

## 📂 Project Structure

```text
AUTONOMOUS-VEHICLE/
│
├── asset/
│   ├── data/
│   │   ├── IMG/                       # Raw training images from simulator
│   │   └── driving_log.csv            # Telemetry data (steering angles, throttle)
│   │
│   ├── demo/
│   │   ├── autonomous_drive.gif       # GIF result for autonomous mode
│   │   └── hough_line_detection.gif   # GIF result for lane detection
│   │
│   ├── image/                         # Static test images & traffic signs
│   │   ├── hough_voting.gif
│   │   ├── test_image.jpg
│   │   └── ... (Traffic sign data)
│   │
│   └── video/
│       └── test2.mp4                  # Input video for Lane Detection
│
├── engine/
│   └── vision/
│       ├── detector.py                # Logic class (Canny, Hough, Masking)
│       └── loader.py                  # Utility class for Video I/O
│
├── notebook/
│   ├── autonomous_car_model_training.ipynb  # Jupyter Notebook for training model.h5
│   └── Road_Sign_Classification.ipynb       # Experiment notebook for sign classification
│
├── drive.py                   # Entry point for Autonomous Driving module (FastAPI)
├── main.py                    # Entry point for Lane Detection module
├── model.h5                   # Pre-trained Keras model (NVIDIA Architecture)
├── requirements.txt           # Python dependencies
├── readme.md
├── LICENSE
└── .gitignore
```

## 🚀 Modules Overview

### Module A: Lane Line Detector
Uses classical computer vision techniques to identify lane boundaries.
* **Entry Point:** `main.py`
* **Core Logic:** `engine/vision/detector.py`
* **Techniques:** Grayscale, Gaussian Blur, Canny Edge Detection, Region of Interest (ROI) Masking, Hough Transform, Slope Averaging.

### Module B: Autonomous Driver (Simulator)
Connects to the Udacity Self-Driving Car Simulator to drive the vehicle.
* **Entry Point:** `drive.py`
* **Techniques:** Deep Learning (Keras/TensorFlow), Image Preprocessing (Crop, YUV, Resize), Socket.IO communication, FastAPI (Asynchronous Server).

### Module C: Model Training
* **Notebook:** `notebook/autonomous_car_model_training.ipynb`
* **Process:** Loads data from `asset/data/`, augments images, trains the NVIDIA architecture, and saves the output as `model.h5`.



## 🛠️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Richterite/autonomous-vehicle/blob/main/readme.md
   cd autonomous-vehicle
   ```
2. **Create Virtual Environment**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Mac/Linux
   source venv/bin/activate
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Prerequisites for Autonomous Mode**
   * Download the [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim).
   * Ensure `model.h5` is present in the root directory.


## 💻 Usage

### 1. Running Lane Detection
To detect lanes on the test video:

1. Ensure `asset/video/test2.mp4` exists.
2. Run the script:
   ```bash
   python main.py
   ```
3. press `q` to close the window.

### 2. Running Lane Detection
To let the AI drive in the simulator:
1. Start the Python server:
   ```bash
   python drive.py
   ```
2. Open the **Udacity Simulator**.
3. Select **Autonomous Mode**.
4. The car will automatically connect and start driving based on `model.h5` predictions.

## 🔧 Technical Details (Preprocessing)
Before the image is fed into the Neural Network in `drive.py`, it undergoes:
1. **Cropping**: Removes sky (top 60px) and hood (bottom 25px).
2. **Color Conversion**: RGB to YUV (NVIDIA standard).
3. **Blur**: Gaussian Blur (3x3).
4. **Resize**: Downscaled to 200x66 pixels.
5. **Normalization**: Pixel values scaled to 0 - 1.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📝 License
This project is open-source and available under the [MIT License](LICENSE).
